# ============================================================
# DEEPFAKE DETECTOR — server.py
# FastAPI backend + frontend host. Run with: python server.py
# Then open http://localhost:8000 in a browser.
# ============================================================

import os, sys, datetime, re, shutil, time, uuid, threading
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd
import cv2
import torch
import timm
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms as T

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn

from src.models.classical import load_all
from src.features.extractors import extract_all_features
from src.preprocessing.pipeline import load_and_preprocess

# ============================================================
# SECTION 1 — MODEL LOADING (runs once, at server startup)
# ============================================================
print("="*60); print("DEEPFAKE DETECTOR SERVER — STARTING UP"); print("="*60)

device = torch.device("cpu")

def make_json_safe(obj):
    """Recursively converts numpy types to plain Python types so json.dumps never chokes."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

OUT_DIR = os.path.join(BASE, "outputs")
VISUALS_DIR = os.path.join(OUT_DIR, "visuals")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")
UPLOADS_DIR = os.path.join(OUT_DIR, "uploads")
for d in [VISUALS_DIR, REPORTS_DIR, UPLOADS_DIR]:
    os.makedirs(d, exist_ok=True)

print("\nLoading classical models...")
classical_dir = os.path.join(BASE, "checkpoints", "photographic", "classical")
models, scaler, feature_cols, label_map = load_all(classical_dir)
print("[OK] Classical models:", list(models.keys()))

cascade_path = os.path.join(BASE, "models", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError(f"[ERROR] Cascade failed to load from {cascade_path}")
print("[OK] Face detector loaded")

SPATIAL_SIZE = 224
FREQ_SIZE = 128

class FrequencyBranch(torch.nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, out_dim, 3, padding=1), torch.nn.BatchNorm2d(out_dim), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

class DualBranchCNN(torch.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.spatial = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, global_pool="avg")
        spatial_dim = self.spatial.num_features
        self.frequency = FrequencyBranch(out_dim=128)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(spatial_dim + 128, 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(256, num_classes),
        )
    def forward(self, spatial_x, freq_x):
        return self.fusion(torch.cat([self.spatial(spatial_x), self.frequency(freq_x)], dim=1))

print("\nLoading CNN v2 (best fine-tuned model)...")
cnn_path = os.path.join(BASE, "checkpoints", "photographic", "cnn_v2_multigen_face", "dual_branch_v2_best.pt")
cnn_model = DualBranchCNN(num_classes=2)
cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
cnn_model.eval()
print("[OK] CNN v2 loaded")

shap_explainer = shap.TreeExplainer(models["random_forest"])
print("[OK] SHAP explainer ready")

spatial_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# SECTION 2 — CORE ANALYSIS FUNCTIONS (same logic as app.py)
# ============================================================

def detect_and_crop_face(img_rgb, margin=0.35):
    h, w, _ = img_rgb.shape
    gray_for_detection = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_for_detection, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return img_rgb, False
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    x1 = max(0, int(x - margin * fw)); y1 = max(0, int(y - margin * fh))
    x2 = min(w, int(x + fw + margin * fw)); y2 = min(h, int(y + fh + margin * fh))
    return img_rgb[y1:y2, x1:x2], True

def compute_fft_image(gray_img, size=FREQ_SIZE):
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)
    mag_log = np.log1p(np.abs(fshift))
    mag_norm = (mag_log - mag_log.min()) / (mag_log.max() - mag_log.min() + 1e-8)
    return cv2.resize(mag_norm.astype(np.float32), (size, size))

def prepare_cnn_inputs(filepath):
    img_rgb = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    spatial_tensor = spatial_transform(cv2.resize(img_rgb, (SPATIAL_SIZE, SPATIAL_SIZE))).unsqueeze(0)
    gray = cv2.cvtColor(cv2.resize(img_rgb, (256, 256)), cv2.COLOR_RGB2GRAY)
    freq_tensor = torch.from_numpy(compute_fft_image(gray)).unsqueeze(0).unsqueeze(0)
    return spatial_tensor, freq_tensor, img_rgb, gray

def compute_saliency_map(spatial_tensor, freq_tensor):
    spatial_tensor = spatial_tensor.clone().requires_grad_(True)
    output = cnn_model(spatial_tensor, freq_tensor)
    pred_class = output.argmax(dim=1)
    score = output[0, pred_class]
    cnn_model.zero_grad()
    score.backward()
    saliency = spatial_tensor.grad.data.abs().squeeze().max(dim=0)[0].numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return cv2.resize(saliency, (SPATIAL_SIZE, SPATIAL_SIZE))

def compute_risk_score(classical_fake_prob, cnn_fake_prob, feature_vector, feature_names):
    freq_idx = [i for i, n in enumerate(feature_names) if n.startswith("fft_")]
    noise_idx = [i for i, n in enumerate(feature_names) if n.startswith("noise_")]
    dwt_idx = [i for i, n in enumerate(feature_names) if n.startswith("dwt_")]
    freq_anomaly = min(100, abs(feature_vector[freq_idx]).mean() / 5 * 100) if freq_idx else 50
    noise_anomaly = min(100, abs(feature_vector[noise_idx]).mean() * 20) if noise_idx else 50
    wavelet_anomaly = min(100, abs(feature_vector[dwt_idx]).mean() / 1000) if dwt_idx else 50
    sub_scores = {
        "Classical Model Score": float(classical_fake_prob * 100),
        "CNN Model Score": float(cnn_fake_prob * 100),
        "Frequency Anomaly": float(freq_anomaly),
        "Noise Anomaly": float(noise_anomaly),
        "Wavelet Anomaly": float(wavelet_anomaly),
    }
    weights = {"Classical Model Score": 0.30, "CNN Model Score": 0.45,
               "Frequency Anomaly": 0.10, "Noise Anomaly": 0.08, "Wavelet Anomaly": 0.07}
    final_risk = sum(sub_scores[k] * weights[k] for k in weights)
    return round(final_risk, 1), sub_scores

# def compute_risk_score(classical_fake_prob, cnn_fake_prob, feature_vector, feature_names):
#     freq_idx = [i for i, n in enumerate(feature_names) if n.startswith("fft_")]
#     noise_idx = [i for i, n in enumerate(feature_names) if n.startswith("noise_")]
#     dwt_idx = [i for i, n in enumerate(feature_names) if n.startswith("dwt_")]
#     freq_anomaly = min(100, abs(feature_vector[freq_idx]).mean() / 5 * 100) if freq_idx else 50
#     noise_anomaly = min(100, abs(feature_vector[noise_idx]).mean() * 20) if noise_idx else 50
#     wavelet_anomaly = min(100, abs(feature_vector[dwt_idx]).mean() / 1000) if dwt_idx else 50
#     sub_scores = {
#         "Classical Model Score": classical_fake_prob * 100,
#         "CNN Model Score": cnn_fake_prob * 100,
#         "Frequency Anomaly": freq_anomaly, "Noise Anomaly": noise_anomaly, "Wavelet Anomaly": wavelet_anomaly,
#     }
#     weights = {"Classical Model Score": 0.30, "CNN Model Score": 0.45,
#                "Frequency Anomaly": 0.10, "Noise Anomaly": 0.08, "Wavelet Anomaly": 0.07}
#     final_risk = sum(sub_scores[k] * weights[k] for k in weights)
#     return round(final_risk, 1), sub_scores

def extract_shap_top_features(feat_vec):
    raw_shap = shap_explainer.shap_values(feat_vec.reshape(1, -1))
    if isinstance(raw_shap, list):
        shap_vals = raw_shap[1][0]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_vals = raw_shap[0, :, 1]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
        shap_vals = raw_shap[0]
    else:
        raise ValueError(f"Unexpected SHAP output: {type(raw_shap)}")
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
    return [(feature_cols[i], float(shap_vals[i])) for i in top_idx]

def sanitize_tag(name):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name)

def analyze_image(filepath, tag):
    """tag = unique identifier used for visual filenames (prevents collisions across requests)"""
    raw_img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    cropped_img, face_found = detect_and_crop_face(raw_img)

    temp_path = os.path.join(VISUALS_DIR, f"_temp_{tag}.jpg")
    cv2.imwrite(temp_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
    working_path = temp_path if face_found else filepath

    data = load_and_preprocess(working_path)
    feats = extract_all_features(data["gray"])
    feat_vec = np.array([feats[c] for c in feature_cols], dtype=np.float32)
    feat_vec_scaled = scaler.transform(feat_vec.reshape(1, -1))

    rf_prob = models["random_forest"].predict_proba(feat_vec.reshape(1, -1))[0, 1]
    xgb_prob = models["xgboost"].predict_proba(feat_vec.reshape(1, -1))[0, 1]
    logreg_prob = models["logreg"].predict_proba(feat_vec_scaled)[0, 1]
    svm_prob = models["svm"].predict_proba(feat_vec_scaled)[0, 1]
    classical_ensemble_prob = np.mean([rf_prob, xgb_prob, logreg_prob, svm_prob])

    spatial_tensor, freq_tensor, img_rgb, gray = prepare_cnn_inputs(working_path)
    with torch.no_grad():
        cnn_output = cnn_model(spatial_tensor, freq_tensor)
        cnn_probs = torch.softmax(cnn_output, dim=1).numpy()[0]
    cnn_fake_prob = cnn_probs[1]

    final_fake_prob = 0.4 * classical_ensemble_prob + 0.6 * cnn_fake_prob
    verdict = "AI-GENERATED" if final_fake_prob >= 0.5 else "REAL"
    confidence = final_fake_prob if verdict == "AI-GENERATED" else (1 - final_fake_prob)

    risk_score, sub_scores = compute_risk_score(classical_ensemble_prob, cnn_fake_prob, feat_vec, feature_cols)
    top_features = extract_shap_top_features(feat_vec)
    saliency = compute_saliency_map(spatial_tensor, freq_tensor)

    panel_path = os.path.join(VISUALS_DIR, f"panel_{tag}.png")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(raw_img); axes[0].set_title("Original", fontsize=10); axes[0].axis("off")
    axes[1].imshow(img_rgb); axes[1].set_title(f"Face Crop ({'found' if face_found else 'NOT found'})", fontsize=10); axes[1].axis("off")
    axes[2].imshow(img_rgb); axes[2].imshow(saliency, cmap="jet", alpha=0.5)
    axes[2].set_title("Model Attention (Saliency)", fontsize=10); axes[2].axis("off")
    axes[3].imshow(compute_fft_image(gray), cmap="viridis"); axes[3].set_title("Frequency Spectrum", fontsize=10); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(panel_path, dpi=120, bbox_inches="tight")
    plt.close()

    gauge_path = os.path.join(VISUALS_DIR, f"gauge_{tag}.png")
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh([0], [40], color="#2E7D5B", left=0)
    ax.barh([0], [30], color="#E8A33D", left=40)
    ax.barh([0], [30], color="#A4331F", left=70)
    ax.axvline(risk_score, color="black", linewidth=3)
    ax.text(risk_score, 0.7, f"{risk_score}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("Deepfake Risk Score (0=Low, 100=High)")
    plt.tight_layout()
    plt.savefig(gauge_path, dpi=120, bbox_inches="tight")
    plt.close()

    shap_path = os.path.join(VISUALS_DIR, f"shap_{tag}.png")
    names = [f[0] for f in top_features][::-1]
    vals = [f[1] for f in top_features][::-1]
    bar_colors = ["#A4331F" if v > 0 else "#0D4D5C" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(names, vals, color=bar_colors)
    ax.set_xlabel("SHAP value  (→ AI-Generated   |   ← Real)", fontsize=9)
    ax.set_title("Top Contributing Features", fontsize=11)
    plt.tight_layout()
    plt.savefig(shap_path, dpi=120, bbox_inches="tight")
    plt.close()

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {
        "filename": os.path.basename(filepath), "face_detected": face_found, "verdict": verdict,
        "confidence_pct": round(confidence * 100, 1), "risk_score": risk_score,
        "classical_ensemble_prob": round(float(classical_ensemble_prob), 4),
        "cnn_prob": round(float(cnn_fake_prob), 4),
        "sub_scores": sub_scores, "top_features": top_features,
        "visuals": {
            "panel": f"/visuals/{os.path.basename(panel_path)}",
            "gauge": f"/visuals/{os.path.basename(gauge_path)}",
            "shap": f"/visuals/{os.path.basename(shap_path)}",
        },
    }

# ============================================================
# SECTION 3 — PDF REPORT GENERATION (same as app.py)
# ============================================================

def build_pdf_report(results, pdf_path):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#0F2A4A"))
    heading_style = ParagraphStyle("HeadingCustom", parent=styles["Heading2"], textColor=colors.HexColor("#0D4D5C"))
    normal_style = styles["Normal"]
    verdict_style_fake = ParagraphStyle("VerdictFake", parent=styles["Heading1"], textColor=colors.HexColor("#A4331F"))
    verdict_style_real = ParagraphStyle("VerdictReal", parent=styles["Heading1"], textColor=colors.HexColor("#2E7D5B"))
    cell_style = ParagraphStyle("TableCell", parent=styles["Normal"], fontSize=8, leading=10)
    header_cell_style = ParagraphStyle("TableHeader", parent=styles["Normal"], fontSize=9,
                                        leading=11, textColor=colors.white, fontName="Helvetica-Bold")

    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                             topMargin=0.6*inch, bottomMargin=0.6*inch,
                             leftMargin=0.6*inch, rightMargin=0.6*inch)
    story = []

    story.append(Paragraph("Deepfake Detection Analysis Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Paragraph("Model: Dual-Branch CNN v2 (Stable Diffusion + AiGenImage fine-tuned) + "
                            "Classical Ensemble (LogReg/RF/SVM/XGBoost)", normal_style))
    story.append(Spacer(1, 16))
    story.append(Paragraph("Summary of All Tested Images", heading_style))
    story.append(Spacer(1, 8))

    table_data = [[Paragraph(h, header_cell_style) for h in
                   ["Filename", "Verdict", "Confidence", "Risk Score", "Face Detected"]]]
    for r in results:
        table_data.append([
            Paragraph(r["filename"], cell_style), Paragraph(r["verdict"], cell_style),
            Paragraph(f"{r['confidence_pct']}%", cell_style), Paragraph(f"{r['risk_score']}/100", cell_style),
            Paragraph("Yes" if r["face_detected"] else "No", cell_style),
        ])
    t = Table(table_data, colWidths=[2.6*inch, 1.3*inch, 0.9*inch, 0.9*inch, 0.9*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F2A4A")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F9FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(PageBreak())

    for r in results:
        story.append(Paragraph(f"Image: {r['filename']}", heading_style))
        story.append(Spacer(1, 6))
        verdict_para = Paragraph(r["verdict"], verdict_style_fake if r["verdict"] == "AI-GENERATED" else verdict_style_real)
        story.append(verdict_para)
        story.append(Paragraph(f"Confidence: {r['confidence_pct']}%", normal_style))
        story.append(Spacer(1, 8))

        panel_full = os.path.join(VISUALS_DIR, os.path.basename(r["visuals"]["panel"]))
        gauge_full = os.path.join(VISUALS_DIR, os.path.basename(r["visuals"]["gauge"]))
        shap_full = os.path.join(VISUALS_DIR, os.path.basename(r["visuals"]["shap"]))

        story.append(RLImage(panel_full, width=6.8*inch, height=1.78*inch))
        story.append(Spacer(1, 10))
        story.append(RLImage(gauge_full, width=5.5*inch, height=1.1*inch))
        story.append(Spacer(1, 10))
        story.append(RLImage(shap_full, width=5.5*inch, height=2.75*inch))
        story.append(Spacer(1, 10))

        sub_table_data = [["Sub-Score Component", "Value (0-100)"]]
        for k, v in r["sub_scores"].items():
            sub_table_data.append([k, f"{v:.1f}"])
        sub_t = Table(sub_table_data, colWidths=[3*inch, 1.5*inch])
        sub_t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D4D5C")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(sub_t)
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "<i>This is an automated analysis based on statistical and learned patterns. "
            "It is not forensic or legal proof of image origin.</i>", normal_style
        ))
        story.append(PageBreak())

    doc.build(story)

# ============================================================
# SECTION 4 — FASTAPI APP
# ============================================================

app = FastAPI(title="Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/visuals", StaticFiles(directory=VISUALS_DIR), name="visuals")


@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": True}


# Only one analysis runs at a time. Without this, two quick clicks on
# "Start Testing" would run heavy CPU work concurrently and their output
# files could overwrite each other.
analysis_lock = threading.Lock()


@app.post("/api/analyze")
def analyze(files: List[UploadFile] = File(...)):
    # Unique per request: two requests in the same second can no longer
    # collide on upload/visual/report filenames.
    report_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    started_at = time.time()
    print(f"[ANALYZE {report_id}] received {len(files)} file(s)")
    results = []

    # Sync endpoint -> FastAPI runs it in a worker thread, so the web server
    # keeps answering (health checks, downloads) while analysis is running.
    with analysis_lock:
        for idx, file in enumerate(files):
            safe_name = sanitize_tag(os.path.splitext(file.filename)[0])
            tag = f"{report_id}_{idx}_{safe_name}"
            ext = os.path.splitext(file.filename)[1] or ".jpg"
            save_path = os.path.join(UPLOADS_DIR, f"{tag}{ext}")

            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            try:
                result = analyze_image(save_path, tag)
                results.append(result)
                print(f"[ANALYZE {report_id}] {file.filename} -> {result['verdict']} "
                      f"({result['confidence_pct']}% confidence)")
            except Exception as e:
                print(f"[ANALYZE {report_id}] {file.filename} FAILED: {e}")
                results.append({
                    "filename": file.filename, "error": str(e),
                    "verdict": "ERROR", "confidence_pct": 0, "risk_score": 0,
                    "face_detected": False, "sub_scores": {}, "top_features": [],
                    "visuals": {},
                })

        valid_results = [r for r in results if "error" not in r]
        pdf_path = os.path.join(REPORTS_DIR, f"Deepfake_Detection_Report_{report_id}.pdf")
        pdf_generated = False
        if valid_results:
            try:
                build_pdf_report(valid_results, pdf_path)
                pdf_generated = True
            except Exception as e:
                # A PDF failure must not fail the whole request — the JSON
                # results are still returned so the frontend can display them.
                print(f"[ANALYZE {report_id}] PDF generation failed: {e}")

    print(f"[ANALYZE {report_id}] finished in {time.time() - started_at:.1f}s")

    return JSONResponse(make_json_safe({
        "report_id": report_id,
        "results": results,
        "download_url": f"/api/download-report/{report_id}" if pdf_generated else None,
        "summary": {
            "total": len(results),
            "ai_generated": sum(1 for r in results if r.get("verdict") == "AI-GENERATED"),
            "real": sum(1 for r in results if r.get("verdict") == "REAL"),
            "errors": sum(1 for r in results if r.get("verdict") == "ERROR"),
        }
    }))


@app.get("/api/download-report/{report_id}")
def download_report(report_id: str):
    pdf_path = os.path.join(REPORTS_DIR, f"Deepfake_Detection_Report_{report_id}.pdf")
    if not os.path.exists(pdf_path):
        return JSONResponse({"error": "Report not found"}, status_code=404)
    return FileResponse(pdf_path, media_type="application/pdf",
                         filename=f"Deepfake_Detection_Report_{report_id}.pdf")


# ============================================================
# SECTION 5 — SERVE THE FRONTEND (same origin as the API)
# ============================================================
# Serving the UI from this server means the browser talks to ONE origin
# (http://localhost:8000). That removes all CORS setup, and — more importantly —
# it stops external dev servers (VS Code Live Server / Live Preview, etc.)
# from auto-reloading the page while the backend writes upload/visual/report
# files, which used to wipe the page state before results could be shown.

FRONTEND_DIR = next(
    (d for d in (os.path.join(os.path.dirname(BASE), "frontend"),
                 os.path.join(BASE, "frontend"))
     if os.path.isdir(d)),
    None,
)

if FRONTEND_DIR:
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    print(f"[OK] Frontend mounted — browser UI available at http://localhost:8000 (from {FRONTEND_DIR})")
else:
    print("[WARN] frontend/ folder not found — open frontend/index.html in a browser manually")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("[START] Server ready — open http://localhost:8000 in your browser")
    print("   API + frontend are served together (no Live Server / CORS needed)")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)