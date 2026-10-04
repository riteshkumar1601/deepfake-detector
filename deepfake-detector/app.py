# ============================================================
# DEEPFAKE DETECTOR — app.py
# Single entry point. Run with: python app.py
#
# What it does, start to finish:
#   1. Loads v2 CNN + all classical models + SHAP explainer + face detector
#   2. Scans test_images/ for every .jpg/.jpeg/.png
#   3. For each image: crops face, runs full ensemble, computes risk score,
#      SHAP explanation, saliency map, frequency spectrum
#   4. Builds ONE combined PDF report covering every tested image
#   5. Saves it to outputs/reports/
# ============================================================

import os, sys, datetime
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import numpy as np
import pandas as pd
import cv2
import torch
import timm
import shap
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, just saving images
import matplotlib.pyplot as plt
import torchvision.transforms as T

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

from src.models.classical import load_all
from src.features.extractors import extract_all_features
from src.preprocessing.pipeline import load_and_preprocess

# ============================================================
# SECTION 1 — SETUP (runs once, at startup)
# ============================================================
print("="*60); print("DEEPFAKE DETECTOR — STARTING UP"); print("="*60)

device = torch.device("cpu")
print(f"Device: {device}")

OUT_DIR = os.path.join(BASE, "outputs")
VISUALS_DIR = os.path.join(OUT_DIR, "visuals")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")
TEST_IMAGES_DIR = os.path.join(BASE, "test_images")
os.makedirs(VISUALS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

# --- Classical models ---
print("\nLoading classical models...")
classical_dir = os.path.join(BASE, "checkpoints", "photographic", "classical")
models, scaler, feature_cols, label_map = load_all(classical_dir)
print("✅ Classical models:", list(models.keys()))

# --- Face cascade (local file, not the broken package path) ---
cascade_path = os.path.join(BASE, "models", "haarcascade_frontalface_default.xml")
if not os.path.isfile(cascade_path):
    raise RuntimeError(f"❌ Cascade file missing at {cascade_path}. Run download_cascade.py once first.")
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError(f"❌ Cascade failed to load from {cascade_path}")
print("✅ Face detector loaded")

# --- CNN architecture (must match training exactly) ---
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
print("✅ CNN v2 loaded")

shap_explainer = shap.TreeExplainer(models["random_forest"])
print("✅ SHAP explainer ready")

spatial_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# SECTION 2 — CORE ANALYSIS FUNCTIONS
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
        "Classical Model Score": classical_fake_prob * 100,
        "CNN Model Score": cnn_fake_prob * 100,
        "Frequency Anomaly": freq_anomaly, "Noise Anomaly": noise_anomaly, "Wavelet Anomaly": wavelet_anomaly,
    }
    weights = {"Classical Model Score": 0.30, "CNN Model Score": 0.45,
               "Frequency Anomaly": 0.10, "Noise Anomaly": 0.08, "Wavelet Anomaly": 0.07}
    final_risk = sum(sub_scores[k] * weights[k] for k in weights)
    return round(final_risk, 1), sub_scores

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

def analyze_image(filepath):
    """Runs the full pipeline on one image. Returns a result dict and saves visuals."""
    base_name = os.path.splitext(os.path.basename(filepath))[0]

    raw_img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    cropped_img, face_found = detect_and_crop_face(raw_img)

    temp_path = os.path.join(VISUALS_DIR, f"_temp_{base_name}.jpg")
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

    # ---- Visual 1: 4-panel image analysis ----
    panel_path = os.path.join(VISUALS_DIR, f"panel_{base_name}.png")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(raw_img); axes[0].set_title("Original", fontsize=10); axes[0].axis("off")
    axes[1].imshow(img_rgb); axes[1].set_title(f"Face Crop ({'found' if face_found else 'NOT found'})", fontsize=10); axes[1].axis("off")
    axes[2].imshow(img_rgb); axes[2].imshow(saliency, cmap="jet", alpha=0.5)
    axes[2].set_title("Model Attention (Saliency)", fontsize=10); axes[2].axis("off")
    axes[3].imshow(compute_fft_image(gray), cmap="viridis"); axes[3].set_title("Frequency Spectrum", fontsize=10); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(panel_path, dpi=120)
    plt.close()

    # ---- Visual 2: risk gauge ----
    gauge_path = os.path.join(VISUALS_DIR, f"gauge_{base_name}.png")
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh([0], [40], color="#2E7D5B", left=0)
    ax.barh([0], [30], color="#E8A33D", left=40)
    ax.barh([0], [30], color="#A4331F", left=70)
    ax.axvline(risk_score, color="black", linewidth=3)
    ax.text(risk_score, 0.7, f"{risk_score}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100); ax.set_yticks([]); ax.set_xlabel("Deepfake Risk Score (0=Low, 100=High)")
    plt.tight_layout()
    plt.savefig(gauge_path, dpi=120)
    plt.close()

    # ---- Visual 3: SHAP feature chart ----

    shap_path = os.path.join(VISUALS_DIR, f"shap_{base_name}.png")
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
        "filepath": filepath, "filename": os.path.basename(filepath),
        "face_detected": face_found, "verdict": verdict,
        "confidence_pct": round(confidence * 100, 1), "risk_score": risk_score,
        "classical_ensemble_prob": round(float(classical_ensemble_prob), 4),
        "cnn_prob": round(float(cnn_fake_prob), 4),
        "sub_scores": sub_scores, "top_features": top_features,
        "panel_path": panel_path, "gauge_path": gauge_path, "shap_path": shap_path,
    }

# ============================================================
# SECTION 3 — PDF REPORT GENERATION
# ============================================================

def build_pdf_report(results, pdf_path):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=20, textColor=colors.HexColor("#0F2A4A"))
    heading_style = ParagraphStyle("HeadingCustom", parent=styles["Heading2"], textColor=colors.HexColor("#0D4D5C"))
    normal_style = styles["Normal"]
    verdict_style_fake = ParagraphStyle("VerdictFake", parent=styles["Heading1"], textColor=colors.HexColor("#A4331F"))
    verdict_style_real = ParagraphStyle("VerdictReal", parent=styles["Heading1"], textColor=colors.HexColor("#2E7D5B"))

    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                             topMargin=0.6*inch, bottomMargin=0.6*inch,
                             leftMargin=0.6*inch, rightMargin=0.6*inch)
    story = []

    # --- Cover / summary page ---
    story.append(Paragraph("Deepfake Detection Analysis Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    story.append(Paragraph(f"Model: Dual-Branch CNN v2 (Stable Diffusion + AiGenImage fine-tuned) + "
                            f"Classical Ensemble (LogReg/RF/SVM/XGBoost)", normal_style))
    story.append(Spacer(1, 16))
    story.append(Paragraph("Summary of All Tested Images", heading_style))
    story.append(Spacer(1, 8))


    cell_style = ParagraphStyle("TableCell", parent=styles["Normal"], fontSize=8, leading=10)
    header_cell_style = ParagraphStyle("TableHeader", parent=styles["Normal"], fontSize=9,
                                        leading=11, textColor=colors.white, fontName="Helvetica-Bold")

    table_data = [[Paragraph(h, header_cell_style) for h in
                   ["Filename", "Verdict", "Confidence", "Risk Score", "Face Detected"]]]
    for r in results:
        table_data.append([
            Paragraph(r["filename"], cell_style),
            Paragraph(r["verdict"], cell_style),
            Paragraph(f"{r['confidence_pct']}%", cell_style),
            Paragraph(f"{r['risk_score']}/100", cell_style),
            Paragraph("Yes" if r["face_detected"] else "No", cell_style),
        ])
    t = Table(table_data, colWidths=[2.6*inch, 1.3*inch, 0.9*inch, 0.9*inch, 0.9*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F2A4A")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F9FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(PageBreak())

    # --- One detailed section per image ---
    for r in results:
        story.append(Paragraph(f"Image: {r['filename']}", heading_style))
        story.append(Spacer(1, 6))

        verdict_para = Paragraph(r["verdict"], verdict_style_fake if r["verdict"] == "AI-GENERATED" else verdict_style_real)
        story.append(verdict_para)
        story.append(Paragraph(f"Confidence: {r['confidence_pct']}%", normal_style))
        story.append(Spacer(1, 8))

        story.append(RLImage(r["panel_path"], width=6.8*inch, height=1.78*inch))
        story.append(Spacer(1, 10))
        story.append(RLImage(r["gauge_path"], width=5.5*inch, height=1.1*inch))
        story.append(Spacer(1, 10))
        story.append(RLImage(r["shap_path"], width=5.5*inch, height=2.75*inch))
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
    print(f"✅ PDF report saved: {pdf_path}")

# ============================================================
# SECTION 4 — MAIN
# ============================================================

if __name__ == "__main__":
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print(f"\n⚠️ No images found in {TEST_IMAGES_DIR}. Add some and rerun.")
        sys.exit(0)

    print(f"\nFound {len(image_files)} image(s) to analyze\n")
    results = []
    for fname in image_files:
        fpath = os.path.join(TEST_IMAGES_DIR, fname)
        print(f"--- {fname} ---")
        result = analyze_image(fpath)
        print(f"  Verdict: {result['verdict']}  (confidence: {result['confidence_pct']}%)  "
              f"Risk: {result['risk_score']}/100  Face detected: {result['face_detected']}")
        results.append(result)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(REPORTS_DIR, f"Deepfake_Detection_Report_{timestamp}.pdf")
    build_pdf_report(results, pdf_path)

    summary_df = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_path") and k != "sub_scores" and k != "top_features"} for r in results])
    summary_df.to_csv(os.path.join(OUT_DIR, "batch_test_summary.csv"), index=False)

    print("\n" + "="*60)
    print(f"🎉 DONE — {len(results)} image(s) analyzed")
    print(f"   PDF report: {pdf_path}")
    print(f"   CSV summary: {os.path.join(OUT_DIR, 'batch_test_summary.csv')}")
    print("="*60)