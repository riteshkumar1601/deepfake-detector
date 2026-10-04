# Temporary end-to-end smoke test for the FastAPI backend (in-process, no sockets).
import os, sys, glob, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)

from fastapi.testclient import TestClient
import server  # loads all models at import time

client = TestClient(server.app)

print("== HEALTH ==", flush=True)
r = client.get("/api/health")
print("status:", r.status_code, "body:", r.text, flush=True)

print("\n== ROOT PAGE (/) ==", flush=True)
r = client.get("/")
print("status:", r.status_code, "| content-type:", r.headers.get("content-type"), "| len:", len(r.content), flush=True)

imgs = sorted(glob.glob(os.path.join(BASE, "test_images", "*.png")), key=os.path.getsize)
img_path = imgs[0]
print("\n== ANALYZE:", os.path.basename(img_path), "|", os.path.getsize(img_path), "bytes ==", flush=True)
with open(img_path, "rb") as f:
    r = client.post("/api/analyze", files=[("files", (os.path.basename(img_path), f, "image/png"))])
print("status:", r.status_code, flush=True)
if r.status_code != 200:
    print("BODY:", r.text[:3000], flush=True)
    sys.exit(1)

data = r.json()
print("report_id:", data.get("report_id"), flush=True)
print("download_url:", data.get("download_url"), flush=True)
print("summary:", json.dumps(data.get("summary")), flush=True)
for res in data["results"]:
    print(" -", res["filename"], "|", res["verdict"], "| conf", res.get("confidence_pct"),
          "| risk", res.get("risk_score"), "| face", res.get("face_detected"),
          "| error:", res.get("error"), flush=True)
    print("   visuals:", res.get("visuals"), flush=True)

rid = data["report_id"]
print("\n== DOWNLOAD REPORT ==", flush=True)
r = client.get(f"/api/download-report/{rid}")
print("status:", r.status_code, "| content-type:", r.headers.get("content-type"),
      "| len:", len(r.content), "| disposition:", r.headers.get("content-disposition"), flush=True)

print("\n== SERVE VISUAL ==", flush=True)
visual_url = data["results"][0]["visuals"]["panel"]
r = client.get(visual_url)
print("status:", r.status_code, "| content-type:", r.headers.get("content-type"),
      "| len:", len(r.content), flush=True)

print("\n== STATIC FRONTEND ==", flush=True)
for path in ("/app.js", "/style.css"):
    r = client.get(path)
    print(path, "->", r.status_code, r.headers.get("content-type"), len(r.content), flush=True)

print("\nDONE", flush=True)
