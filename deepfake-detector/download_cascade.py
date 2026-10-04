import urllib.request
import os

BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
target_path = os.path.join(BASE, "models", "haarcascade_frontalface_default.xml")

print(f"Downloading from {url}")
urllib.request.urlretrieve(url, target_path)
print(f"✅ Saved to {target_path}")
print(f"File size: {os.path.getsize(target_path)} bytes")