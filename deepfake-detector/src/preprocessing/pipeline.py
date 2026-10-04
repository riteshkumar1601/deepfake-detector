
import cv2
import numpy as np

TARGET_SIZE = (256, 256)

def load_and_preprocess(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    normalized = img.astype(np.float32) / 255.0
    return {"rgb": img, "gray": gray, "normalized": normalized}
