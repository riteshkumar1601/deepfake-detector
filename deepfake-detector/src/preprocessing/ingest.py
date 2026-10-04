
import os
import pandas as pd

def scan_folder(root, label, source, category, split_tag):
    rows = []
    if not os.path.isdir(root):
        return rows
    for fname in os.listdir(root):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append({
                "filepath": os.path.join(root, fname),
                "label": label, "source": source,
                "category": category, "original_split": split_tag
            })
    return rows

def scan_by_keyword(root, source, split_tag="train"):
    rows = []
    real_kw = ["real", "human", "authentic", "photo"]
    fake_kw = ["ai", "fake", "generated", "synthetic", "gan", "diffusion"]
    unmatched = 0
    for dirpath, dirnames, filenames in os.walk(root):
        imgs = [f for f in filenames if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not imgs:
            continue
        leaf_name = os.path.basename(dirpath).lower()
        label = None
        if any(k in leaf_name for k in fake_kw):
            label = "fake"
        elif any(k in leaf_name for k in real_kw):
            label = "real"
        if label is None:
            unmatched += len(imgs)
            continue
        category = os.path.basename(dirpath)
        for f in imgs:
            rows.append({
                "filepath": os.path.join(dirpath, f),
                "label": label, "source": source,
                "category": category, "original_split": split_tag
            })
    if unmatched:
        print(f"WARNING: {source}: {unmatched} images in unrecognized folders skipped.")
    return rows

def build_manifest():
    rows = []
    gan_root = "/content/gan_local/real_vs_fake/real-vs-fake"
    for split in ["train", "valid", "test"]:
        rows += scan_folder(f"{gan_root}/{split}/real", "real", "gan", "face", split)
        rows += scan_folder(f"{gan_root}/{split}/fake", "fake", "gan", "face", split)

    diff_root = "/content/diffusion_local"
    for split in ["train", "test"]:
        rows += scan_folder(f"{diff_root}/{split}/REAL", "real", "diffusion", "cifar_object", split)
        rows += scan_folder(f"{diff_root}/{split}/FAKE", "fake", "diffusion", "cifar_object", split)

    unseen_root = "/content/unseen_test_local/real_and_fake_face_detection/real_and_fake_face"
    rows += scan_folder(f"{unseen_root}/training_real", "real", "unseen", "face", "test")
    rows += scan_folder(f"{unseen_root}/training_fake", "fake", "unseen", "face", "test")

    intel_root = "/content/intel_local"
    scene_categories = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    for dirpath, dirnames, filenames in os.walk(intel_root):
        cat = os.path.basename(dirpath).lower()
        if cat in scene_categories:
            split_tag = "train" if "train" in dirpath.lower() else "test"
            rows += scan_folder(dirpath, "real", "intel", cat, split_tag)

    rows += scan_by_keyword("/content/general_ai_local", "general_ai", "train")
    return pd.DataFrame(rows)
