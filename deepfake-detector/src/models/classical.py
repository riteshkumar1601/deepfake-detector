
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, roc_curve)
import xgboost as xgb
import joblib, json, os

NON_FEATURE_COLS = ["filepath", "label", "source", "category", "original_split"]
LABEL_MAP = {"real": 0, "fake": 1}

def get_feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS]

def prepare_xy(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].map(LABEL_MAP).values
    return X, y

def compute_eer(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)

def evaluate_model(model, X, y, model_name="model", split_name="split", use_scaled=None):
    X_eval = use_scaled if use_scaled is not None else X
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    return {
        "model": model_name, "split": split_name,
        "accuracy": accuracy_score(y, y_pred), "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred), "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob), "eer": compute_eer(y, y_prob),
        "n_samples": len(y),
    }

def train_all_models(X_train, y_train, X_train_scaled):
    models = {}
    logreg = LogisticRegression(max_iter=2000, random_state=42)
    logreg.fit(X_train_scaled, y_train)
    models["logreg"] = logreg
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    models["random_forest"] = rf
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    models["svm"] = svm
    xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   eval_metric="logloss", n_jobs=-1, random_state=42)
    xgb_model.fit(X_train, y_train)
    models["xgboost"] = xgb_model
    return models

def needs_scaled_input(model_name):
    return model_name in ("logreg", "svm")

def save_all(models, scaler, feature_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(out_dir, f"{name}.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump({"feature_columns": feature_cols, "label_map": LABEL_MAP}, f, indent=2)

def load_all(out_dir):
    with open(os.path.join(out_dir, "metadata.json")) as f:
        meta = json.load(f)
    models = {}
    for name in ["logreg", "random_forest", "svm", "xgboost"]:
        path = os.path.join(out_dir, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler = joblib.load(os.path.join(out_dir, "scaler.pkl"))
    return models, scaler, meta["feature_columns"], meta["label_map"]

def ensemble_predict_proba(models, X, X_scaled):
    probs = []
    for name, model in models.items():
        inp = X_scaled if needs_scaled_input(name) else X
        probs.append(model.predict_proba(inp)[:, 1])
    return np.mean(probs, axis=0)
