"""
evaluate_model.py
=================
Loads the best saved model and produces a full evaluation report:
  • Accuracy / Precision / Recall / F1 per class
  • Top-5 confused pairs
  • Feature importance (for tree-based models)
  • Confidence distribution

Usage:
    python evaluate_model.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, top_k_accuracy_score
)

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import (
    load_data, clean_data, engineer_features,
    prepare_splits, CROP_NAMES
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.join(os.path.dirname(__file__), "..")
DATA       = os.path.join(BASE, "data/crop_data.csv")
MODELS     = os.path.join(BASE, "models")
MODEL_PATH = os.path.join(MODELS, "best_model.pkl")
SCALER_PATH= os.path.join(MODELS, "scaler.pkl")
META_PATH  = os.path.join(MODELS, "model_metadata.json")


def load_artefacts():
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"[INFO] Loaded model  : {meta['best_model_name']}")
    print(f"[INFO] Features used : {len(meta['feature_names'])}")
    return model, scaler, meta


def evaluate(model, X_test, y_test, crop_names: dict):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print("\n" + "="*60)
    print("FULL EVALUATION REPORT")
    print("="*60)

    # ── Overall metrics ──
    print(f"\n  Overall Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print(f"  Macro Precision   : {report_dict['macro avg']['precision']:.4f}")
    print(f"  Macro Recall      : {report_dict['macro avg']['recall']:.4f}")
    print(f"  Macro F1-Score    : {report_dict['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1-Score : {report_dict['weighted avg']['f1-score']:.4f}")

    # ── Top-3 Accuracy ──
    try:
        proba = model.predict_proba(X_test)
        top3  = top_k_accuracy_score(y_test, proba, k=3)
        print(f"  Top-3 Accuracy    : {top3:.4f}  ({top3*100:.2f}%)")
    except Exception:
        pass

    # ── Per-class report (worst performers) ──
    print("\n── Bottom 10 crops by F1-Score ────────────────────────")
    per_class = {
        label: {
            "crop"     : crop_names.get(str(label), f"crop_{label}"),
            "precision": report_dict[str(label)]["precision"],
            "recall"   : report_dict[str(label)]["recall"],
            "f1"       : report_dict[str(label)]["f1-score"],
            "support"  : report_dict[str(label)]["support"],
        }
        for label in sorted(set(y_test))
        if str(label) in report_dict
    }
    worst = sorted(per_class.values(), key=lambda x: x["f1"])[:10]
    print(f"  {'Crop':<22} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>6}")
    print("  " + "-"*50)
    for r in worst:
        print(f"  {r['crop']:<22} {r['precision']:>6.3f} {r['recall']:>6.3f} "
              f"{r['f1']:>6.3f} {r['support']:>6}")

    # ── Confusion: Top-5 most confused pairs ──
    print("\n── Top 5 Most Confused Pairs ──────────────────────────")
    cm = confusion_matrix(y_test, y_pred)
    np.fill_diagonal(cm, 0)
    confused = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j] > 0:
                confused.append((cm[i,j],
                    crop_names.get(str(i), f"crop_{i}"),
                    crop_names.get(str(j), f"crop_{j}")))
    confused.sort(reverse=True)
    for cnt, true_c, pred_c in confused[:5]:
        print(f"  True={true_c:<18} → Pred={pred_c:<18}  ({cnt} times)")

    return acc, report_dict


def print_feature_importance(model, feature_names: list):
    if not hasattr(model, "feature_importances_"):
        print("\n[INFO] Feature importance not available for this model type.")
        return

    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]

    print("\n── Top-15 Feature Importances ─────────────────────────")
    print(f"  {'Feature':<25} {'Importance':>12}")
    print("  " + "-"*40)
    for i in idx[:15]:
        bar = "█" * int(importances[i] * 200)
        print(f"  {feature_names[i]:<25} {importances[i]:>12.4f}  {bar}")


def print_confidence_analysis(model, X_test, y_test, crop_names: dict):
    if not hasattr(model, "predict_proba"):
        return
    proba = model.predict_proba(X_test)
    top_conf = np.max(proba, axis=1)

    print("\n── Confidence Distribution ────────────────────────────")
    buckets = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels  = ["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    for lo, hi, lbl in zip(buckets[:-1], buckets[1:], labels):
        cnt = np.sum((top_conf >= lo) & (top_conf < hi))
        pct = cnt / len(top_conf) * 100
        print(f"  {lbl}: {cnt:5d} samples  ({pct:.1f}%)")


if __name__ == "__main__":
    # Load artefacts
    model, scaler, meta = load_artefacts()

    # Rebuild test set
    df = load_data(DATA)
    df = clean_data(df)
    df = engineer_features(df)
    _, X_test, _, y_test, _, feats = prepare_splits(df)
    X_test_s = scaler.transform(X_test)  # re-apply saved scaler

    # Note: prepare_splits re-fits scaler. Use saved scaler instead.
    # Override with the loaded scaler
    from data_preprocessing import ENGINEERED_FEATURES
    X_raw = df[ENGINEERED_FEATURES].values
    from sklearn.model_selection import train_test_split
    _, X_test_raw, _, y_test = train_test_split(
        X_raw, df["crop"].values,
        test_size=0.2, stratify=df["crop"].values, random_state=42
    )
    X_test_final = scaler.transform(X_test_raw)

    crop_names = meta["crop_names"]

    acc, report = evaluate(model, X_test_final, y_test, crop_names)
    print_feature_importance(model, meta["feature_names"])
    print_confidence_analysis(model, X_test_final, y_test, crop_names)

    print("\n" + "="*60)
    print(f"[✓] Evaluation complete. Best model accuracy: {acc*100:.2f}%")
    print("="*60 + "\n")