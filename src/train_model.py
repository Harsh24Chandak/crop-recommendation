"""
train_model.py  —  FINAL VERSION
Trains Random Forest + XGBoost + Neural Network + Price Predictor.
Run: python train_model.py
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from sklearn.ensemble      import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from xgboost               import XGBClassifier
from sklearn.metrics       import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import (
    load_data, clean_data, engineer_features,
    prepare_splits, CROP_NAMES, ENGINEERED_FEATURES
)

BASE   = os.path.join(os.path.dirname(__file__), "..")
DATA   = os.path.join(BASE, "data/dataset.csv")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=20,
            min_samples_split=5, n_jobs=-1,
            random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            solver="adam", max_iter=300, early_stopping=True,
            validation_fraction=0.1, random_state=42,
            learning_rate_init=0.001
        ),
    }

def train_all(X_train, X_test, y_train, y_test):
    results = {}
    for name, model in get_models().items():
        print(f"\n-- Training {name} --")
        t0    = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        y_pred  = model.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        rep     = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            "model": model,
            "accuracy":      round(acc,                                4),
            "macro_f1":      round(rep["macro avg"]["f1-score"],       4),
            "macro_prec":    round(rep["macro avg"]["precision"],      4),
            "macro_recall":  round(rep["macro avg"]["recall"],         4),
            "train_time_s":  round(elapsed, 2),
        }
        print(f"   Accuracy: {acc*100:.2f}%  |  Macro F1: {rep['macro avg']['f1-score']:.4f}  |  Time: {elapsed:.1f}s")
    return results

if __name__ == "__main__":
    print("\n" + "="*55)
    print("AI CROP RECOMMENDATION — TRAINING")
    print("="*55)

    # 1. Load + preprocess
    df = load_data(DATA)
    df = clean_data(df)
    df = engineer_features(df)

    scaler_path = os.path.join(MODELS, "scaler.pkl")
    X_train, X_test, y_train, y_test, scaler, feats = prepare_splits(
        df, scaler_path=scaler_path
    )

    # 2. Train classifiers
    results   = train_all(X_train, X_test, y_train, y_test)
    best_name = max(results, key=lambda k: results[k]["macro_f1"])

    # 3. Save all models
    for name, info in results.items():
        joblib.dump(info["model"], os.path.join(MODELS, f"{name.lower()}_model.pkl"))
    joblib.dump(results[best_name]["model"], os.path.join(MODELS, "best_model.pkl"))

    # 4. Train price predictor (uses market_price column)
    print("\n-- Training Price Predictor --")
    X_full  = scaler.transform(df[ENGINEERED_FEATURES].values)
    y_price = df["market_price"].values
    price_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    price_model.fit(X_full, y_price)
    joblib.dump(price_model, os.path.join(MODELS, "price_model.pkl"))
    print("   Price model saved.")

    # 5. Save metadata
    meta = {
        "best_model_name": best_name,
        "feature_names":   feats,
        "crop_names":      {str(k): v for k, v in CROP_NAMES.items()},
        "model_comparison": {
            n: {k: v for k, v in info.items() if k != "model"}
            for n, info in results.items()
        }
    }
    with open(os.path.join(MODELS, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 6. Print comparison table
    print("\n" + "="*65)
    print("MODEL COMPARISON")
    print("="*65)
    print(f"{'Model':<18} {'Accuracy':>10} {'Macro F1':>10} {'Time(s)':>8}")
    print("-"*65)
    for name, info in results.items():
        mark = " <- BEST" if name == best_name else ""
        print(f"{name:<18} {info['accuracy']:>10.4f} {info['macro_f1']:>10.4f}"
              f" {info['train_time_s']:>8.1f}{mark}")
    print("="*65)
    print(f"\n[OK] Best model: {best_name}  |  Accuracy: {results[best_name]['accuracy']*100:.2f}%\n")