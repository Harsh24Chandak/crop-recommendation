"""
predict.py  —  FINAL VERSION (with fertilizer + price + confidence alert)
CLI usage:
  python predict.py --N 90 --P 42 --K 43 --temperature 20.9 \
      --humidity 82 --ph 6.5 --rainfall 202 --soil 0 --season 3 --water 887
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, joblib

BASE        = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR  = os.path.join(BASE, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
META_PATH   = os.path.join(MODELS_DIR, "model_metadata.json")
PRICE_PATH  = os.path.join(MODELS_DIR, "price_model.pkl")

# Singleton globals — loaded once, reused for every API call
_model = _scaler = _meta = _price_model = None

def _load():
    global _model, _scaler, _meta, _price_model
    if _model is None:
        _model       = joblib.load(MODEL_PATH)
        _scaler      = joblib.load(SCALER_PATH)
        _price_model = joblib.load(PRICE_PATH) if os.path.exists(PRICE_PATH) else None
        with open(META_PATH) as f:
            _meta = json.load(f)

# ── YIELD ESTIMATION DATABASE (kg per acre) ───────────────────────────────────
YIELD_ESTIMATES = {
    "rice": {"min": 2000, "max": 3000, "unit": "kg/acre", "duration": "120-150 days"},
    "wheat": {"min": 1500, "max": 2500, "unit": "kg/acre", "duration": "120-150 days"},
    "maize": {"min": 2000, "max": 3500, "unit": "kg/acre", "duration": "90-120 days"},
    "sugarcane": {"min": 30000, "max": 50000, "unit": "kg/acre", "duration": "12-18 months"},
    "cotton": {"min": 500, "max": 1200, "unit": "kg/acre", "duration": "150-180 days"},
    "potato": {"min": 8000, "max": 15000, "unit": "kg/acre", "duration": "90-120 days"},
    "tomato": {"min": 10000, "max": 20000, "unit": "kg/acre", "duration": "90-120 days"},
    "onion": {"min": 5000, "max": 10000, "unit": "kg/acre", "duration": "90-120 days"},
    "banana": {"min": 15000, "max": 30000, "unit": "kg/acre", "duration": "9-12 months"},
    "mango": {"min": 5000, "max": 15000, "unit": "kg/acre", "duration": "3-6 months"},
    "sugarcane": {"min": 30000, "max": 50000, "unit": "kg/acre", "duration": "12-18 months"},
    "groundnut": {"min": 800, "max": 1500, "unit": "kg/acre", "duration": "120-150 days"},
    "mustard": {"min": 500, "max": 1200, "unit": "kg/acre", "duration": "120-150 days"},
    "soybean": {"min": 800, "max": 1500, "unit": "kg/acre", "duration": "90-120 days"},
    "chickpea": {"min": 500, "max": 1200, "unit": "kg/acre", "duration": "120-150 days"},
    "pigeonpeas": {"min": 600, "max": 1200, "unit": "kg/acre", "duration": "150-180 days"},
    "lentil": {"min": 400, "max": 1000, "unit": "kg/acre", "duration": "90-120 days"},
    "blackgram": {"min": 400, "max": 800, "unit": "kg/acre", "duration": "60-90 days"},
    "mungbeans": {"min": 400, "max": 800, "unit": "kg/acre", "duration": "60-75 days"},
    "ginger": {"min": 4000, "max": 8000, "unit": "kg/acre", "duration": "8-10 months"},
    "turmeric": {"min": 5000, "max": 10000, "unit": "kg/acre", "duration": "8-9 months"},
    "coriander": {"min": 300, "max": 600, "unit": "kg/acre", "duration": "60-90 days"},
    "garlic": {"min": 2000, "max": 5000, "unit": "kg/acre", "duration": "120-150 days"},
    "cabbage": {"min": 15000, "max": 25000, "unit": "kg/acre", "duration": "90-120 days"},
    "cauliflower": {"min": 10000, "max": 20000, "unit": "kg/acre", "duration": "90-120 days"},
    "carrot": {"min": 5000, "max": 10000, "unit": "kg/acre", "duration": "90-120 days"},
    "radish": {"min": 5000, "max": 10000, "unit": "kg/acre", "duration": "45-60 days"},
    "spinach": {"min": 2000, "max": 4000, "unit": "kg/acre", "duration": "25-30 days"},
    "lettuce": {"min": 2000, "max": 4000, "unit": "kg/acre", "duration": "45-60 days"},
    "watermelon": {"min": 15000, "max": 30000, "unit": "kg/acre", "duration": "80-90 days"},
    "muskmelon": {"min": 10000, "max": 20000, "unit": "kg/acre", "duration": "70-80 days"},
    "pumpkin": {"min": 10000, "max": 25000, "unit": "kg/acre", "duration": "90-120 days"},
    "strawberry": {"min": 2000, "max": 5000, "unit": "kg/acre", "duration": "6-8 months"},
    "grapes": {"min": 8000, "max": 15000, "unit": "kg/acre", "duration": "12-18 months"},
    "pomegranate": {"min": 10000, "max": 20000, "unit": "kg/acre", "duration": "12-18 months"},
    "lemon": {"min": 5000, "max": 15000, "unit": "kg/acre", "duration": "8-10 months"},
    "orange": {"min": 5000, "max": 15000, "unit": "kg/acre", "duration": "8-12 months"},
    "papaya": {"min": 20000, "max": 50000, "unit": "kg/acre", "duration": "12-18 months"},
    "coconut": {"min": 5000, "max": 10000, "unit": "nuts/acre", "duration": "60-80 nuts/tree"},
    "coffee": {"min": 500, "max": 1500, "unit": "kg/acre", "duration": "8-10 months"},
    "tea": {"min": 1500, "max": 2500, "unit": "kg/acre", "duration": "continuous"},
    "jute": {"min": 1500, "max": 2500, "unit": "kg/acre", "duration": "120 days"},
    "tobacco": {"min": 1000, "max": 2000, "unit": "kg/acre", "duration": "90-120 days"},
    "rubber": {"min": 500, "max": 1000, "unit": "kg/acre", "duration": "continuous"},
    "sunflower": {"min": 600, "max": 1200, "unit": "kg/acre", "duration": "90-110 days"},
    "sesamum": {"min": 200, "max": 500, "unit": "kg/acre", "duration": "80-100 days"},
    "rapeseed": {"min": 500, "max": 1200, "unit": "kg/acre", "duration": "120-150 days"},
    "barley": {"min": 1500, "max": 2500, "unit": "kg/acre", "duration": "120-150 days"},
    "oat": {"min": 1000, "max": 2000, "unit": "kg/acre", "duration": "90-120 days"},
}

# ── FERTILIZER DATABASE ───────────────────────────────────────────────────────
IDEAL_NPK = {
    "rice":       {"N":120,"P":60, "K":60},
    "wheat":      {"N":150,"P":75, "K":50},
    "maize":      {"N":120,"P":60, "K":40},
    "sugarcane":  {"N":250,"P":100,"K":120},
    "cotton":     {"N":120,"P":60, "K":60},
    "jute":       {"N":60, "P":30, "K":30},
    "banana":     {"N":200,"P":60, "K":300},
    "mango":      {"N":100,"P":50, "K":100},
    "potato":     {"N":180,"P":90, "K":120},
    "tomato":     {"N":150,"P":80, "K":100},
    "chickpea":   {"N":20, "P":60, "K":40},
    "lentil":     {"N":20, "P":50, "K":30},
    "blackgram":  {"N":25, "P":50, "K":30},
    "mungbeans":  {"N":20, "P":40, "K":20},
    "pigeonpeas": {"N":20, "P":60, "K":40},
    "kidneybeans":{"N":20, "P":60, "K":40},
    "soybean":    {"N":25, "P":60, "K":40},
    "coffee":     {"N":100,"P":50, "K":100},
    "coconut":    {"N":100,"P":40, "K":200},
    "papaya":     {"N":250,"P":120,"K":250},
    "orange":     {"N":100,"P":50, "K":150},
    "grapes":     {"N":100,"P":60, "K":100},
    "watermelon": {"N":80, "P":60, "K":80},
    "muskmelon":  {"N":80, "P":60, "K":80},
    "pomegranate":{"N":100,"P":50, "K":100},
    # Additional crops
    "groundnut":  {"N":20, "P":60, "K":40},
    "mustard":    {"N":60, "P":30, "K":30},
    "sunflower":  {"N":80, "P":40, "K":40},
    "sesamum":    {"N":40, "P":25, "K":25},
    "rapeseed":   {"N":60, "P":30, "K":30},
    "barley":     {"N":80, "P":40, "K":30},
    "oat":        {"N":60, "P":30, "K":30},
    "tobacco":    {"N":100,"P":50, "K":100},
    "rubber":     {"N":50, "P":25, "K":50},
    "cashew":     {"N":60, "P":30, "K":60},
    "arecanut":   {"N":50, "P":25, "K":50},
    "tea":        {"N":150,"P":75, "K":100},
    "blackpepper":{"N":120,"P":60, "K":100},
    "cardamom":   {"N":60, "P":30, "K":60},
    "ginger":     {"N":100,"P":50, "K":80},
    "turmeric":   {"N":120,"P":60, "K":100},
    "coriander":  {"N":30, "P":20, "K":20},
    "cumin":      {"N":25, "P":20, "K":20},
    "fennel":     {"N":40, "P":25, "K":25},
    "fenugreek":  {"N":25, "P":20, "K":20},
    "garlic":     {"N":60, "P":30, "K":60},
    "onion":      {"N":60, "P":30, "K":50},
    "cauliflower":{"N":100,"P":50, "K":80},
    "cabbage":    {"N":80, "P":40, "K":60},
    "carrot":     {"N":60, "P":30, "K":50},
    "turnip":     {"N":50, "P":25, "K":40},
    "beetroot":   {"N":60, "P":30, "K":50},
    "drumstick":  {"N":60, "P":30, "K":40},
}
FERTILIZER_MAP = {
    "N": {"name": "Urea",    "ratio": 0.46, "unit": "kg/ha"},
    "P": {"name": "DAP",     "ratio": 0.18, "unit": "kg/ha"},
    "K": {"name": "MOP",     "ratio": 0.60, "unit": "kg/ha"},
}

def get_fertilizer_advice(crop_name: str, N: float, P: float, K: float) -> dict:
    ideal = IDEAL_NPK.get(crop_name.lower(), {"N":100,"P":50,"K":50})
    advice, status = [], "balanced"
    for nut, curr in [("N",N),("P",P),("K",K)]:
        deficit = ideal[nut] - curr
        fert    = FERTILIZER_MAP[nut]
        if deficit > 10:
            product_kg = round(deficit / fert["ratio"], 1)
            advice.append({
                "nutrient":    nut,
                "deficit_kg":  round(deficit, 1),
                "fertilizer":  fert["name"],
                "apply_kg_ha": product_kg,
                "message":     f"Apply {product_kg} kg/ha of {fert['name']} (adds {round(deficit,1)} kg/ha {nut})"
            })
            status = "needs_fertilizer"
        elif deficit < -20:
            advice.append({
                "nutrient": nut,
                "deficit_kg": round(deficit, 1),
                "fertilizer": None,
                "apply_kg_ha": 0,
                "message": f"Soil is HIGH in {nut} — avoid {fert['name']}"
            })
    if not advice:
        advice = [{"message": "Soil nutrients are well-balanced for this crop ✓",
                   "nutrient": None, "fertilizer": None, "apply_kg_ha": 0}]
    return {"status": status, "ideal_NPK": ideal, "advice": advice}

def get_yield_estimation(crop_name: str) -> dict:
    """Get yield estimation for a crop"""
    yield_data = YIELD_ESTIMATES.get(crop_name.lower(), {"min": 1000, "max": 2000, "unit": "kg/acre", "duration": "varies"})
    return yield_data

# ── FEATURE BUILDER ──────────────────────────────────────────────────────────
def _build_features(inputs: dict) -> np.ndarray:
    N    = float(inputs["N"])
    P    = float(inputs["P"])
    K    = float(inputs["K"])
    temp = float(inputs["temperature"])
    hum  = float(inputs["humidity"])
    ph   = float(inputs["ph"])
    rain = float(inputs["rainfall"])
    soil = int(inputs["soil"])
    seas = int(inputs["season"])
    wat  = float(inputs["water"])
    npk  = N + P + K
    raw  = np.array([[
        N, P, K, temp, hum, ph, rain, soil, seas, wat,
        npk, N/(npk+1e-6), P/(npk+1e-6), K/(npk+1e-6),
        temp*hum/100, wat-rain
    ]])
    return _scaler.transform(raw)

# ── MAIN PREDICT ─────────────────────────────────────────────────────────────
def predict(inputs: dict) -> dict:
    _load()
    X          = _build_features(inputs)
    crop_id    = int(_model.predict(X)[0])
    crop_names = _meta["crop_names"]
    crop_name  = crop_names.get(str(crop_id), f"crop_{crop_id}")

    # Confidence + top-3
    confidence, top_3 = 1.0, []
    if hasattr(_model, "predict_proba"):
        proba      = _model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        for idx in np.argsort(proba)[::-1][:3]:
            top_3.append({
                "crop":       crop_names.get(str(idx), f"crop_{idx}"),
                "confidence": round(float(proba[idx]), 4)
            })
    else:
        top_3 = [{"crop": crop_name, "confidence": 1.0}]

    # Market price
    market_price = {}
    if _price_model is not None:
        p = float(_price_model.predict(X)[0])
        market_price = {
            "estimated_price_per_quintal": int(round(p)),
            "price_range": f"Rs.{int(round(p*0.9))} - Rs.{int(round(p*1.1))}/quintal"
        }

    # Fertilizer
    fertilizer = get_fertilizer_advice(
        crop_name, inputs["N"], inputs["P"], inputs["K"]
    )

    # Confidence alert
    if confidence < 0.60:
        alert_level = "warning"
        alert_msg   = (f"Low confidence ({confidence*100:.0f}%). "
                       "Soil values are unusual. Please verify with a soil test.")
    elif confidence < 0.80:
        alert_level = "moderate"
        alert_msg   = f"Moderate confidence ({confidence*100:.0f}%). Recommendation is likely correct."
    else:
        alert_level = "safe"
        alert_msg   = f"High confidence ({confidence*100:.0f}%). Reliable recommendation."

    return {
        "crop_id":     crop_id,
        "crop_name":   crop_name,
        "confidence":  round(confidence, 4),
        "top_3":       top_3,
        "model_used":  _meta.get("best_model_name","unknown"),
        "market_price":market_price,
        "fertilizer":  fertilizer,
        "alert_level": alert_level,
        "alert_msg":   alert_msg,
    }

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for arg in ["N","P","K","temperature","humidity","ph","rainfall","water"]:
        ap.add_argument(f"--{arg}", type=float, required=True)
    ap.add_argument("--soil",   type=int, required=True)
    ap.add_argument("--season", type=int, required=True)
    args   = ap.parse_args()
    result = predict(vars(args))
    print("\n" + "="*50)
    print("  CROP RECOMMENDATION")
    print("="*50)
    print(f"  Crop      : {result['crop_name'].upper()}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  Alert     : {result['alert_msg']}")
    if result.get("market_price"):
        print(f"  Price     : {result['market_price']['price_range']}")
    print("\n  Fertilizer Advice:")
    for a in result["fertilizer"]["advice"]:
        print(f"    - {a['message']}")
    print("\n  Top-3 Alternatives:")
    for i, t in enumerate(result["top_3"], 1):
        print(f"    {i}. {t['crop']}  ({t['confidence']*100:.1f}%)")
    print("="*50)