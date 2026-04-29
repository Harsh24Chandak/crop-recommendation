"""
app.py  —  FINAL VERSION (FastAPI backend + HTML Frontend)
Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000
Swagger docs: http://localhost:8000/docs
"""
import os, sys
from pathlib import Path

# Tell Python where to find our modules (they are inside src/ folder)
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))

from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from predict  import predict as ml_predict, get_yield_estimation
from weather  import get_weather
from database import save_prediction, get_history, get_stats, delete_prediction, delete_all_predictions, get_crop_details
from data_preprocessing import CROP_NAMES, SOIL_NAMES, SEASON_NAMES

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AI Crop Recommendation API",
    description = "Predict best crop + fertilizer advice + market price.",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"], allow_credentials=True,
)

# Serve static HTML files
STATIC_DIR = BASE_DIR

@app.get("/")
async def root():
    """Serve main index.html"""
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/{page}")
async def serve_page(page: str):
    """Serve other HTML pages"""
    html_pages = {
        "analytics": "analytics.html",
        "profile": "profile.html",
        "growth": "growth.html",
    }
    if page in html_pages:
        return FileResponse(STATIC_DIR / html_pages[page])
    return FileResponse(STATIC_DIR / "index.html")

# ── Schemas ───────────────────────────────────────────────────────────────────
class CropInput(BaseModel):
    N:           float = Field(..., ge=0, le=200,  description="Nitrogen (kg/ha)")
    P:           float = Field(..., ge=0, le=150,  description="Phosphorus (kg/ha)")
    K:           float = Field(..., ge=0, le=200,  description="Potassium (kg/ha)")
    temperature: float = Field(..., ge=0, le=50,   description="Temperature (C)")
    humidity:    float = Field(..., ge=0, le=100,  description="Humidity (%)")
    ph:          float = Field(..., ge=0, le=14,   description="Soil pH")
    rainfall:    float = Field(..., ge=0, le=300,  description="Rainfall (mm)")
    soil:        int   = Field(..., ge=0, le=34,   description="Soil type code 0-34")
    season:      int   = Field(..., ge=0, le=3,    description="Season 0=Kharif 1=Rabi 2=Zaid 3=Whole")
    water:       float = Field(..., ge=0, le=3000, description="Water requirement (mm/year)")
    city:        Optional[str] = Field(None, description="City name (optional, for logs)")

    class Config:
        json_schema_extra = {"example": {
            "N":90,"P":42,"K":43,"temperature":20.9,"humidity":82.0,
            "ph":6.5,"rainfall":202.9,"soil":0,"season":3,"water":887.3,"city":"Kolhapur"
        }}

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status":"ok","timestamp":datetime.utcnow().isoformat(),"version":"2.0.0"}


@app.post("/predict", tags=["Prediction"])
def predict_crop(data: CropInput):
    """
    Main endpoint. Returns:
    - crop_name + confidence
    - top_3 alternatives
    - fertilizer advice (which fertilizer, how much kg/ha)
    - market_price estimate
    - alert_level (safe / moderate / warning)
    """
    try:
        inp    = data.dict()
        result = ml_predict(inp)
        result["timestamp"] = datetime.utcnow().isoformat()
        save_prediction(inp, result)          # save to SQLite history
        return result
    except FileNotFoundError:
        raise HTTPException(503, "Model not found. Run train_model.py first.")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/weather/{city}", tags=["Weather"])
def weather(city: str):
    """
    Returns live weather data for the given city.
    Frontend uses this to auto-fill temperature, humidity, rainfall fields.
    """
    return get_weather(city)


@app.get("/history", tags=["History"])
def history(limit: int = 20):
    """Returns the last N predictions stored in the database."""
    return {"predictions": get_history(limit), "count": limit}


@app.get("/stats", tags=["History"])
def stats():
    """Returns usage stats: total predictions, avg confidence, top crops."""
    return get_stats()


@app.get("/crops", tags=["Reference"])
def crops():
    return {"total": len(CROP_NAMES),
            "crops": [{"id":k,"name":v} for k,v in sorted(CROP_NAMES.items())]}


@app.get("/soils", tags=["Reference"])
def soils():
    return {"total": len(SOIL_NAMES),
            "soils": [{"id":k,"name":v} for k,v in sorted(SOIL_NAMES.items())]}


@app.get("/seasons", tags=["Reference"])
def seasons():
    return {"seasons": [{"id":k,"name":v} for k,v in SEASON_NAMES.items()]}


@app.delete("/history/{id}", tags=["History"])
def delete_pred(id: int):
    """Delete a specific prediction by ID"""
    delete_prediction(id)
    return {"message": f"Prediction {id} deleted successfully", "success": True}


@app.delete("/history", tags=["History"])
def delete_all():
    """Delete all predictions"""
    delete_all_predictions()
    return {"message": "All predictions deleted successfully", "success": True}


@app.get("/crop-details/{crop_name}", tags=["Reference"])
def crop_details(crop_name: str):
    """Get detailed statistics for a specific crop"""
    return get_crop_details(crop_name)


@app.get("/yield/{crop_name}", tags=["Reference"])
def yield_estimation(crop_name: str):
    """Get yield estimation for a specific crop"""
    return get_yield_estimation(crop_name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)