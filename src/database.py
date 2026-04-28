"""
database.py  —  FINAL VERSION
SQLite database for storing every prediction made by the app.
No setup needed — creates the DB file automatically.
"""
import sqlite3, json, os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "predictions.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            city        TEXT,
            N           REAL, P REAL, K REAL,
            temperature REAL, humidity REAL,
            ph          REAL, rainfall REAL,
            soil        INTEGER, season INTEGER, water REAL,
            crop_name   TEXT,
            confidence  REAL,
            alert_level TEXT,
            price       INTEGER
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(inputs: dict, result: dict):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    price = 0
    if result.get("market_price"):
        price = result["market_price"].get("estimated_price_per_quintal", 0) or 0
    conn.execute("""
        INSERT INTO predictions
        (timestamp, city, N, P, K, temperature, humidity, ph,
         rainfall, soil, season, water, crop_name, confidence, alert_level, price)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(),
        inputs.get("city", ""),
        inputs["N"], inputs["P"], inputs["K"],
        inputs["temperature"], inputs["humidity"], inputs["ph"],
        inputs["rainfall"], inputs["soil"], inputs["season"], inputs["water"],
        result.get("crop_name", ""),
        result.get("confidence", 0),
        result.get("alert_level", ""),
        int(price)
    ))
    conn.commit()
    conn.close()

def get_history(limit: int = 50):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return rows

def get_stats():
    init_db()
    conn  = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    top   = conn.execute(
        "SELECT crop_name, COUNT(*) as cnt FROM predictions "
        "GROUP BY crop_name ORDER BY cnt DESC LIMIT 5"
    ).fetchall()
    avg_conf = conn.execute(
        "SELECT AVG(confidence) FROM predictions"
    ).fetchone()[0] or 0
    conn.close()
    return {
        "total_predictions": total,
        "avg_confidence":    round(avg_conf * 100, 1),
        "top_5_crops": [{"crop": r[0], "count": r[1]} for r in top]
    }

def delete_prediction(id: int):
    """Delete a specific prediction by ID"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions WHERE id = ?", (id,))
    conn.commit()
    conn.close()

def delete_all_predictions():
    """Delete all predictions (admin function)"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

def get_crop_details(crop_name: str):
    """Get detailed information about a specific crop"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    stats = conn.execute("""
        SELECT COUNT(*) as count, AVG(confidence) as avg_conf, MAX(price) as max_price, MIN(price) as min_price
        FROM predictions WHERE crop_name = ?
    """, (crop_name,)).fetchone()
    conn.close()
    return {
        "count": stats[0] or 0,
        "avg_confidence": round((stats[1] or 0) * 100, 1),
        "max_price": stats[2] or 0,
        "min_price": stats[3] or 0
    }