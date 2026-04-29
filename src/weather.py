"""
weather.py  —  FINAL VERSION
Fetches live weather from OpenWeatherMap API.
Get your FREE API key at: openweathermap.org (takes 2 minutes)
"""
import requests
from urllib.parse import quote

# ── PASTE YOUR FREE API KEY HERE ─────────────────────────────────────────────
WEATHER_API_KEY = "206f49066840164c3994b0f2328bef5c"
# ─────────────────────────────────────────────────────────────────────────────

def get_weather(city: str) -> dict:
    """
    Returns temperature (C), humidity (%), and estimated rainfall (mm)
    for the given city name.
    """
    if WEATHER_API_KEY == "YOUR_API_KEY_HERE":
        # Return dummy data so app works even without API key during dev/demo
        return {
            "city":        city,
            "temperature": 25.0,
            "humidity":    70.0,
            "rainfall":    180.0,
            "note":        "Demo data — add real API key in weather.py"
        }

    try:
        url  = (f"https://api.openweathermap.org/data/2.5/weather"
                f"?q={quote(city)}&appid={WEATHER_API_KEY}")
        data = requests.get(url, timeout=5).json()

        if data.get("cod") != 200:
            return {"error": f"City not found: {city}"}

        temp     = round(data["main"]["temp"] - 273.15, 1)     # Kelvin → Celsius
        humidity = data["main"]["humidity"]
        rain_1h  = data.get("rain", {}).get("1h", 0)
        # Rough annual estimate from 1-hour reading
        rainfall = round(rain_1h * 24 * 120, 1)

        return {
            "city":        city,
            "temperature": temp,
            "humidity":    float(humidity),
            "rainfall":    rainfall,
            "description": data["weather"][0]["description"].title(),
        }
    except Exception as e:
        return {"error": str(e)}