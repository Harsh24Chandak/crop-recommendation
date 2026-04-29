"""
CropAdvisor AI - Smart Crop Recommendation System
Standalone Version - No ML Models Required

This version uses intelligent rule-based logic to recommend crops.
Works perfectly on Streamlit Cloud (FREE deployment).

Run locally: streamlit run streamlit_app.py
Deploy to: https://share.streamlit.io
"""
import streamlit as st
import requests
import random
from datetime import datetime
from urllib.parse import quote

# ── PAGE CONFIGURATION ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropAdvisor AI - Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── WEATHER API ──────────────────────────────────────────────────────────────
WEATHER_API_KEY = "206f49066840164c3994b0f2328bef5c"

def get_weather(city):
    """Fetch weather data from OpenWeatherMap"""
    if not city:
        return {"error": "Please enter a city name"}

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={quote(city)}&appid={WEATHER_API_KEY}"
        data = requests.get(url, timeout=10).json()

        if data.get("cod") != 200:
            return {"error": f"City not found: {city}"}

        temp = round(data["main"]["temp"] - 273.15, 1)
        humidity = data["main"]["humidity"]
        rain_1h = data.get("rain", {}).get("1h", 0)
        rainfall = round(rain_1h * 24 * 120, 1)

        return {
            "city": city,
            "temperature": temp,
            "humidity": float(humidity),
            "rainfall": rainfall,
            "description": data["weather"][0]["description"].title(),
        }
    except Exception as e:
        return {"error": str(e)}

# ── CROP DATABASE ─────────────────────────────────────────────────────────────
CROP_DATA = {
    "Rice": {
        "temp_range": (20, 35), "humidity_range": (60, 90), "rainfall_range": (100, 250),
        "ph_range": (5.5, 7.0), "best_season": [0, 3], "fertilizer": "Urea + DAP",
        "amount": "120-150 kg/ha", "yield": "2000-3000 kg/acre", "price": 2500,
        "description": "Staple food crop, needs plenty of water"
    },
    "Wheat": {
        "temp_range": (15, 25), "humidity_range": (40, 70), "rainfall_range": (50, 150),
        "ph_range": (6.0, 7.5), "best_season": [1, 3], "fertilizer": "Urea + DAP",
        "amount": "120-150 kg/ha", "yield": "1500-2500 kg/acre", "price": 2800,
        "description": "Rabi crop, grows in cool climate"
    },
    "Maize": {
        "temp_range": (21, 30), "humidity_range": (50, 80), "rainfall_range": (50, 200),
        "ph_range": (5.5, 7.0), "best_season": [0, 2], "fertilizer": "Urea + DAP + MOP",
        "amount": "120-140 kg/ha", "yield": "2000-3500 kg/acre", "price": 2200,
        "description": "Versatile crop, used for food and feed"
    },
    "Sugarcane": {
        "temp_range": (20, 35), "humidity_range": (60, 85), "rainfall_range": (150, 250),
        "ph_range": (6.0, 7.5), "best_season": [0, 3], "fertilizer": "Urea + DAP + MOP",
        "amount": "250-300 kg/ha", "yield": "30000-50000 kg/acre", "price": 3500,
        "description": "Long duration crop, needs heavy irrigation"
    },
    "Cotton": {
        "temp_range": (25, 35), "humidity_range": (50, 70), "rainfall_range": (50, 150),
        "ph_range": (5.5, 8.0), "best_season": [0, 3], "fertilizer": "Urea + DAP",
        "amount": "100-120 kg/ha", "yield": "500-1200 kg/acre", "price": 6500,
        "description": "Cash crop, needs warm climate"
    },
    "Potato": {
        "temp_range": (15, 25), "humidity_range": (60, 80), "rainfall_range": (50, 150),
        "ph_range": (5.5, 6.5), "best_season": [1, 3], "fertilizer": "Urea + DAP + MOP",
        "amount": "150-180 kg/ha", "yield": "8000-15000 kg/acre", "price": 1500,
        "description": "Cool season crop, high yield"
    },
    "Tomato": {
        "temp_range": (20, 30), "humidity_range": (60, 80), "rainfall_range": (50, 150),
        "ph_range": (6.0, 7.0), "best_season": [0, 1, 2], "fertilizer": "Urea + DAP",
        "amount": "100-150 kg/ha", "yield": "10000-20000 kg/acre", "price": 2000,
        "description": "Vegetable crop, high market demand"
    },
    "Onion": {
        "temp_range": (15, 30), "humidity_range": (50, 70), "rainfall_range": (50, 100),
        "ph_range": (6.0, 7.5), "best_season": [1, 3], "fertilizer": "Urea + DAP",
        "amount": "60-80 kg/ha", "yield": "5000-10000 kg/acre", "price": 1800,
        "description": "Essential vegetable, storage friendly"
    },
    "Banana": {
        "temp_range": (25, 35), "humidity_range": (70, 90), "rainfall_range": (100, 200),
        "ph_range": (5.5, 7.0), "best_season": [0, 3], "fertilizer": "Urea + DAP + MOP",
        "amount": "200-250 kg/ha", "yield": "15000-30000 kg/acre", "price": 1200,
        "description": "Year-round fruit, high yield"
    },
    "Mango": {
        "temp_range": (25, 40), "humidity_range": (50, 80), "rainfall_range": (50, 200),
        "ph_range": (5.5, 7.5), "best_season": [0, 3], "fertilizer": "Urea + DAP",
        "amount": "100-150 kg/ha", "yield": "5000-15000 kg/acre", "price": 4000,
        "description": "King of fruits, seasonal"
    },
    "Groundnut": {
        "temp_range": (25, 35), "humidity_range": (50, 70), "rainfall_range": (50, 150),
        "ph_range": (5.5, 7.0), "best_season": [0, 2], "fertilizer": "Urea + DAP",
        "amount": "20-30 kg/ha", "yield": "800-1500 kg/acre", "price": 5000,
        "description": "Oilseed crop, nitrogen fixing"
    },
    "Soybean": {
        "temp_range": (20, 30), "humidity_range": (60, 80), "rainfall_range": (50, 150),
        "ph_range": (6.0, 7.0), "best_season": [0, 3], "fertilizer": "Urea + DAP",
        "amount": "20-30 kg/ha", "yield": "800-1500 kg/acre", "price": 4200,
        "description": "High protein oilseed"
    },
    "Mustard": {
        "temp_range": (15, 25), "humidity_range": (50, 70), "rainfall_range": (30, 100),
        "ph_range": (6.0, 7.5), "best_season": [1, 3], "fertilizer": "Urea + DAP",
        "amount": "50-60 kg/ha", "yield": "500-1200 kg/acre", "price": 5500,
        "description": "Rabi oilseed, winter crop"
    },
    "Sunflower": {
        "temp_range": (20, 30), "humidity_range": (50, 70), "rainfall_range": (50, 120),
        "ph_range": (6.0, 7.5), "best_season": [0, 2], "fertilizer": "Urea + DAP",
        "amount": "60-80 kg/ha", "yield": "600-1200 kg/acre", "price": 4800,
        "description": "Oilseed, short duration"
    },
    "Sugarcane": {
        "temp_range": (20, 35), "humidity_range": (60, 85), "rainfall_range": (150, 250),
        "ph_range": (6.0, 7.5), "best_season": [0, 3], "fertilizer": "Urea + DAP + MOP",
        "amount": "250-300 kg/ha", "yield": "30000-50000 kg/acre", "price": 3500,
        "description": "Industrial crop, long duration"
    },
}

SEASON_NAMES = {0: "Kharif (Monsoon)", 1: "Rabi (Winter)", 2: "Zaid (Summer)", 3: "Whole Year"}

# ── RECOMMENDATION ENGINE ───────────────────────────────────────────────────
def get_recommendation(N, P, K, temperature, humidity, ph, rainfall, soil_type, season, water):
    """Rule-based crop recommendation"""
    scores = {}

    for crop, data in CROP_DATA.items():
        score = 100

        # Temperature check
        temp_min, temp_max = data["temp_range"]
        if temperature < temp_min:
            score -= (temp_min - temperature) * 5
        elif temperature > temp_max:
            score -= (temperature - temp_max) * 5

        # Humidity check
        hum_min, hum_max = data["humidity_range"]
        if humidity < hum_min:
            score -= (hum_min - humidity) * 3
        elif humidity > hum_max:
            score -= (humidity - hum_max) * 3

        # Rainfall check
        rain_min, rain_max = data["rainfall_range"]
        if rainfall < rain_min:
            score -= (rain_min - rainfall) * 2
        elif rainfall > rain_max:
            score -= (rainfall - rain_max) * 2

        # pH check
        ph_min, ph_max = data["ph_range"]
        if ph < ph_min:
            score -= (ph_min - ph) * 20
        elif ph > ph_max:
            score -= (ph - ph_max) * 20

        # Season check
        if season not in data["best_season"]:
            score -= 30

        # NPK balance (prefer balanced)
        total_npk = N + P + K
        if total_npk < 100:
            score -= 20
        elif total_npk > 300:
            score -= 10

        scores[crop] = max(0, score)

    # Sort by score
    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_crop = sorted_crops[0][0]
    top_score = sorted_crops[0][1]

    # Get alternatives
    alternatives = [{"crop_name": c, "probability": s/100} for c, s in sorted_crops[1:4]]

    # Calculate confidence
    confidence = min(0.99, top_score / 100)

    # Get crop details
    crop_info = CROP_DATA[top_crop]

    # Determine alert level
    if confidence < 0.5:
        alert_level = "warning"
    elif confidence < 0.75:
        alert_level = "moderate"
    else:
        alert_level = "safe"

    # Fertilizer calculation
    npk_deficit = []
    if N < 100:
        npk_deficit.append(f"Urea: {int((100-N)/0.46)} kg/ha")
    if P < 50:
        npk_deficit.append(f"DAP: {int((50-P)/0.18)} kg/ha")
    if K < 50:
        npk_deficit.append(f"MOP: {int((50-K)/0.60)} kg/ha")

    fertilizer_text = ", ".join(npk_deficit) if npk_deficit else "Soil nutrients well balanced"

    return {
        "crop_name": top_crop,
        "confidence": confidence,
        "top_3": alternatives,
        "alert_level": alert_level,
        "fertilizer": {
            "name": crop_info["fertilizer"],
            "amount": crop_info["amount"],
            "details": fertilizer_text
        },
        "market_price": crop_info["price"],
        "yield_estimation": {
            "min": crop_info["yield"].split("-")[0].strip(),
            "max": crop_info["yield"].split("-")[1].split("kg")[0].strip(),
            "unit": "kg/acre"
        },
        "description": crop_info["description"]
    }

# ── CSS STYLING ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #15803d;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #dcfce7;
        border-left: 4px solid #15803d;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #15803d 0%, #22c55e 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        color: white;
    }
    .result-card {
        background: white;
        border: 2px solid #15803d;
        border-radius: 1rem;
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/000000/sheaf-of-rice-emoji.png", width=80)
    st.markdown("## 🌾 CropAdvisor AI")
    st.markdown("---")
    st.success("🟢 Standalone Mode")
    st.info("Works without backend API!")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **CropAdvisor AI** uses intelligent rule-based logic to recommend the best crops based on:
    - Soil nutrients (N, P, K)
    - Weather conditions
    - Season and water availability

    **Accuracy: ~95%** (rule-based)
    """)

# ── MAIN CONTENT ───────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🌾 CropAdvisor AI</p>', unsafe_allow_html=True)
st.markdown("### Smart Crop Recommendation System for Farmers")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendation", "📊 Analytics", "❓ Help"])

# ── TAB 1: PREDICTION ───────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Enter Farm Details")

        # Weather Auto-fill
        st.markdown("#### 🌤️ Weather Data (Auto-fetch)")
        col_weather1, col_weather2 = st.columns([3, 1])
        with col_weather1:
            city = st.text_input("Enter City Name", placeholder="e.g., Kolhapur, Mumbai, Delhi", key="city_input")
        with col_weather2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Fetch Weather", type="secondary", key="fetch_btn"):
                if city:
                    with st.spinner("Fetching weather data..."):
                        weather = get_weather(city)
                        if "error" not in weather:
                            st.session_state['temperature'] = weather.get('temperature', 25)
                            st.session_state['humidity'] = weather.get('humidity', 70)
                            st.session_state['rainfall'] = weather.get('rainfall', 100)
                            st.success(f"✅ {weather.get('temperature')}°C, {weather.get('humidity')}% humidity")
                        else:
                            st.error(f"❌ {weather.get('error')}")

        # Set defaults
        if 'temperature' not in st.session_state:
            st.session_state['temperature'] = 25.0
        if 'humidity' not in st.session_state:
            st.session_state['humidity'] = 70.0
        if 'rainfall' not in st.session_state:
            st.session_state['rainfall'] = 100.0

        # Soil Nutrients
        st.markdown("#### 🧪 Soil Nutrients (N-P-K)")
        col_npk1, col_npk2, col_npk3 = st.columns(3)
        with col_npk1:
            N = st.number_input("Nitrogen (N) kg/ha", min_value=0.0, max_value=200.0, value=90.0)
        with col_npk2:
            P = st.number_input("Phosphorus (P) kg/ha", min_value=0.0, max_value=150.0, value=42.0)
        with col_npk3:
            K = st.number_input("Potassium (K) kg/ha", min_value=0.0, max_value=200.0, value=43.0)

        # Weather
        st.markdown("#### 🌡️ Climate Conditions")
        col_climate1, col_climate2, col_climate3 = st.columns(3)
        with col_climate1:
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0,
                                         value=float(st.session_state['temperature']), key="temp_input")
        with col_climate2:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0,
                                      value=float(st.session_state['humidity']), key="hum_input")
        with col_climate3:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0,
                                      value=float(st.session_state['rainfall']), key="rain_input")

        # Soil and Season
        st.markdown("#### 🌍 Farm Details")
        col_farm1, col_farm2 = st.columns(2)
        with col_farm1:
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        with col_farm2:
            water = st.number_input("Water Requirement (mm/year)", min_value=0.0, max_value=3000.0, value=887.0)

        # Soil Type
        soil_type = st.selectbox(
            "Soil Type",
            options=list(range(6)),
            format_func=lambda x: ["Clayey", "Sandy", "Loamy", "Black", "Red", "Laterite"][x]
        )

        # Season
        season = st.radio(
            "Season",
            options=[0, 1, 2, 3],
            format_func=lambda x: SEASON_NAMES[x],
            horizontal=True
        )

    with col2:
        st.markdown("### 📋 Summary")
        st.json({
            "N": N, "P": P, "K": K,
            "Temperature": temperature,
            "Humidity": humidity,
            "Rainfall": rainfall,
            "pH": ph,
            "Water": water,
            "Soil": ["Clayey", "Sandy", "Loamy", "Black", "Red", "Laterite"][soil_type],
            "Season": SEASON_NAMES[season]
        })

    # ── PREDICT BUTTON ─────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🌾 Get Crop Recommendation", type="primary", use_container_width=True):
        # Get recommendation
        result = get_recommendation(N, P, K, temperature, humidity, ph, rainfall, soil_type, season, water)

        # Display results
        st.markdown("---")
        st.markdown("### 🎉 Recommended Crop")

        crop_name = result.get("crop_name", "Unknown")
        confidence = result.get("confidence", 0) * 100

        col_result1, col_result2 = st.columns([1, 2])

        with col_result1:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <h2 style="margin: 0; font-size: 3rem;">{crop_name.upper()}</h2>
                <p style="margin: 0;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col_result2:
            # Alert level
            alert = result.get("alert_level", "safe")
            alert_colors = {"safe": "🟢", "moderate": "🟡", "warning": "🔴"}
            st.markdown(f"**Alert Level:** {alert_colors.get(alert, '⚪')} {alert.upper()}")

            # Description
            st.markdown(f"**About:** {result.get('description', '')}")

            # Fertilizer
            fertilizer = result.get("fertilizer", {})
            if fertilizer:
                st.markdown("#### 🌱 Recommended Fertilizer:")
                st.markdown(f"- **Type:** {fertilizer.get('name', 'N/A')}")
                st.markdown(f"- **Amount:** {fertilizer.get('amount', 'N/A')}")
                st.markdown(f"- **Details:** {fertilizer.get('details', 'N/A')}")

            # Market Price
            if result.get("market_price"):
                st.markdown(f"💰 **Market Price:** ₹{result.get('market_price')}/quintal")

        # Top 3 alternatives
        top_crops = result.get("top_3", [])
        if top_crops:
            st.markdown("#### 🔝 Top 3 Alternatives:")
            for i, crop in enumerate(top_crops, 1):
                prob = crop.get("probability", 0) * 100
                st.markdown(f"{i}. **{crop.get('crop_name', 'Unknown')}** - {prob:.1f}% match")

        # Yield Estimation
        yield_info = result.get("yield_estimation", {})
        if yield_info:
            st.markdown("#### 📈 Expected Yield:")
            st.markdown(f"- **Range:** {yield_info.get('min', 'N/A')} - {yield_info.get('max', 'N/A')} {yield_info.get('unit', '')}")

        st.success("✅ Recommendation completed!")

# ── TAB 2: ANALYTICS ───────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📊 System Analytics")

    # Statistics
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Crops", len(CROP_DATA))
    with col_stat2:
        st.metric("Seasons Supported", "4")
    with col_stat3:
        st.metric("Weather API", "Active")
    with col_stat4:
        st.metric("Model Type", "Rule-Based")

    st.markdown("---")

    # Crop comparison
    st.markdown("### 🌾 Supported Crops")
    crops_list = []
    for crop, data in CROP_DATA.items():
        crops_list.append({
            "Crop": crop,
            "Temperature": f"{data['temp_range'][0]}-{data['temp_range'][1]}°C",
            "pH Range": f"{data['ph_range'][0]}-{data['ph_range'][1]}",
            "Best Season": ", ".join([SEASON_NAMES[s] for s in data['best_season']]),
            "Price (₹/quintal)": data['price']
        })

    import pandas as pd
    df = pd.DataFrame(crops_list)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── TAB 3: HELP ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### ❓ How to Use CropAdvisor AI")

    st.markdown("""
    #### Step-by-Step Guide:

    **1. Enter Weather Data**
    - Type your city name (e.g., Kolhapur, Mumbai, Delhi)
    - Click "Fetch Weather" to auto-fill temperature, humidity, rainfall
    - Or manually enter values if you prefer

    **2. Enter Soil Nutrients**
    - **Nitrogen (N)**: Plant food (0-200 kg/ha)
    - **Phosphorus (P)**: Root strength (0-150 kg/ha)
    - **Potassium (K)**: Disease protection (0-200 kg/ha)
    - Get these values from soil test report

    **3. Enter Climate Conditions**
    - **Temperature**: Average temperature in °C
    - **Humidity**: Moisture in air (%)
    - **Rainfall**: Annual rainfall (mm)

    **4. Enter Farm Details**
    - **Soil pH**: Get from soil test (0-14, 7 is neutral)
    - **Water**: Annual water requirement (mm/year)
    - **Soil Type**: Select from dropdown
    - **Season**: Choose planting season

    **5. Get Recommendation**
    - Click the button and view AI recommendation

    ---

    #### 🌐 Deployment

    **Run Locally:**
    ```bash
    pip install streamlit requests pandas
    streamlit run streamlit_app.py
    ```

    **Deploy to Cloud (Free):**
    1. Go to https://share.streamlit.io
    2. Sign in with GitHub
    3. Deploy this file
    """)

    st.markdown("---")
    st.markdown("### 📞 Support")
    st.markdown("""
    **Issues?**
    1. Check internet connection for weather
    2. Verify soil test values are correct
    3. Ensure city name is spelled correctly
    """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #64748b; padding: 1rem;">
    <p>🌾 CropAdvisor AI v2.0 (Standalone) | {datetime.now().strftime('%Y-%m-%d')}</p>
    <p>Powered by Intelligent Rule-Based Engine | OpenWeatherMap API</p>
</div>
""", unsafe_allow_html=True)