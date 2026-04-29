"""
CropAdvisor AI - Smart Crop Recommendation System
Streamlit Web Application

This is the user-friendly web interface for the AI-powered crop recommendation system.
Run locally: streamlit run streamlit_app.py
Deploy to: streamlit.io (free cloud hosting)
"""
import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime

# ── PAGE CONFIGURATION ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropAdvisor AI - Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── API CONFIGURATION ────────────────────────────────────────────────────────
# For local development, use localhost. For deployment, update this to your deployed API URL
API_BASE_URL = "http://localhost:8000"

# ── CSS STYLING ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #15803d;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
    }
    .success-box {
        background-color: #dcfce7;
        border-left: 4px solid #15803d;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .info-box {
        background-color: #e0f2fe;
        border-left: 4px solid #0ea5e9;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #15803d 0%, #22c55e 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def get_weather(city):
    """Fetch weather data from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/weather/{city}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Weather service unavailable"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def get_prediction(data):
    """Send prediction request to the API"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Prediction failed: {response.text}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# ── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/000000/sheaf-of-rice-emoji.png", width=80)
    st.markdown("## 🌾 CropAdvisor AI")
    st.markdown("---")

    # API Status
    if check_api_health():
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Not Connected")
        st.info("Make sure to run: uvicorn app:app --reload")

    st.markdown("---")
    st.markdown("### 📊 Quick Links")
    st.markdown("- [GitHub Repository](#)")  # Add your GitHub URL
    st.markdown("- [Model Documentation](#)")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **CropAdvisor AI** uses machine learning to recommend the best crops based on:
    - Soil nutrients (N, P, K)
    - Weather conditions
    - Season and water availability
    """)

# ── MAIN CONTENT ───────────────────────────────────────────────────────────

# Header
st.markdown('<p class="main-header">🌾 CropAdvisor AI</p>', unsafe_allow_html=True)
st.markdown("### Smart Crop Recommendation System for Farmers")

# Create tabs for different features
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Get Recommendation", "📖 History", "📊 Analytics", "❓ Help"])

# ── TAB 1: PREDICTION ───────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Enter Farm Details")

        # Weather Auto-fill
        st.markdown("#### 🌤️ Weather Data (Optional)")
        col_weather1, col_weather2 = st.columns([3, 1])
        with col_weather1:
            city = st.text_input("Enter City Name", placeholder="e.g., Kolhapur, Mumbai, Delhi")
        with col_weather2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Fetch Weather", type="secondary"):
                if city:
                    with st.spinner("Fetching weather data..."):
                        weather = get_weather(city)
                        if "error" not in weather:
                            st.session_state['weather_data'] = weather
                            st.success(f"✅ Weather loaded: {weather.get('temperature')}°C, {weather.get('humidity')}% humidity")
                        else:
                            st.error(f"❌ {weather.get('error')}")

        # Initialize session state
        if 'weather_data' not in st.session_state:
            st.session_state['weather_data'] = None

        # Soil Nutrients
        st.markdown("#### 🧪 Soil Nutrients")
        col_npk1, col_npk2, col_npk3 = st.columns(3)
        with col_npk1:
            N = st.number_input("Nitrogen (N) kg/ha", min_value=0.0, max_value=200.0, value=90.0, help="Nitrogen in soil (kg per hectare)")
        with col_npk2:
            P = st.number_input("Phosphorus (P) kg/ha", min_value=0.0, max_value=150.0, value=42.0, help="Phosphorus in soil (kg per hectare)")
        with col_npk3:
            K = st.number_input("Potassium (K) kg/ha", min_value=0.0, max_value=200.0, value=43.0, help="Potassium in soil (kg per hectare)")

        # Weather
        st.markdown("#### 🌡️ Climate Conditions")
        col_climate1, col_climate2, col_climate3 = st.columns(3)
        with col_climate1:
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=20.9, help="Average temperature in Celsius")
        with col_climate2:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, help="Relative humidity percentage")
        with col_climate3:
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=202.9, help="Annual rainfall in mm")

        # Soil and Season
        st.markdown("#### 🌍 Farm Details")
        col_farm1, col_farm2 = st.columns(2)
        with col_farm1:
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1, help="Soil acidity/alkalinity (0-14)")
        with col_farm2:
            water = st.number_input("Water Requirement (mm/year)", min_value=0.0, max_value=3000.0, value=887.3, help="Annual water requirement")

        # Soil Type
        soil_type = st.selectbox(
            "Soil Type",
            options=list(range(35)),
            format_func=lambda x: ["Clayey", "Sandy", "Loamy", "Black", "Red", "Laterite"][x % 6] + f" (Type {x})"
        )

        # Season
        season = st.radio(
            "Season",
            options=[0, 1, 2, 3],
            format_func=lambda x: ["Kharif (Monsoon)", "Rabi (Winter)", "Zaid (Summer)", "Whole Year"][x],
            horizontal=True
        )

    with col2:
        st.markdown("### 📋 Summary")
        st.markdown("#### Current Inputs:")
        st.json({
            "N": N,
            "P": P,
            "K": K,
            "Temperature": temperature,
            "Humidity": humidity,
            "Rainfall": rainfall,
            "pH": ph,
            "Water": water,
            "Soil Type": soil_type,
            "Season": ["Kharif", "Rabi", "Zaid", "Whole"][season]
        })

    # ── PREDICT BUTTON ─────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🌾 Get Crop Recommendation", type="primary", use_container_width=True):
        if not city:
            st.warning("⚠️ Please enter a city name to get weather data (or manually fill in the fields)")

        # Prepare input data
        input_data = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
            "soil": soil_type,
            "season": season,
            "water": water,
            "city": city
        }

        with st.spinner("🤖 AI is analyzing your farm data..."):
            result = get_prediction(input_data)

        if "error" not in result:
            st.markdown("---")
            st.markdown("### 🎉 Recommended Crop")

            # Main result
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

                # Fertilizer
                fertilizer = result.get("fertilizer", {})
                if fertilizer:
                    st.markdown("#### 🌱 Recommended Fertilizer:")
                    st.markdown(f"- **Name:** {fertilizer.get('name', 'N/A')}")
                    st.markdown(f"- **Amount:** {fertilizer.get('amount', 'N/A')} kg/ha")

                # Market Price
                if result.get("market_price"):
                    st.markdown(f"💰 **Market Price:** ₹{result.get('market_price')}/quintal")

            # Top 3 alternatives
            top_crops = result.get("top_3", [])
            if top_crops:
                st.markdown("#### 🔝 Top 3 Alternatives:")
                for i, crop in enumerate(top_crops, 1):
                    st.markdown(f"{i}. **{crop.get('crop_name', 'Unknown')}** - {crop.get('probability', 0)*100:.1f}%")

            # Yield Estimation
            if result.get("yield_estimation"):
                st.markdown("#### 📈 Yield Estimation:")
                yield_info = result.get("yield_estimation", {})
                st.markdown(f"- **Range:** {yield_info.get('min', 0)} - {yield_info.get('max', 0)} {yield_info.get('unit', 'kg/acre')}")
                st.markdown(f"- **Duration:** {yield_info.get('duration', 'N/A')}")

            st.success("✅ Prediction completed successfully!")

        else:
            st.error(f"❌ Error: {result.get('error')}")

# ── TAB 2: HISTORY ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 📖 Prediction History")

    try:
        response = requests.get(f"{API_BASE_URL}/history?limit=20", timeout=10)
        if response.status_code == 200:
            data = response.json()
            predictions = data.get("predictions", [])

            if predictions:
                # Create a DataFrame for display
                df = pd.DataFrame(predictions)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No prediction history found. Make your first prediction!")
        else:
            st.error("Could not fetch history. Make sure API is running.")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")

# ── TAB 3: ANALYTICS ────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 System Analytics")

    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()

            # Display metrics
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

            with col_stat1:
                st.metric("Total Predictions", stats.get("total_predictions", 0))

            with col_stat2:
                st.metric("Avg Confidence", f"{stats.get('avg_confidence', 0)*100:.1f}%")

            with col_stat3:
                st.metric("Unique Crops", stats.get("unique_crops", 0))

            with col_stat4:
                st.metric("API Version", stats.get("version", "N/A"))

            # Top crops
            top_crops = stats.get("top_crops", [])
            if top_crops:
                st.markdown("#### 🏆 Top Recommended Crops:")
                for crop in top_crops:
                    st.markdown(f"- **{crop.get('crop_name')}**: {crop.get('count')} times")
        else:
            st.error("Could not fetch stats. Make sure API is running.")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")

# ── TAB 4: HELP ─────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### ❓ How to Use CropAdvisor AI")

    st.markdown("""
    #### Step-by-Step Guide:

    **1. Enter Weather Data (Optional)**
    - Type your city name and click "Fetch Weather"
    - The system will automatically fill temperature, humidity, and rainfall

    **2. Enter Soil Nutrients**
    - **Nitrogen (N)**: Essential for plant growth (0-200 kg/ha)
    - **Phosphorus (P)**: Important for root development (0-150 kg/ha)
    - **Potassium (K)**: Helps with disease resistance (0-200 kg/ha)

    **3. Enter Climate Conditions**
    - **Temperature**: Average temperature in Celsius
    - **Humidity**: Relative humidity percentage
    - **Rainfall**: Annual rainfall in mm

    **4. Enter Farm Details**
    - **Soil pH**: Acidity/alkalinity of soil (0-14, 7 is neutral)
    - **Water Requirement**: Annual water needed in mm
    - **Soil Type**: Select your soil type from the dropdown
    - **Season**: Choose the appropriate season

    **5. Get Recommendation**
    - Click the "Get Crop Recommendation" button
    - View the AI-predicted best crop for your farm

    #### 🌐 Deployment Options:

    **Local Deployment:**
    ```bash
    # Terminal 1 - Start API
    uvicorn app:app --reload

    # Terminal 2 - Start Streamlit
    streamlit run streamlit_app.py
    ```

    **Cloud Deployment:**
    - Deploy API to: Railway, Render, or Heroku
    - Deploy Frontend to: Streamlit Cloud (free)
    """)

    st.markdown("---")
    st.markdown("### 📞 Support")
    st.markdown("""
    For issues or questions:
    - Check the GitHub repository
    - Review the API documentation at `/docs`
    """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #64748b; padding: 1rem;">
    <p>🌾 CropAdvisor AI v2.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d')}</p>
    <p>Powered by Machine Learning | OpenWeatherMap API</p>
</div>
""", unsafe_allow_html=True)