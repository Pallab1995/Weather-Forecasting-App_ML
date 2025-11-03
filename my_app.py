import streamlit as st
from datetime import date, timedelta
import requests
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# ------------------- Custom CSS -------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(120deg, #74ebd5, #ACB6E5, #89f7fe, #66a6ff);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Trebuchet MS', sans-serif;
    }

    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        color: #FFD700;
        text-shadow: 3px 3px 6px #00000066;
    }

    .subtitle {
        text-align: center;
        font-size: 22px;
        color: #ffffff;
        margin-bottom: 25px;
    }

    /* Make widget labels dark */
    .stSelectbox label, .stDateInput label {
        color: #222 !important;
        font-weight: 600;
        font-size: 18px;
    }

    .card {
        background: rgba(255, 255, 255, 0.15);
        padding: 20px;
        border-radius: 20px;
        color: #fff;
        text-align: center;
        font-size: 20px;
        box-shadow: 0px 8px 25px rgba(0,0,0,0.25);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------- Title -------------------
st.markdown('<div class="title">🌡 Future Temperature Predictor</div>', unsafe_allow_html=True)

# ------------------- API Endpoints -------------------
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
HIST_URL = "https://archive-api.open-meteo.com/v1/archive"

# ------------------- Helper Functions -------------------
def geocode_city(name):
    params = {"name": name, "count": 1, "language": "en", "Format": "json"}
    r = requests.get(GEOCODE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data.get('results'):
        raise ValueError(f"Could not geocode city '{name}'. Try a different spelling.")
    res = data['results'][0]
    return {
        'name': res.get('name'),
        'latitude': res['latitude'],
        'longitude': res['longitude'],
        'timezone': res.get('timezone'),
        'country': res.get('country'),
        'admin1': res.get('admin1')
    }

def fetch_history(lat, lon, start_date, end_date, timezone='auto'):
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'daily': ['temperature_2m_max', 'temperature_2m_min'],
        'timezone': timezone
    }
    r = requests.get(HIST_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    daily = data.get('daily', {})
    if not daily or 'time' not in daily:
        return pd.DataFrame()
    df = pd.DataFrame(daily)
    df['time'] = pd.to_datetime(df['time'])
    df["temp_mean"] = (pd.to_numeric(df["temperature_2m_max"], errors='coerce') +
                       pd.to_numeric(df["temperature_2m_min"], errors='coerce')) / 2.0
    df["date"] = pd.to_datetime(df["time"])
    df = df.dropna(subset=['temp_mean']).reset_index(drop=True)
    return df[["date", "temperature_2m_min", "temperature_2m_max", "temp_mean"]]

def predict_future_temp(hist_df, target_date):
    hist_df["days"] = (hist_df["date"] - hist_df["date"].min()).dt.days
    X = hist_df[["days"]]
    y = hist_df["temp_mean"]
    model = Pipeline([("poly", PolynomialFeatures(degree=3)), ("linear", LinearRegression())])
    model.fit(X, y)
    target_days = (target_date - hist_df["date"].min().date()).days
    pred_temp = model.predict(pd.DataFrame({"days": [target_days]}))[0]
    return pred_temp

# ------------------- UI Elements -------------------
cities = ["Kolkata", "Delhi", "Mumbai", "Chennai", "Bengaluru", "Hyderabad"]
city = st.selectbox("🏙️ Enter a city name in India:", cities)
future_date = st.date_input("📅 Select future date:", date.today() + timedelta(days=1))

# ------------------- Predict Button -------------------
if st.button("🔮 Predict"):
    with st.spinner("Fetching data & predicting temperature..."):
        try:
            place = geocode_city(city)
            lat, lon, tz = place['latitude'], place['longitude'], place['timezone']
            today = date.today()
            start_date = today - timedelta(days=120)
            end_date = today - timedelta(days=1)

            hist_df = fetch_history(lat, lon, start_date, end_date, tz)
            if hist_df.empty:
                st.error("No historical data available.")
            else:
                pred_temp = predict_future_temp(hist_df, future_date)

                # Select image and message based on temperature
                if pred_temp < 20:
                    img_url = "https://img.icons8.com/emoji/96/snowflake.png"
                    msg = "🥶 It's going to be quite chilly!"
                elif pred_temp < 30:
                    img_url = "https://img.icons8.com/emoji/96/sun-behind-cloud.png"
                    msg = "😊 Pleasant weather ahead!"
                else:
                    img_url = "https://img.icons8.com/emoji/96/sun.png"
                    msg = "🔥 It's going to be hot, stay hydrated!"

                st.image(img_url, width=100)

                # Styled result card
                st.markdown(
                    f"""
                    <div class="result-card" style="color:black;">
                        🌍 <b>{city}</b><br>
                        📅 {future_date}<br>
                        🌡️ Predicted Temperature: <b>{pred_temp:.2f}°C</b><br>
                        {msg}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Show historical trend + prediction
                chart_df = hist_df.copy()
                chart_df = chart_df.set_index("date")
                st.line_chart(chart_df["temp_mean"],use_container_width=True, height=250)

                st.markdown(
                     f'<p style="color:black; font-weight:bold;">🔎 Historical trend of mean daily temperature for {city}</p>',
                  unsafe_allow_html=True
                  )
        except Exception as e:
            st.error(f"Error: {str(e)}")






