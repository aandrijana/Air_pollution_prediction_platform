import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import datetime


# PAGE CONFIG
st.set_page_config(
    page_title="Sarajevo Air Quality Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS & STYLES
st.markdown("""
<style>
    .main-header { color: #1E3A8A; text-align: center; padding-bottom: 20px; border-bottom: 3px solid #3B82F6; margin-bottom: 30px; }
    .sub-header { font-size: 1.2rem; text-align: center; color: #6b7280; margin-bottom: 2rem; }
    .info-box { background: linear-gradient(135deg, #E0F2FE 0%, #DBEAFE 100%); padding: 20px; border-radius: 15px; border-left: 5px solid #3B82F6; margin: 20px 0; color: #1E40AF; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1); }
    .prediction-day { background: white; padding: 10px; border-radius: 10px; border: 1px solid #E2E8F0; margin: 3px; text-align: center; height: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    .day-header { color: #1E3A8A; font-weight: bold; font-size: 14px; margin-bottom: 8px; }
    .metric-value { color: #2563EB; font-weight: bold; font-size: 20px; margin: 5px 0; }
    .metric-label { color: #64748B; font-size: 12px; margin-bottom: 3px; }
    .level-badge { padding: 5px 12px; border-radius: 15px; font-size: 11px; font-weight: bold; text-align: center; margin-top: 8px; display: inline-block; color: white; width: 100%; }
    
    .good { background: #22c55e; }
    .moderate { background: #eab308; }
    .usg { background: #f97316; }
    .unhealthy { background: #ef4444; }
    .very-unhealthy { background: #a855f7; }
    .hazardous { background: #37055e; }
    
    .aqi-scale { width: 100%; border-collapse: collapse; margin: 10px 0; }
    .aqi-scale th, .aqi-scale td { padding: 8px; text-align: left; border-bottom: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)


# AQI LOGIC & SCALE
COLOR_LEVELS = [
    {"name": "Good", "hex": "#22c55e", "p25": "0-15", "p10": "0-50"},
    {"name": "Moderate", "hex": "#eab308", "p25": "16-35", "p10": "51-100"},
    {"name": "Poor / USG", "hex": "#f97316", "p25": "36-75", "p10": "101-150"},
    {"name": "Unhealthy", "hex": "#ef4444", "p25": "76-150", "p10": "151-250"},
    {"name": "Very Unhealthy", "hex": "#a855f7", "p25": "151-250", "p10": "251-350"},
    {"name": "Hazardous", "hex": "#37055e", "p25": ">250", "p10": ">350"}
]

def get_aqi_category(pm25, pm10):
    categories = [
        ("Good", 0, 15, 0, 50, "good"),
        ("Moderate", 16, 35, 51, 100, "moderate"),
        ("Poor", 36, 75, 101, 150, "usg"),
        ("Unhealthy", 76, 150, 151, 250, "unhealthy"),
        ("Very Unhealthy", 151, 250, 251, 350, "very-unhealthy"),
        ("Hazardous", 251, 1e9, 351, 1e9, "hazardous")
    ]
    p25_idx = next((i for i, c in enumerate(categories) if c[1] <= pm25 <= c[2]), 5)
    p10_idx = next((i for i, c in enumerate(categories) if c[3] <= pm10 <= c[4]), 5)
    idx = max(p25_idx, p10_idx)
    return categories[idx][0], categories[idx][5]

def create_color_bar():
    html = '<div style="height: 20px; border-radius: 10px; overflow: hidden; display: flex; margin-bottom: 10px;">'
    for level in COLOR_LEVELS:
        html += f'<div style="flex: 1; background: {level["hex"]};"></div>'
    html += '</div><div style="display: flex; justify-content: space-between;">'
    for level in COLOR_LEVELS:
        html += f'<div style="font-size: 10px; color: #475569; font-weight: 600;">{level["name"]}</div>'
    html += '</div>'
    return html

# DATA LOADING
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/aqi_dataset_processed.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df_master = load_data()


# SIDEBAR
st.sidebar.header("Prediction Settings")
stations = sorted(df_master["station"].unique())
station = st.sidebar.selectbox("Select Station", stations)
station_df = df_master[df_master["station"] == station].sort_values("date")


# MODELS LOADING
@st.cache_resource
def load_all_models(st_name):
    xgb_m = joblib.load(f"models_xgb/xgboost_model_{st_name}.pkl")
    xgb_xs = joblib.load(f"models_xgb/xgb_X_scaler_{st_name}.pkl")
    xgb_ys = joblib.load(f"models_xgb/xgb_y_scaler_{st_name}.pkl")
    gru_m = load_model(f"models_gru/pm10_model_{st_name}.keras")
    gru_s = joblib.load(f"models_gru/scaler_{st_name}.pkl")
    return xgb_m, xgb_xs, xgb_ys, gru_m, gru_s

try:
    xgb_model, xgb_X_scaler, xgb_y_scaler, gru_model, gru_scaler = load_all_models(station)
except Exception as e:
    st.error(f"Models for {station} could not be found.")
    st.stop()

# FORECAST LOGIC
def add_gru_features(df):
    d = df.copy()
    d["pm25_lag1"] = d["pm25"].shift(1)
    d["pm25_lag7"] = d["pm25"].shift(7)
    d["pm25_roll7_mean"] = d["pm25"].rolling(7).mean()
    d["doy_sin"] = np.sin(2*np.pi*d["date"].dt.dayofyear/365.25)
    d["doy_cos"] = np.cos(2*np.pi*d["date"].dt.dayofyear/365.25)
    d["o3_inverse"] = 1/(d["o3"]+1)
    return d.dropna()

def predict_7_days(input_df, xgb_m, xgb_xs, xgb_ys, gru_m, gru_s):
    features_gru = ["pm25","o3","no2","so2","pm10","pm25_lag1","pm25_lag7","pm25_roll7_mean","doy_sin","doy_cos","o3_inverse"]
    
    # 1. PM10 - GRU 
    gru_prep = add_gru_features(input_df)
    X_seq = gru_prep[features_gru].tail(30).values
    X_scaled = gru_s.transform(X_seq).reshape(1, 30, len(features_gru))
    
    gru_residuals = gru_m.predict(X_scaled, verbose=0)[0] 
    last_pm10_log = np.log1p(input_df["pm10"].iloc[-1])
    pm10_forecast = np.expm1(last_pm10_log + gru_residuals)

    # 2. PM2.5 - XGBoost 
    pm25_forecast = []
    temp_df = input_df.copy()
    numeric_cols = input_df.select_dtypes(include=["int64", "float64"]).columns

    for i in range(7):
        xgb_window = temp_df[numeric_cols].tail(7).values
        xgb_in = xgb_xs.transform(xgb_window).flatten().reshape(1, -1)
        p25_scaled = xgb_m.predict(xgb_in)
        p25_val = xgb_ys.inverse_transform(p25_scaled.reshape(-1,1))[0,0]
        pm25_forecast.append(p25_val)
        
        new_row = temp_df.iloc[-1:].copy()
        new_row["pm25"] = p25_val
        new_row["pm10"] = pm10_forecast[i]
        new_row["date"] = temp_df["date"].max() + pd.Timedelta(days=1)
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

    # 3. Format Output
    results = []
    color_map = {"good": "#22c55e", "moderate": "#eab308", "usg": "#f97316", "unhealthy": "#ef4444", "very-unhealthy": "#a855f7", "hazardous": "#37055e"}
    for i in range(7):
        target_date = input_df["date"].max() + pd.Timedelta(days=i+1)
        cat, css = get_aqi_category(pm25_forecast[i], pm10_forecast[i])
        results.append({
            "date": target_date, "pm25": pm25_forecast[i], "pm10": pm10_forecast[i],
            "AQI": cat, "color": color_map.get(css, "#6b7280"), "day_short": target_date.strftime('%a')
        })
    return results

# MAIN DASHBOARD
st.markdown('<h1 class="main-header">Sarajevo Air Quality Predictor</h1>', unsafe_allow_html=True)

# Pretty Color Bar
st.markdown("### Air Quality Scale")
st.markdown(create_color_bar(), unsafe_allow_html=True)

# Station Summary
st.markdown(f"""
<div class='info-box'>
    <b>Monitoring Station:</b> {station} <br>
    <b>Database Status:</b> Records up to {station_df['date'].max().strftime('%d.%B %Y')}
</div>
""", unsafe_allow_html=True)

# Threshold Table 
with st.expander("📊 View Detailed Pollutant Thresholds"):
    html_table = '<table class="aqi-scale"><tr><th>Category</th><th>PM2.5</th><th>PM10</th></tr>'
    for l in COLOR_LEVELS:
        html_table += f'<tr style="color: {l["hex"]}; font-weight: bold;"><td>{l["name"]}</td><td>{l["p25"]}</td><td>{l["p10"]}</td></tr>'
    html_table += '</table>'
    st.markdown(html_table, unsafe_allow_html=True)

# FORECAST SECTION
st.markdown("---")
if st.button("Generate 7-Day Forecast", type="primary", use_container_width=True):
    with st.spinner("Calculating atmospheric residuals..."):
        preds = predict_7_days(station_df, xgb_model, xgb_X_scaler, xgb_y_scaler, gru_model, gru_scaler)
    
    # Grid of Cards
    cols = st.columns(7)
    for idx, (p, col) in enumerate(zip(preds, cols)):
        with col:
            st.markdown(f"""
            <div class="prediction-day">
                <div class="day-header">{p['day_short']}<br>{p['date'].strftime('%d.%m')}</div>
                <div class="metric-label">PM2.5</div><div class="metric-value">{p['pm25']:.1f}</div>
                <div class="metric-label">PM10</div><div class="metric-value">{p['pm10']:.1f}</div>
                <div class="level-badge" style="background-color: {p['color']};">{p['AQI']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Trend Chart
    st.markdown("### 📊 Trend Analysis")
    c_df = pd.DataFrame(preds).set_index("date")
    st.line_chart(c_df[["pm25", "pm10"]])

# HISTORICAL SECTION
st.markdown("---")
st.markdown("### 📅 Past Performance Explorer")
h_date = st.date_input("Select a historical date to see model performance:", station_df["date"].max().date())
if st.button("Check Model Recall"):
    h_df = station_df[station_df["date"] < pd.to_datetime(h_date)]
    if len(h_df) >= 40:
        res = predict_7_days(h_df, xgb_model, xgb_X_scaler, xgb_y_scaler, gru_model, gru_scaler)[0]
        st.metric("Model Predicted PM2.5", f"{res['pm25']:.2f} µg/m³")
        st.metric("Model Predicted PM10", f"{res['pm10']:.2f} µg/m³")
        st.write(f"Health Risk Assessment: **{res['AQI']}**")
    else:
        st.warning("Insufficient data historical window for this date.")

st.caption("Causal AI • Sarajevo AQI Framework • 2018–2026")
