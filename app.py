import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="Sarajevo Air Quality Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        color: #1E3A8A;
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 3px solid #3B82F6;
        margin-bottom: 30px;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #E0F2FE 0%, #DBEAFE 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #3B82F6;
        margin: 20px 0;
        color: #1E40AF;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1);
    }
    
    .prediction-day {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin: 5px;
        text-align: center;
        height: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .day-header {
        color: #1E3A8A;
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        color: #2563EB;
        font-weight: bold;
        font-size: 20px;
        margin: 5px 0;
    }
    
    .metric-label {
        color: #64748B;
        font-size: 12px;
        margin-bottom: 3px;
    }
    
    .level-badge {
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: bold;
        text-align: center;
        margin-top: 8px;
        display: inline-block;
        color: white;
    }
    
    .metric-box {
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
    }
    
    .good { background: #22c55e; }
    .moderate { background: #eab308; }
    .usg { background: #f97316; }
    .unhealthy { background: #ef4444; }
    .very-unhealthy { background: #a855f7; }
    .hazardous { background: #37055e; }
    
    .aqi-scale th, .aqi-scale td {
        padding: 0.5rem;
        text-align: left;
    }
    
    .aqi-scale th {
        background: #000000;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# TITLES
st.markdown('<h1 class="main-header">Sarajevo Air Quality Prediction</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">XGBoost (PM2.5) + GRU (PM10)</div>', unsafe_allow_html=True)

# AQI LOGIC
def get_aqi_category(pm25, pm10):
    categories = [
        ("Good", 0, 15, 0, 50, "good"),
        ("Moderate", 16, 35, 51, 100, "moderate"),
        ("Poor / USG", 36, 75, 101, 150, "usg"),
        ("Unhealthy", 76, 150, 151, 250, "unhealthy"),
        ("Very Unhealthy", 151, 250, 251, 350, "very-unhealthy"),
        ("Hazardous / Extreme", 251, 1e9, 351, 1e9, "hazardous")
    ]

    pm25_idx = pm10_idx = 0
    for i, (_, p25_lo, p25_hi, p10_lo, p10_hi, _) in enumerate(categories):
        if p25_lo <= pm25 <= p25_hi: pm25_idx = i
        if p10_lo <= pm10 <= p10_hi: pm10_idx = i

    idx = max(pm25_idx, pm10_idx)
    return categories[idx][0], categories[idx][5]

COLOR_LEVELS = [
    {"name": "Good", "hex": "#22c55e"},
    {"name": "Moderate", "hex": "#eab308"},
    {"name": "Poor / USG", "hex": "#f97316"},
    {"name": "Unhealthy", "hex": "#ef4444"},
    {"name": "Very Unhealthy", "hex": "#a855f7"},
    {"name": "Hazardous", "hex": "#37055e"}
]

def create_color_bar():
    """Create color scale visualization"""
    colors = [level['hex'] for level in COLOR_LEVELS]
    labels = [level['name'] for level in COLOR_LEVELS]
    
    html = """
    <div style="height: 25px; border-radius: 12px; overflow: hidden; margin: 20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; height: 100%;">
    """
    
    for color in colors:
        html += f'<div style="flex: 1; background: {color};"></div>'
    
    html += """
        </div>
    </div>
    <div style="display: flex; justify-content: space-between;">
    """
    
    for label in labels:
        html += f'<div style="font-size: 10px; text-align: center; margin-top: 5px; color: #475569; font-weight: 500;">{label}</div>'
    
    html += "</div>"
    return html

# DATA LOADING
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/aqi_dataset_processed.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# SIDEBAR
st.sidebar.header("Prediction Settings")
stations = sorted(df["station"].unique())
station = st.sidebar.selectbox("Select Station", stations)

# FILTER DATA
station_df = df[df["station"] == station].sort_values("date")
if len(station_df) < 40:
    st.error("Not enough historical data for this station.")
    st.stop()

# STATION INFO BOX
last_date = station_df["date"].max().date()
first_date = station_df["date"].min().date()
total_records = len(station_df)

st.markdown(f"""
<div class='info-box'>
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h3 style="margin: 0; color: #1E3A8A;">Station Information</h3>
            <p style="margin: 5px 0 0 0; color: #475569;">
                Station: <strong>{station}</strong><br>
                Data Range: <strong>{first_date.strftime('%d.%m.%Y')} - {last_date.strftime('%d.%m.%Y')}</strong><br>
                Total Records: <strong>{total_records}</strong><br>
                Models: <strong>XGBoost (PM2.5) + GRU (PM10)</strong>
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# COLOR SCALE
st.markdown("### Air Quality Level Scale (Sarajevo Context)")
st.markdown(create_color_bar(), unsafe_allow_html=True)

# AQI SCALE TABLE
with st.expander("📊 Detailed Air Quality Thresholds", expanded=False):
    st.markdown("""
    <table class="aqi-scale">
    <tr>
    <th>Category</th><th>PM2.5 (µg/m³)</th><th>PM10 (µg/m³)</th><th>Context</th>
    </tr>
    <tr class="good"><td> Good</td><td>0–15</td><td>0–50</td><td>Rare in winter; typical clean air days</td></tr>
    <tr class="moderate"><td> Moderate</td><td>16–35</td><td>51–100</td><td>Common autumn/spring level</td></tr>
    <tr class="usg"><td> Poor/USG</td><td>36–75</td><td>101–150</td><td>Often seen during smog episodes</td></tr>
    <tr class="unhealthy"><td> Unhealthy</td><td>76–150</td><td>151–250</td><td>Frequent winter peaks; health effects for most people</td></tr>
    <tr class="very-unhealthy"><td> Very Unhealthy</td><td>151–250</td><td>251–350</td><td>Severe pollution; authorities warn staying indoors</td></tr>
    <tr class="hazardous"><td> Hazardous/Extreme</td><td>>250</td><td>>350</td><td>Episode conditions seen in Sarajevo's worst events</td></tr>
    </table>
    """, unsafe_allow_html=True)

# DISPLAY RECENT HISTORY
with st.expander("📈 Recent Observations (Last 7 Days)", expanded=True):
    recent_data = station_df[["date", "pm25", "pm10", "o3", "no2", "so2"]].tail(7).copy()
    recent_data['date'] = recent_data['date'].dt.strftime('%d.%m.%Y')
    st.dataframe(recent_data, use_container_width=True, hide_index=True)

# LOAD MODELS
try:
    # XGBoost (PM2.5)
    xgb_model = joblib.load(f"models_xgb/xgboost_model_{station}.pkl")
    xgb_X_scaler = joblib.load(f"models_xgb/xgb_X_scaler_{station}.pkl")
    xgb_y_scaler = joblib.load(f"models_xgb/xgb_y_scaler_{station}.pkl")

    # GRU (PM10)
    gru_model = load_model(f"models_gru/pm10_model_{station}.keras")
    gru_scaler = joblib.load(f"models_gru/scaler_{station}.pkl")
    
    st.sidebar.success(f"Models loaded for {station}")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# FEATURE ENGINEERING
def add_gru_features(df):
    d = df.copy()
    d["pm25_lag1"] = d["pm25"].shift(1)
    d["pm25_lag7"] = d["pm25"].shift(7)
    d["pm25_roll7_mean"] = d["pm25"].rolling(7).mean()
    d["doy_sin"] = np.sin(2*np.pi*d["date"].dt.dayofyear/365.25)
    d["doy_cos"] = np.cos(2*np.pi*d["date"].dt.dayofyear/365.25)
    d["o3_inverse"] = 1/(d["o3"]+1)
    return d.dropna()

# 7-DAY FORECAST FUNCTION
def predict_7_days(station_df, xgb_model, xgb_X_scaler, xgb_y_scaler,
                    gru_model, gru_scaler):
    df = station_df.copy()
    features_gru = [
        "pm25","o3","no2","so2","pm10",
        "pm25_lag1","pm25_lag7","pm25_roll7_mean",
        "doy_sin","doy_cos","o3_inverse"
    ]

    preds = []

    for i in range(7):
        # PM2.5 — XGBoost
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        xgb_window = df[numeric_cols].tail(7).values
        xgb_scaled = xgb_X_scaler.transform(xgb_window)
        xgb_input = xgb_scaled.flatten().reshape(1, -1)
        pm25_scaled = xgb_model.predict(xgb_input)
        pm25_pred = xgb_y_scaler.inverse_transform(pm25_scaled.reshape(-1,1))[0,0]

        # PM10 — GRU
        gru_df = add_gru_features(df)
        X_seq = gru_df[features_gru].tail(30).values
        X_scaled = gru_scaler.transform(X_seq).reshape(1,30,len(features_gru))
        residuals = gru_model.predict(X_scaled, verbose=0)
        last_log = np.log1p(gru_df["pm10"].iloc[-1])
        pm10_pred = np.expm1(last_log + residuals[0,0])

        # Date for prediction
        next_date = df["date"].max() + pd.Timedelta(days=1)

        # Append prediction to df for next iteration
        new_row = {
            "date": next_date,
            "pm25": round(pm25_pred, 2),
            "pm10": round(pm10_pred, 2),
            "o3": df["o3"].iloc[-1],
            "no2": df["no2"].iloc[-1],
            "so2": df["so2"].iloc[-1]
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # AQI category
        category, css = get_aqi_category(pm25_pred, pm10_pred)
        
        # Get color for display
        color_map = {
            "good": "#22c55e",
            "moderate": "#eab308",
            "usg": "#f97316",
            "unhealthy": "#ef4444",
            "very-unhealthy": "#a855f7",
            "hazardous": "#37055e"
        }

        preds.append({
            "date": next_date.date(),
            "day_name": next_date.strftime('%A'),
            "day_short": next_date.strftime('%a'),
            "pm25": round(pm25_pred, 1),
            "pm10": round(pm10_pred, 1),
            "AQI": category,
            "color": color_map.get(css, "#6b7280")
        })

    return preds

# 7-DAY FORECAST
st.markdown("---")
st.markdown("### 7-Day Air Quality Forecast")

if st.button("Generate 7-Day Forecast", type="primary"):
    with st.spinner("Generating predictions..."):
        predictions = predict_7_days(station_df, xgb_model, xgb_X_scaler, xgb_y_scaler,
                                     gru_model, gru_scaler)
    
    st.success("Forecast generated successfully!")
    
    # Display predictions in columns
    cols = st.columns(7)
    
    for idx, (pred, col) in enumerate(zip(predictions, cols)):
        with col:
            day_num = idx + 1
            day_prefix = "Tomorrow" if day_num == 1 else f"Day {day_num}"
            
            st.markdown(f"""
            <div class="prediction-day">
                <div class="day-header">{pred['day_short']}<br><small>{day_prefix}</small></div>
                <div style="color: #64748B; font-size: 11px; margin: 5px 0;">
                    {pred['date'].strftime('%d.%m.')}
                </div>
                <div class="metric-label">PM2.5</div>
                <div class="metric-value">{pred['pm25']:.2f}<small>µg/m³</small></div>
                <div class="metric-label">PM10</div>
                <div class="metric-value">{pred['pm10']:.2f}<small>µg/m³</small></div>
                <div class="level-badge" style="background-color: {pred['color']};">
                    {pred['AQI']}
                </div>
            </div>
            """, unsafe_allow_html=True)


    
    # Chart
    st.markdown("### 📊 Pollution Trend Forecast")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    days = [p['day_short'] for p in predictions]
    pm25_vals = [p['pm25'] for p in predictions]
    pm10_vals = [p['pm10'] for p in predictions]
    
    x = range(len(days))
    
    ax.plot(x, pm25_vals, marker='o', linewidth=2, color='#3B82F6', label='PM2.5')
    ax.plot(x, pm10_vals, marker='s', linewidth=2, color='#1E40AF', label='PM10')
    
    ax.set_xlabel('Day', fontsize=12, color='#1E3A8A')
    ax.set_ylabel('Concentration (µg/m³)', fontsize=12, color='#1E3A8A')
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_title(f'7-Day Forecast for {station}', fontsize=14, color='#1E3A8A')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#F8FAFC')
    
    plt.tight_layout()
    st.pyplot(fig)

# CUSTOM DATE PREDICTION
st.markdown("---")
st.markdown("### 📅 Explore Model Predictions for Past Dates")

col1, col2 = st.columns([2, 1])

with col1:
    prediction_date = st.date_input(
        "Select prediction date",
        value=station_df["date"].max().date(),
        min_value=station_df["date"].min().date(),
        max_value=station_df["date"].max().date()
    )

with col2:
    st.write("")
    st.write("")
    predict_custom = st.button("🔮 Predict Air Quality")

if predict_custom:
    # Filter historical data up to selected date
    df_hist = station_df[station_df["date"] < pd.to_datetime(prediction_date)]
    if len(df_hist) < 40:
        st.error("Not enough historical data before this date.")
    else:
        with st.spinner("Generating prediction..."):
            # PM2.5 — XGBoost
            numeric_cols = df_hist.select_dtypes(include=["int64", "float64"]).columns
            xgb_window = df_hist[numeric_cols].tail(7).values
            xgb_scaled = xgb_X_scaler.transform(xgb_window)
            xgb_input = xgb_scaled.flatten().reshape(1, -1)
            pm25_scaled = xgb_model.predict(xgb_input)
            pm25_pred = xgb_y_scaler.inverse_transform(pm25_scaled.reshape(-1,1))[0,0]

            # PM10 — GRU
            gru_df = add_gru_features(df_hist)
            features_gru = [
                "pm25","o3","no2","so2","pm10",
                "pm25_lag1","pm25_lag7","pm25_roll7_mean",
                "doy_sin","doy_cos","o3_inverse"
            ]
            X_seq = gru_df[features_gru].tail(30).values
            X_scaled = gru_scaler.transform(X_seq).reshape(1,30,len(features_gru))
            residuals = gru_model.predict(X_scaled, verbose=0)
            last_log = np.log1p(gru_df["pm10"].iloc[-1])
            pm10_pred = np.expm1(last_log + residuals[0,0])

            # AQI
            category, css = get_aqi_category(pm25_pred, pm10_pred)

        # DISPLAY
        st.success("Prediction complete!")
        st.markdown(f"""
        <div class="metric-box {css}">
            {category}<br>{prediction_date.strftime('%d.%m.%Y')}
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("PM2.5 (µg/m³)", f"{pm25_pred:.2f}")
        c2.metric("PM10 (µg/m³)", f"{pm10_pred:.2f}")

# FOOTER
st.markdown("---")
st.caption(f"Sarajevo Air Quality Prediction • {station} • XGBoost + GRU Models • 2018–2025")
