import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Weather Predictor", page_icon="🌦", layout="wide")

# -------------------- UI STYLE (Premium Purple Theme) --------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #1a0026 0%, #4a0e78 50%, #8a2be2 100%);
        font-family: 'Poppins', sans-serif;
        color: white;
    }

    h1 {
        font-size: 44px !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #ffffff, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    h2, h3 {
        color: #ede9fe;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #14001f;
        border-right: 1px solid #8a2be2;
    }

    /* Premium Button */
    div.stButton > button:first-child {
        background: rgba(124, 58, 237, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 12px;
        color: white;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(124, 58, 237, 0.5);
    }

    div.stButton > button:first-child:hover {
        transform: scale(1.05);
        background: #a78bfa;
        box-shadow: 0 0 25px rgba(167, 139, 250, 0.8);
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.08);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
    }

    .stDataFrame {
        background: white;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🌦 Weather Prediction System")
st.markdown("---")

# -------------------- LOAD DATA + MODEL --------------------
@st.cache_resource
def load_assets():
    df = pd.read_csv("weather_data.csv")
    model = pickle.load(open("model.pkl", "rb"))

    # SAFE feature matching (fixes your KeyError)
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    X = df[model.feature_names_in_]
    y = df["temperature_celsius"]

    preds = model.predict(X)

    metrics = {
        "r2": r2_score(y, preds),
        "mae": mean_absolute_error(y, preds),
        "mse": mean_squared_error(y, preds)
    }

    return df, model, metrics

try:
    df, model, metrics = load_assets()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# -------------------- METRICS --------------------
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"<div class='card'><h3>R² Score</h3><h2>{metrics['r2']:.3f}</h2></div>", unsafe_allow_html=True)

with m2:
    st.markdown(f"<div class='card'><h3>MAE</h3><h2>{metrics['mae']:.2f}</h2></div>", unsafe_allow_html=True)

with m3:
    st.markdown(f"<div class='card'><h3>MSE</h3><h2>{metrics['mse']:.2f}</h2></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- USER INPUT --------------------
st.sidebar.header("⚙️ Input Features")

def user_input():
    with st.sidebar:
        latitude = st.slider("Latitude", -50.0, 70.0, 20.0)
        longitude = st.slider("Longitude", -180.0, 180.0, 50.0)
        humidity = st.slider("Humidity (%)", 0, 100, 50)
        wind_mph = st.slider("Wind Speed", 0.0, 50.0, 10.0)
        pressure_mb = st.slider("Pressure", 900.0, 1100.0, 1010.0)
        cloud = st.slider("Cloud", 0, 100, 50)
        uv_index = st.slider("UV Index", 0.0, 12.0, 5.0)
        precip_mm = st.slider("Precipitation", 0.0, 50.0, 0.0)
        month = st.select_slider("Month", options=list(range(1, 13)), value=6)

    return pd.DataFrame([{
        "latitude": latitude,
        "longitude": longitude,
        "humidity": humidity,
        "wind_mph": wind_mph,
        "pressure_mb": pressure_mb,
        "cloud": cloud,
        "uv_index": uv_index,
        "precip_mm": precip_mm,
        "month": month
    }])

input_df = user_input()

# -------------------- PREDICTION --------------------
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("🌡 Prediction")

    input_ready = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(input_ready)

    st.markdown(f"<div class='card'><h1>{prediction[0]:.2f} °C</h1></div>", unsafe_allow_html=True)

    if st.button("Generate Report"):
        st.success("Report generated!")

with col2:
    st.subheader("📋 Input Data")
    st.dataframe(input_df, use_container_width=True)

# -------------------- VISUALIZATION --------------------
st.markdown("### 📊 Analytics")

tab1, tab2, tab3 = st.tabs(["Histogram", "Boxplot", "Counterplot"])

with tab1:
    fig, ax = plt.subplots()
    sns.histplot(df["temperature_celsius"], kde=True, color="#a78bfa", ax=ax)
    ax.set_facecolor("#14001f")
    ax.tick_params(colors='white')
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots()
    sns.boxplot(x="month", y="temperature_celsius", data=df, palette="Purples", ax=ax)
    ax.set_facecolor("#14001f")
    ax.tick_params(colors='white')
    st.pyplot(fig)
with tab3:
    st.subheader("🌦 Top 10 Weather Conditions")

    # Get top 10 conditions
    top_conditions = df['condition_text'].value_counts().head(10)
    total = top_conditions.sum()

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    # Countplot
    sns.countplot(
        data=df[df['condition_text'].isin(top_conditions.index)],
        y='condition_text',
        order=top_conditions.index,
        palette="Purples",
        ax=ax3
    )

    # Add percentage labels
    for i, value in enumerate(top_conditions):
        percent = (value / total) * 100
        ax3.text(value + 5, i, f"{percent:.1f}%", va='center')

    # Styling
    ax3.set_title("Top 10 Weather Conditions", fontsize=14, fontweight="bold", color="#4c1d95")
    ax3.set_xlabel("Count")
    ax3.set_ylabel("Condition")
    ax3.set_facecolor("#faf5ff")

    st.pyplot(fig3, use_container_width=True)
# -------------------- FOOTER --------------------
st.info("💡 Random Forest Model | Purple UI Enhanced")