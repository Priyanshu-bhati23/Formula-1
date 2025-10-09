import fastf1
import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import streamlit as st
import warnings

# Suppress pandas and fastf1 warnings for clean Streamlit output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
QUALI_WEIGHT = 2.0
MONACO_WEATHER_COORD = "43.7384,7.4246" # Monaco coordinates

# Set Streamlit page configuration
st.set_page_config(layout="centered", page_title="F1 Race Prediction")

# --- Model & Data Logic (Cached) ---

@st.cache_data(show_spinner="Loading F1 data and training model...", ttl=60*60*4) # Cache for 4 hours
def run_prediction(cache_path="Predictions/cache_monaco"):
    """
    Loads F1 data (with persistent Streamlit caching), trains model, 
    and returns 2025 Monaco GP predictions.
    """
    st.write("### ⚙️ Loading Data and Model Training")
    
    # -------------------------------------------------
    # ⚙️ Handle caching (local or Streamlit Cloud)
    # fastf1 cache setup
    if os.path.exists(cache_path):
        fastf1.Cache.enable_cache(cache_path)
        print(f"✅ Using local cache: {cache_path}")
    else:
        # Use Streamlit's cache directory for persistence in cloud deployments
        # Note: st.cache_data.get_cache_path() might not be reliable across all deployments
        cloud_cache = os.path.join("fastf1_cache") # Simple directory name
        os.makedirs(cloud_cache, exist_ok=True)
        fastf1.Cache.enable_cache(cloud_cache)
        print(f"☁️ Using Streamlit cloud cache folder: {cloud_cache}")

    # -------------------------------------------------
    # 🏎️ Load historical data (Monaco + early 2025)
    data_years = []
    
    # Historical Monaco GPs
    for yr, rnd in [(2022, 7), (2023, 7), (2024, 8)]:
        try:
            qual = fastf1.get_session(yr, rnd, "Q")
            race = fastf1.get_session(yr, rnd, "R")
            qual.load(telemetry=False, weather=False); race.load(telemetry=False, weather=False)

            qual_times = qual.results[["Abbreviation", "Q3", "Q2", "Q1"]].copy()
            qual_times.rename(columns={"Abbreviation": "Driver"}, inplace=True)
            for col in ["Q1", "Q2", "Q3"]:
                qual_times[col] = pd.to_timedelta(qual_times[col], errors="coerce").dt.total_seconds()
            qual_times["QualifyingTime (s)"] = qual_times[["Q1", "Q2", "Q3"]].min(axis=1)

            race_results = race.results[["Abbreviation", "Position"]].rename(
                columns={"Abbreviation": "Driver", "Position": "RacePosition"}
            )

            merged = pd.merge(qual_times, race_results, on="Driver", how="inner")
            merged["Season"] = yr
            data_years.append(merged)
        except Exception as e:
            print(f"Error loading {yr} Monaco GP: {e}")

    # Early 2025 races for current form (Rounds 1 through 7)
    for rnd in range(1, 8):
        try:
            qual = fastf1.get_session(2025, rnd, "Q")
            race = fastf1.get_session(2025, rnd, "R")
            qual.load(telemetry=False, weather=False); race.load(telemetry=False, weather=False)

            qual_times = qual.results[["Abbreviation", "Q3", "Q2", "Q1"]].copy()
            qual_times.rename(columns={"Abbreviation": "Driver"}, inplace=True)
            for col in ["Q1", "Q2", "Q3"]:
                qual_times[col] = pd.to_timedelta(qual_times[col], errors="coerce").dt.total_seconds()
            qual_times["QualifyingTime (s)"] = qual_times[["Q1", "Q2", "Q3"]].min(axis=1)

            race_results = race.results[["Abbreviation", "Position"]].rename(
                columns={"Abbreviation": "Driver", "Position": "RacePosition"}
            )

            merged = pd.merge(qual_times, race_results, on="Driver", how="inner")
            merged["Season"] = 2025
            merged["Round"] = rnd
            data_years.append(merged)
        except Exception as e:
            print(f"Error loading 2025 round {rnd}: {e}")

    if not data_years:
        return None

    historical = pd.concat(data_years, ignore_index=True).copy()

    # -------------------------------------------------
    # 🧠 Static data & Feature Mapping
    clean_air_race_pace = {
        "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
        "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
        "OCO": 95.682128
    }
    team_points = {
        "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51, "Ferrari": 114,
        "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 10, "Alpine": 7
    }
    max_points = max(team_points.values())
    team_perf = {t: pts / max_points for t, pts in team_points.items()}
    driver_team = {
        "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
        "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
        "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
    }

    historical["Team"] = historical["Driver"].map(driver_team)
    historical["CleanAirRacePace (s)"] = historical["Driver"].map(clean_air_race_pace)
    historical["TeamPerformanceScore"] = historical["Team"].map(team_perf)
    historical.dropna(subset=["RacePosition", "QualifyingTime (s)"], inplace=True)

    # -------------------------------------------------
    # 🌦️ Weather API Call
    load_dotenv()
    API_KEY = os.getenv("WeatherAPI", "")
    rain_probability, temperature = 0, 22 # Default if API fails

    if API_KEY:
        params = {"key": API_KEY, "q": MONACO_WEATHER_COORD, "days": 1, "aqi": "no", "alerts": "no"}
        try:
            resp = requests.get("http://api.weatherapi.com/v1/forecast.json", params=params, timeout=5)
            weather = resp.json()
            forecast = weather.get("forecast", {}).get("forecastday", [])
            if forecast:
                # Use the first hour's forecast as a proxy
                hour = forecast[0].get("hour", [])[0]
                rain_probability = hour.get("chance_of_rain", 0) / 100
                temperature = hour.get("temp_c", 22)
            st.info(f"☁️ Monaco Weather: Rain Chance: {rain_probability*100:.0f}%, Temp: {temperature}°C")
        except Exception as e:
            st.warning(f"⚠️ Weather API failed (defaulting to clear/22°C): {e}")

    historical["RainProbability"] = rain_probability
    historical["Temperature"] = temperature

    # -------------------------------------------------
    # 🧮 Train Model
    features = ["QualifyingTime (s)", "TeamPerformanceScore", "CleanAirRacePace (s)", "RainProbability", "Temperature"]
    X = historical[features].copy()
    X["QualifyingTime (s)"] *= QUALI_WEIGHT
    y = historical["RacePosition"]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    st.write(f"**MAE on validation:** {mean_absolute_error(y_test, model.predict(X_test)):.2f}")

    # -------------------------------------------------
    # 🔮 Predict 2025 Monaco GP
    # Hardcoded 2025 Qualifying Results for the prediction input
    qual_2025 = pd.DataFrame({
        "Driver": ["NOR", "LEC", "PIA", "VER", "HAM", "ALB", "HAD", "ALO", "OCO", "LAW",
                   "SAI", "TSU", "HUL", "ANT", "BOR", "BEA", "GAS", "STR", "COL", "RUS"],
        "QualifyingTime (s)": [69.954, 70.063, 70.129, 70.669, 70.382, 70.732, 70.923, 70.924, 70.942, 71.129,
                               71.362, 71.415, 71.596, 71.880, 71.902, 71.979, 71.994, 72.563, 72.597, 77.597]
    })

    qual_2025["Team"] = qual_2025["Driver"].map(driver_team)
    qual_2025["TeamPerformanceScore"] = qual_2025["Team"].map(team_perf)
    qual_2025["CleanAirRacePace (s)"] = qual_2025["Driver"].map(clean_air_race_pace)
    qual_2025["RainProbability"] = rain_probability
    qual_2025["Temperature"] = temperature

    # Prepare features for prediction
    X_pred = qual_2025[features].copy()
    X_pred["QualifyingTime (s)"] *= QUALI_WEIGHT
    X_pred = imputer.transform(X_pred) # Use the imputer fitted on the training data

    # Predict and sort
    qual_2025["PredictedPosition"] = model.predict(X_pred)
    qual_2025.sort_values("PredictedPosition", inplace=True)
    
    # Return results and model for display outside the cached function
    return qual_2025[["Driver", "PredictedPosition"]], model, features


# --- Streamlit App Entry Point ---

def main():
    st.title("Formula 1 🏎️ Monaco GP Race Prediction")
    st.markdown("---")
    
    # Run the prediction function (it will use the cache if possible)
    results_and_model = run_prediction()
    
    if results_and_model is None:
        return

    qual_2025, model, features = results_and_model

    # -------------------------------------------------
    # 🎯 Display Results
    st.subheader("🏁 Predicted 2025 Monaco GP Results")
    st.dataframe(qual_2025.reset_index(drop=True).style.format({"PredictedPosition": "{:.2f}"}))

    st.subheader("🏆 Predicted Podium")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🥇 P1", f"**{qual_2025.iloc[0]['Driver']}**", delta=f"Score: {qual_2025.iloc[0]['PredictedPosition']:.2f}")
    with col2:
        st.metric("🥈 P2", f"**{qual_2025.iloc[1]['Driver']}**", delta=f"Score: {qual_2025.iloc[1]['PredictedPosition']:.2f}")
    with col3:
        st.metric("🥉 P3", f"**{qual_2025.iloc[2]['Driver']}**", delta=f"Score: {qual_2025.iloc[2]['PredictedPosition']:.2f}")

    st.markdown("---")

    # -------------------------------------------------
    # 📊 Feature Importance Plot
    st.subheader("📊 Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, model.feature_importances_, color="teal")
    ax.set_title("Feature Importance (Monaco GP Race Prediction)")
    ax.set_xlabel("Relative Importance")
    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()