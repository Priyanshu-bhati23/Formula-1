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

# Suppress pandas and fastf1 warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIG ---
QUALI_WEIGHT = 2.0
BAHRAIN_WEATHER_COORD = "26.0325,50.5106" 

# Set Streamlit page configuration
st.set_page_config(layout="centered", page_title="F1 Bahrain Prediction")

# Define static features outside the cached function for clarity
# (assuming these are constant across predictions, as per your previous code)
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
features = ["QualifyingTime (s)", "TeamPerformanceScore", "CleanAirRacePace (s)", "RainProbability", "Temperature"]


@st.cache_data(show_spinner="Loading F1 data and training model for Bahrain...", ttl=60*60*4)
def run_prediction(cache_path="cache_bahrain"):
    st.write("### ⚙️ Loading Data and Model Training")
    
    # -------------------------------------------------
    # ⚙️ Handle FastF1 caching 
    fastf1.Cache.enable_cache(cache_path)
    print(f"✅ Using FastF1 cache path: {cache_path}")

    # -------------------------------------------------
    # 🏎️ Load historical Bahrain GP data (Round 1)
    data_years = []
    
    for yr, rnd in [(2022, 1), (2023, 1), (2024, 1)]:
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
            print(f"Error loading {yr} Bahrain GP: {e}")

    if not data_years:
        # 🚨 FIX for structural error: Return an empty DataFrame, not None
        # This prevents the outer app from failing the '.empty' check on a 'tuple'
        st.error("❌ No historical data found. Cannot train model.")
        return pd.DataFrame(), None, None # Return empty DataFrame, None model, None features

    historical = pd.concat(data_years, ignore_index=True).copy()

    # -------------------------------------------------
    # 🧠 Feature Mapping
    historical["Team"] = historical["Driver"].map(driver_team)
    historical["CleanAirRacePace (s)"] = historical["Driver"].map(clean_air_race_pace)
    historical["TeamPerformanceScore"] = historical["Team"].map(team_perf)
    historical.dropna(subset=["RacePosition", "QualifyingTime (s)"], inplace=True)

    # -------------------------------------------------
    # 🌦️ Weather API Call (Bahrain)
    # ... (Weather API logic as before) ...
    load_dotenv()
    API_KEY = os.getenv("WeatherAPI", "")
    rain_probability, temperature = 0, 22 

    if API_KEY:
        params = {"key": API_KEY, "q": BAHRAIN_WEATHER_COORD, "days": 1, "aqi": "no", "alerts": "no"}
        try:
            resp = requests.get("http://api.weatherapi.com/v1/forecast.json", params=params, timeout=5)
            weather = resp.json()
            forecast = weather.get("forecast", {}).get("forecastday", [])
            if forecast:
                hour = forecast[0].get("hour", [])[0]
                rain_probability = hour.get("chance_of_rain", 0) / 100
                temperature = hour.get("temp_c", 22)
            st.info(f"☁️ Bahrain Weather: Rain Chance: {rain_probability*100:.0f}%, Temp: {temperature}°C")
        except Exception as e:
            st.warning(f"⚠️ Weather API failed (defaulting to clear/22°C): {e}")

    historical["RainProbability"] = rain_probability
    historical["Temperature"] = temperature

    # -------------------------------------------------
    # 🧮 Train Model
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
    # 🔮 Predict 2025 Bahrain GP (Hardcoded Qualifying Results)
    # ... (Prediction input data as before) ...
    qual_2025 = pd.DataFrame({
        "Driver": [
            "VER", "NOR", "PIA", "LEC", "RUS", "HAM", "SAI", "ALO", "GAS", "OCO",
            "STR", "TSU", "HUL", "ALB", "LAW", "BOT", "ZHO", "RIC", "MAG", "SAR"
        ],
        "QualifyingTime (s)": [
            89.210, 89.315, 89.440, 89.480, 89.522, 89.540, 89.565, 89.643, 89.701, 89.712,
            89.735, 89.840, 89.900, 89.950, 90.010, 90.050, 90.070, 90.150, 90.210, 90.250
        ]
    })

    qual_2025["Team"] = qual_2025["Driver"].map(driver_team)
    qual_2025["TeamPerformanceScore"] = qual_2025["Team"].map(team_perf)
    qual_2025["CleanAirRacePace (s)"] = qual_2025["Driver"].map(clean_air_race_pace)
    qual_2025["RainProbability"] = rain_probability
    qual_2025["Temperature"] = temperature

    for feature in ["TeamPerformanceScore", "CleanAirRacePace (s)"]:
        qual_2025[feature].fillna(historical[feature].median(), inplace=True)
        
    X_pred = qual_2025[features].copy()
    X_pred["QualifyingTime (s)"] *= QUALI_WEIGHT
    X_pred = imputer.transform(X_pred)

    qual_2025["PredictedPosition"] = model.predict(X_pred)
    qual_2025.sort_values("PredictedPosition", inplace=True)
    
    # Return results, model, and features as a tuple
    return qual_2025[["Driver", "PredictedPosition"]], model, features


# --- Streamlit App Entry Point ---

def main():
    st.title("Formula 1 🇧🇭 Bahrain GP Race Prediction")
    st.markdown("---")
    
    # Run the prediction function 
    results_and_model = run_prediction()
    
    # 🚨 FIX: Correctly check for success (results_and_model is a tuple)
    # and unpack the result.
    if results_and_model[0] is None or results_and_model[0].empty:
        # This catches both the 'None' model/features and the empty DataFrame returned on failure.
        st.error("Prediction failed: Could not load required historical data or train model.")
        return

    # UNPACK the tuple correctly
    qual_2025, model, features = results_and_model

    # -------------------------------------------------
    # 🎯 Display Results
    st.subheader("🏁 Predicted 2025 Bahrain GP Results")
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
    ax.set_title("Feature Importance (Bahrain GP Race Prediction)")
    ax.set_xlabel("Relative Importance")
    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()