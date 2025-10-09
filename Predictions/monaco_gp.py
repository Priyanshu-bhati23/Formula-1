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

QUALI_WEIGHT = 2.0


@st.cache_data(show_spinner=False)
def run_prediction(cache_path="Predictions/cache_monaco"):
    """
    Loads F1 data (with persistent Streamlit caching), trains model, 
    and returns 2025 Monaco GP predictions.
    """

    # -------------------------------------------------
    # ⚙️ Handle caching (local or Streamlit Cloud)
    if os.path.exists(cache_path):
        fastf1.Cache.enable_cache(cache_path)
        print(f"✅ Using local cache: {cache_path}")
    else:
        cloud_cache = os.path.join(st.cache_data.get_cache_path(), "fastf1_cache")
        os.makedirs(cloud_cache, exist_ok=True)
        fastf1.Cache.enable_cache(cloud_cache)
        print(f"☁️ Using Streamlit cloud cache: {cloud_cache}")

    # -------------------------------------------------
    # 🏎️ Load historical data (Monaco + early 2025)
    data_years = []
    for yr, rnd in [(2022, 7), (2023, 7), (2024, 8)]:
        try:
            qual = fastf1.get_session(yr, rnd, "Q")
            race = fastf1.get_session(yr, rnd, "R")
            qual.load()
            race.load()

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

    for rnd in range(1, 8):
        try:
            qual = fastf1.get_session(2025, rnd, "Q")
            race = fastf1.get_session(2025, rnd, "R")
            qual.load()
            race.load()

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
        st.error("❌ No historical data found.")
        return None

    historical = pd.concat(data_years, ignore_index=True).copy()

    # -------------------------------------------------
    # 🧠 Static data
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
    # 🌦️ Weather
    load_dotenv()
    API_KEY = os.getenv("WeatherAPI", "")
    params = {"key": API_KEY, "q": "43.7384,7.4246", "days": 1, "aqi": "no", "alerts": "no"}
    try:
        resp = requests.get("http://api.weatherapi.com/v1/forecast.json", params=params, timeout=10)
        weather = resp.json()
        forecast = weather.get("forecast", {}).get("forecastday", [])
        rain_probability, temperature = 0, 22
        if forecast:
            hour = forecast[0].get("hour", [])[0]
            rain_probability = hour.get("chance_of_rain", 0) / 100
            temperature = hour.get("temp_c", 22)
    except Exception as e:
        print(f"Weather API failed: {e}")
        rain_probability, temperature = 0, 22

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

    X_pred = qual_2025[features].copy()
    X_pred["QualifyingTime (s)"] *= QUALI_WEIGHT
    X_pred = imputer.transform(X_pred)
    qual_2025["PredictedPosition"] = model.predict(X_pred)
    qual_2025.sort_values("PredictedPosition", inplace=True)

    # -------------------------------------------------
    # 🎯 Display Results
    st.subheader("🏁 Predicted 2025 Monaco GP Results")
    st.dataframe(qual_2025[["Driver", "PredictedPosition"]])

    st.subheader("🏆 Predicted Podium")
    st.write(f"🥇 **{qual_2025.iloc[0]['Driver']}**")
    st.write(f"🥈 **{qual_2025.iloc[1]['Driver']}**")
    st.write(f"🥉 **{qual_2025.iloc[2]['Driver']}**")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, model.feature_importances_, color="salmon")
    ax.set_title("Feature Importance (Monaco GP Race Prediction)")
    st.pyplot(fig)

    return qual_2025[["Driver", "PredictedPosition"]]
