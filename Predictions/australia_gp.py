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

# ✅ Import Streamlit secrets safely (for API key)
try:
    import streamlit as st
    API_KEY = st.secrets.get("WEATHER_API_KEY", "")
except Exception:
    # fallback if running locally
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv("WeatherAPI", "")

QUALI_WEIGHT = 2.0


def run_prediction(cache_path="Predictions/cache_australia"):
    print(f"✅ Enabling cache at: {cache_path}")
    fastf1.Cache.enable_cache(cache_path)

    # ----------------------------
    # Collect historical race data
    data_years = []
    for yr, rnd in [(2022, 3), (2023, 2), (2024, 3)]:
        try:
            qual = fastf1.get_session(yr, rnd, "Q")
            race = fastf1.get_session(yr, rnd, "R")
            qual.load()
            race.load()

            qual_times = qual.results[["Abbreviation", "Q3", "Q2", "Q1"]].copy()
            qual_times.rename(columns={"Abbreviation": "Driver"}, inplace=True)

            for col in ["Q1", "Q2", "Q3"]:
                if col in qual_times.columns:
                    qual_times[col] = pd.to_timedelta(qual_times[col], errors="coerce").dt.total_seconds()

            qual_times["QualifyingTime (s)"] = qual_times[["Q1", "Q2", "Q3"]].min(axis=1)

            race_results = race.results[["Abbreviation", "Position"]].rename(
                columns={"Abbreviation": "Driver", "Position": "RacePosition"}
            )

            merged = pd.merge(qual_times, race_results, on="Driver", how="inner")
            merged["Season"] = yr
            data_years.append(merged)
        except Exception as e:
            print(f"Error loading {yr} Australian GP: {e}")

    if not data_years:
        print("❌ No historical data found.")
        return None

    historical = pd.concat(data_years, ignore_index=True).copy()

    # ----------------------------
    # Static features
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

    # ----------------------------
    # Weather data (Melbourne)
    rain_probability, temperature = 0, 22  # defaults
    try:
        if not API_KEY:
            raise ValueError("❌ WeatherAPI key missing!")

        params = {"key": API_KEY, "q": "-37.8497,144.968", "days": 1, "aqi": "no", "alerts": "no"}
        resp = requests.get("https://api.weatherapi.com/v1/forecast.json", params=params, timeout=10)

        if resp.status_code == 200:
            weather = resp.json()
            forecast = weather.get("forecast", {}).get("forecastday", [])
            if forecast:
                hour = forecast[0].get("hour", [])[0]
                rain_probability = hour.get("chance_of_rain", 0) / 100
                temperature = hour.get("temp_c", 22)
        else:
            print(f"⚠️ Weather API error: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Weather data fetch failed: {e}")

    historical.loc[:, "RainProbability"] = rain_probability
    historical.loc[:, "Temperature"] = temperature

    # ----------------------------
    # Train model
    features = [
        "QualifyingTime (s)", "TeamPerformanceScore",
        "CleanAirRacePace (s)", "RainProbability", "Temperature"
    ]
    X = historical[features].copy()
    X["QualifyingTime (s)"] *= QUALI_WEIGHT
    y = historical["RacePosition"]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    print(f"📊 MAE on validation: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")

    # ----------------------------
    # Predict Australian GP 2025
    qual_2025 = pd.DataFrame({
        "Driver": [
            "NOR", "PIA", "VER", "RUS", "TSU", "ALB", "LEC", "HAM", "GAS", "SAI",
            "HAD", "ALO", "STR", "DOO", "BOR", "ANT", "HUL", "LAW", "OCO", "BEA"
        ],
        "QualifyingTime (s)": [
            75.096, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755, 75.919, 75.980, 75.931,
            76.175, 76.288, 76.369, 76.315, 76.516, 76.525, 76.579, 77.094, 77.147, np.nan
        ]
    })

    qual_2025["QualifyingTime (s)"].replace([np.inf, -np.inf], np.nan, inplace=True)
    qual_2025["QualifyingTime (s)"].fillna(qual_2025["QualifyingTime (s)"].median(), inplace=True)

    qual_2025["Team"] = qual_2025["Driver"].map(driver_team)
    qual_2025["TeamPerformanceScore"] = qual_2025["Team"].map(team_perf)
    qual_2025["CleanAirRacePace (s)"] = qual_2025["Driver"].map(clean_air_race_pace)

    for feature in ["TeamPerformanceScore", "CleanAirRacePace (s)"]:
        qual_2025.loc[:, feature].fillna(historical[feature].median(), inplace=True)

    qual_2025["RainProbability"] = rain_probability
    qual_2025["Temperature"] = temperature

    X_pred = qual_2025[features].copy()
    X_pred["QualifyingTime (s)"] *= QUALI_WEIGHT
    X_pred = imputer.transform(X_pred)
    qual_2025["PredictedPosition"] = model.predict(X_pred)
    qual_2025 = qual_2025.sort_values("PredictedPosition")

    print("\n🏁 Predicted Australian GP 2025 Results 🏁")
    print(qual_2025[["Driver", "PredictedPosition"]])

    print("\n🏆 Predicted Podium 🏆")
    print(f"🥇 P1: {qual_2025.iloc[0]['Driver']}")
    print(f"🥈 P2: {qual_2025.iloc[1]['Driver']}")
    print(f"🥉 P3: {qual_2025.iloc[2]['Driver']}")

    plt.figure(figsize=(8, 5))
    plt.barh(features, model.feature_importances_, color="salmon")
    plt.title("Feature Importance (Australian GP Race Prediction)")
    plt.tight_layout()
    plt.show()

    return qual_2025[["Driver", "PredictedPosition"]]
