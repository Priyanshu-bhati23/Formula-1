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

QUALI_WEIGHT = 2.0

def run_prediction(cache_path="Predictions/cache_china"):
    print(f"✅ Enabling cache at: {cache_path}")
    fastf1.Cache.enable_cache(cache_path)

    # ----------------------------
    # Load cached data for 2022–2024 Chinese GPs
    data_years = []
    for yr in [2022, 2023, 2024]:
        try:
            qual = fastf1.get_session(yr, 5, "Q")
            race = fastf1.get_session(yr, 5, "R")
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
            print(f"⚠️ Error loading {yr} Chinese GP from cache: {e}")

    if not data_years:
        print("❌ No cached data found for Chinese GP.")
        return None

    historical = pd.concat(data_years, ignore_index=True).copy()

    # ----------------------------
    # Static features
    clean_air_race_pace = {
        "VER": 93.050, "HAM": 94.100, "LEC": 93.500, "NOR": 93.400, "ALO": 94.700,
        "PIA": 93.300, "RUS": 93.900, "SAI": 94.400, "STR": 95.200, "HUL": 95.300,
        "OCO": 95.600
    }

    team_points = {
        "McLaren": 246, "Mercedes": 141, "Red Bull": 105, "Williams": 37,
        "Ferrari": 94, "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6,
        "Racing Bulls": 8, "Alpine": 7
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
    historical = historical.dropna(subset=["RacePosition", "QualifyingTime (s)"])

    # ----------------------------
    # Weather data for Shanghai (China GP)
    load_dotenv()
    API_KEY = os.getenv("WeatherAPI", "")
    rain_probability, temperature = 0, 22  # defaults

    if API_KEY:
        try:
            params = {
                "key": API_KEY,
                "q": "31.3389,121.2217",
                "days": 1,
                "aqi": "no",
                "alerts": "no"
            }
            resp = requests.get("http://api.weatherapi.com/v1/forecast.json", params=params)
            weather = resp.json()
            forecast = weather.get("forecast", {}).get("forecastday", [])
            if forecast:
                hour = forecast[0].get("hour", [])[0]
                rain_probability = hour.get("chance_of_rain", 0) / 100
                temperature = hour.get("temp_c", 22)
        except Exception as e:
            print(f"⚠️ Weather API error: {e}")

    historical["RainProbability"] = rain_probability
    historical["Temperature"] = temperature

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
    # Predicted 2025 Chinese GP
    qual_2025 = pd.DataFrame({
        "Driver": [
            "PIA", "RUS", "NOR", "VER", "HAM", "LEC", "HAD", "ANT", "TSU", "ALB",
            "OCO", "HUL", "ALO", "STR", "SAI", "GAS", "BEA", "DOO", "BOR", "LAW"
        ],
        "QualifyingTime (s)": [
            90.641, 90.723, 90.793, 90.817, 90.927,
            91.021, 91.079, 91.103, 91.638, 91.706,
            91.625, 91.632, 91.688, 91.773, 91.840,
            91.992, 92.018, 92.092, 92.141, 92.174
        ]
    })

    try:
        race_2025 = fastf1.get_session(2025, 5, "R")
        race_2025.load()
        race_results = race_2025.results[["Abbreviation", "Position"]].rename(
            columns={"Abbreviation": "Driver", "Position": "RacePosition"}
        )
        qual_2025 = pd.merge(qual_2025, race_results, on="Driver", how="left")
    except Exception as e:
        print(f"⚠️ Could not load 2025 race data: {e}")
        qual_2025["RacePosition"] = np.nan

    qual_2025["Team"] = qual_2025["Driver"].map(driver_team)
    qual_2025["TeamPerformanceScore"] = qual_2025["Team"].map(team_perf)
    qual_2025["CleanAirRacePace (s)"] = qual_2025["Driver"].map(clean_air_race_pace)
    qual_2025["RainProbability"] = rain_probability
    qual_2025["Temperature"] = temperature

    X_pred = qual_2025[features].copy()
    X_pred["QualifyingTime (s)"] *= QUALI_WEIGHT
    X_pred = imputer.transform(X_pred)
    qual_2025["PredictedPosition"] = model.predict(X_pred)
    qual_2025 = qual_2025.sort_values("PredictedPosition")

    print("\n🏁 Predicted 2025 Chinese GP Results 🏁")
    print(qual_2025[["Driver", "PredictedPosition", "RacePosition"]])

    print("\n🏆 Predicted Podium 🏆")
    print(f"🥇 P1: {qual_2025.iloc[0]['Driver']}")
    print(f"🥈 P2: {qual_2025.iloc[1]['Driver']}")
    print(f"🥉 P3: {qual_2025.iloc[2]['Driver']}")

    plt.figure(figsize=(8, 5))
    plt.barh(features, model.feature_importances_, color="salmon")
    plt.title("Feature Importance (Chinese GP Race Prediction)")
    plt.tight_layout()
    plt.show()

    return qual_2025[["Driver", "PredictedPosition", "RacePosition"]]
