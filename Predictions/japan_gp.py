import fastf1
import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

def run_prediction(cache_path="Predictions/cache_japan"):
    print(f"✅ Enabling cache at: {cache_path}")
    fastf1.Cache.enable_cache(cache_path)

    QUALI_WEIGHT = 2.0
    data_years = []

    # Historical Japanese GP data
    for yr, rnd in [(2022, 17), (2023, 17), (2024, 17)]:
        try:
            qual = fastf1.get_session(yr, rnd, "Q")
            race = fastf1.get_session(yr, rnd, "R")
            qual.load(); race.load()

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
            print(f"Error loading {yr} Japanese GP: {e}")

    # 2025 races before Japanese GP
    for yr, rnd in [(2025, 1), (2025, 2)]:
        try:
            qual = fastf1.get_session(yr, rnd, "Q")
            race = fastf1.get_session(yr, rnd, "R")
            qual.load(); race.load()

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
            print(f"Error loading {yr} race {rnd}: {e}")

    if not data_years:
        print("❌ No historical data found.")
        return pd.DataFrame()

    historical = pd.concat(data_years, ignore_index=True)

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
    team_perf = {t: pts/max_points for t, pts in team_points.items()}
    driver_team = {
        "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
        "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
        "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
    }
    historical["Team"] = historical["Driver"].map(driver_team)
    historical["CleanAirRacePace (s)"] = historical["Driver"].map(clean_air_race_pace)
    historical["TeamPerformanceScore"] = historical["Team"].map(team_perf)

    # Weather data
    load_dotenv()
    API_KEY = os.getenv("WeatherAPI", "")
    params = {"key": API_KEY, "q": "34.8431,136.5419", "days": 1, "aqi": "no", "alerts": "no"}
    resp = requests.get("http://api.weatherapi.com/v1/forecast.json", params=params)
    weather = resp.json()
    forecast = weather.get("forecast", {}).get("forecastday", [])
    rain_probability = 0; temperature = 22
    if forecast:
        hour = forecast[0].get("hour", [])[0]
        rain_probability = hour.get("chance_of_rain", 0) / 100
        temperature = hour.get("temp_c", 22)
    historical["RainProbability"] = rain_probability
    historical["Temperature"] = temperature

    # Train model
    features = ["QualifyingTime (s)", "TeamPerformanceScore", "CleanAirRacePace (s)", "RainProbability", "Temperature"]
    X = historical[features].copy()
    X["QualifyingTime (s)"] *= QUALI_WEIGHT
    y = historical["RacePosition"]
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Predict Japanese GP
    qual_2025 = historical[historical["Season"] == 2025].drop_duplicates(subset=["Driver"])
    qual_2025["RainProbability"] = rain_probability
    qual_2025["Temperature"] = temperature
    qual_2025["QualifyingTime (s)"].replace([np.inf, -np.inf], np.nan, inplace=True)
    qual_2025["QualifyingTime (s)"].fillna(qual_2025["QualifyingTime (s)"].median(), inplace=True)
    qual_2025["Team"] = qual_2025["Driver"].map(driver_team)
    qual_2025["TeamPerformanceScore"] = qual_2025["Team"].map(team_perf)
    qual_2025["CleanAirRacePace (s)"] = qual_2025["Driver"].map(clean_air_race_pace)
    for feature in ["TeamPerformanceScore", "CleanAirRacePace (s)"]:
        qual_2025.loc[:, feature].fillna(historical[feature].median(), inplace=True)

    X_pred = qual_2025[features].copy()
    X_pred["QualifyingTime (s)"] *= QUALI_WEIGHT
    X_pred = imputer.transform(X_pred)
    qual_2025["PredictedPosition"] = model.predict(X_pred)
    qual_2025 = qual_2025.sort_values("PredictedPosition").reset_index(drop=True)

    print("\n🏁 Predicted Japanese GP 2025 Results 🏁")
    print(qual_2025[["Driver", "PredictedPosition"]])

    return qual_2025[["Driver", "PredictedPosition"]]
