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
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def run_prediction(cache_path="Predictions/cache_japan"):
    print(f"✅ Enabling cache at: {cache_path}")
    fastf1.Cache.enable_cache(cache_path)

    QUALI_WEIGHT = 2.0
    data_years = []

    # ----------------------------------------------------
    # 1. Historical Data Loading (Japanese GP 2022-2024)
    # ----------------------------------------------------
    print("\n⏳ Loading historical Japanese GP data (2022-2024)...")
    # Japanese GP is typically round 17, but it can vary by year
    for yr, rnd in [(2022, 17), (2023, 17), (2024, 4)]: # 2024 was Round 4
        try:
            qual = fastf1.get_session(yr, rnd, "Q")
            race = fastf1.get_session(yr, rnd, "R")
            qual.load(telemetry=False, weather=False); race.load(telemetry=False, weather=False)

            qual_times = qual.results[["Abbreviation", "Q3", "Q2", "Q1"]].copy()
            qual_times.rename(columns={"Abbreviation": "Driver"}, inplace=True)

            for col in ["Q1", "Q2", "Q3"]:
                if col in qual_times.columns:
                    # Convert timedelta objects to total seconds
                    qual_times[col] = pd.to_timedelta(qual_times[col], errors="coerce").dt.total_seconds()

            qual_times["QualifyingTime (s)"] = qual_times[["Q1", "Q2", "Q3"]].min(axis=1)

            race_results = race.results[["Abbreviation", "Position"]].rename(
                columns={"Abbreviation": "Driver", "Position": "RacePosition"}
            )

            merged = pd.merge(qual_times, race_results, on="Driver", how="inner")
            merged["Season"] = yr
            data_years.append(merged)
        except Exception as e:
            print(f"⚠️ Error loading {yr} Japanese GP (Round {rnd}): {e}")

    # ----------------------------------------------------
    # 2. 2025 Data Loading for Training/Prediction Base
    # ----------------------------------------------------
    # Load 2025 races (e.g., Rounds 1 and 2) to establish 2025 form
    print("⏳ Loading 2025 races (Rounds 1 & 2) for current form...")
    for yr, rnd in [(2025, 1), (2025, 2)]:
        try:
            qual = fastf1.get_session(yr, rnd, "Q")
            race = fastf1.get_session(yr, rnd, "R")
            qual.load(telemetry=False, weather=False); race.load(telemetry=False, weather=False)

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
            print(f"⚠️ Error loading {yr} race {rnd}: {e}")

    if not data_years:
        print("❌ No historical or current data found. Cannot run prediction.")
        return pd.DataFrame()

    historical = pd.concat(data_years, ignore_index=True)

    # ----------------------------------------------------
    # 3. Static Features & Driver/Team Mapping
    # ----------------------------------------------------
    # These values are based on the input code's definition
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

    # ----------------------------------------------------
    # 4. Weather Data (Suzuka, Japan)
    # ----------------------------------------------------
    load_dotenv()
    API_KEY = os.getenv("WeatherAPI", "")
    rain_probability, temperature = 0, 22 # defaults

    if API_KEY:
        try:
            # Coordinates for Suzuka Circuit: 34.8431, 136.5419
            params = {"key": API_KEY, "q": "34.8431,136.5419", "days": 1, "aqi": "no", "alerts": "no"}
            resp = requests.get("http://api.weatherapi.com/v1/forecast.json", params=params)
            weather = resp.json()
            forecast = weather.get("forecast", {}).get("forecastday", [])
            if forecast:
                # Taking the first hour's forecast as a proxy
                hour = forecast[0].get("hour", [])[0]
                rain_probability = hour.get("chance_of_rain", 0) / 100
                temperature = hour.get("temp_c", 22)
            print(f"☁️ Current Suzuka weather: Rain chance: {rain_probability*100:.0f}%, Temp: {temperature}°C")
        except Exception as e:
            print(f"⚠️ Weather API error (Suzuka): {e}")

    historical["RainProbability"] = rain_probability
    historical["Temperature"] = temperature

    # ----------------------------------------------------
    # 5. Model Training (Gradient Boosting Regressor)
    # ----------------------------------------------------
    print("🧠 Training the Gradient Boosting Regressor model...")
    features = ["QualifyingTime (s)", "TeamPerformanceScore", "CleanAirRacePace (s)", "RainProbability", "Temperature"]
    X = historical[features].copy()
    X["QualifyingTime (s)"] *= QUALI_WEIGHT  # Apply weight

    # Drop rows with NaN target variable (RacePosition)
    historical_clean = historical.dropna(subset=["RacePosition"]).copy()
    y = historical_clean["RacePosition"]
    X = historical_clean[features].copy()
    X["QualifyingTime (s)"] *= QUALI_WEIGHT

    # Impute remaining missing feature values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    print(f"📊 MAE on validation set: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")

    # ----------------------------------------------------
    # 6. Prediction for 2025 Japanese GP
    # ----------------------------------------------------
    # Use the 2025 race data as the prediction input (current form)
    qual_2025 = historical[historical["Season"] == 2025].drop_duplicates(subset=["Driver"], keep='last').copy()

    # Apply weather and static features to the prediction set
    qual_2025["RainProbability"] = rain_probability
    qual_2025["Temperature"] = temperature
    qual_2025["Team"] = qual_2025["Driver"].map(driver_team)
    qual_2025["TeamPerformanceScore"] = qual_2025["Team"].map(team_perf)
    qual_2025["CleanAirRacePace (s)"] = qual_2025["Driver"].map(clean_air_race_pace)

    # Impute missing qualifying data with median from the 2025 prediction set itself
    median_qual_time = qual_2025["QualifyingTime (s)"].median()
    qual_2025["QualifyingTime (s)"].replace([np.inf, -np.inf], np.nan, inplace=True)
    qual_2025["QualifyingTime (s)"].fillna(median_qual_time, inplace=True)

    # Impute missing static data with median from the historical training set
    for feature in ["TeamPerformanceScore", "CleanAirRacePace (s)"]:
        qual_2025.loc[:, feature].fillna(historical[feature].median(), inplace=True)

    X_pred = qual_2025[features].copy()
    X_pred["QualifyingTime (s)"] *= QUALI_WEIGHT
    X_pred_imputed = imputer.transform(X_pred)

    # Predict and sort
    qual_2025["PredictedPosition"] = model.predict(X_pred_imputed)
    qual_2025 = qual_2025.sort_values("PredictedPosition").reset_index(drop=True)

    # ----------------------------------------------------
    # 7. Output Results and Feature Importance
    # ----------------------------------------------------
    print("\n🏁 Predicted 2025 Japanese GP Race Results 🏁")
    # Displaying the top 10 positions
    print(qual_2025[["Driver", "PredictedPosition"]].head(10).to_string(index=False))

    print("\n🏆 Predicted Podium 🏆")
    print(f"🥇 P1: {qual_2025.iloc[0]['Driver']} (Predicted: {qual_2025.iloc[0]['PredictedPosition']:.2f})")
    print(f"🥈 P2: {qual_2025.iloc[1]['Driver']} (Predicted: {qual_2025.iloc[1]['PredictedPosition']:.2f})")
    print(f"🥉 P3: {qual_2025.iloc[2]['Driver']} (Predicted: {qual_2025.iloc[2]['PredictedPosition']:.2f})")

    plt.figure(figsize=(8, 5))
    plt.barh(features, model.feature_importances_, color="teal")
    plt.title("Feature Importance (Japanese GP Race Prediction)")
    plt.tight_layout()
    plt.show()

    return qual_2025[["Driver", "PredictedPosition"]]

# Example of how to call the function:
# if __name__ == "__main__":
#     run_prediction()