import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# --- CONFIG ---
QUALI_WEIGHT = 2.0

# Enable cache for Bahrain
fastf1.Cache.enable_cache("cache_bahrain")

# ----------------------------
# Collect race and qualifying data for past years
data_years = []

for yr, rnd in [(2022, 1), (2023, 1), (2024, 1)]:
    try:
        qual = fastf1.get_session(yr, rnd, "Q")
        race = fastf1.get_session(yr, rnd, "R")
        qual.load(); race.load()

        qual_times = (
            qual.results[["Abbreviation", "Q3", "Q2", "Q1"]]
            .copy()
            .rename(columns={"Abbreviation": "Driver"})
        )

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
        print(f"Error loading {yr} Bahrain GP: {e}")

historical = pd.concat(data_years, ignore_index=True).copy()

# ----------------------------
# Add static features
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

historical.loc[:, "RainProbability"] = 0
historical.loc[:, "Temperature"] = 22

# ----------------------------
# Weather data for Bahrain GP (Sakhir)
load_dotenv()
API_KEY = os.getenv("WeatherAPI", "0941fda9fd2b4435b26122000250610")
params = {
    "key": API_KEY,
    "q": "26.0325,50.5106",  # Bahrain International Circuit coords
    "days": 1, "aqi": "no", "alerts": "no"
}
try:
    resp = requests.get("http://api.weatherapi.com/v1/forecast.json", params=params)
    weather = resp.json()
    forecast = weather.get("forecast", {}).get("forecastday", [])
    rain_probability = 0; temperature = 22
    if forecast:
        hour = forecast[0].get("hour", [])[0]
        rain_probability = hour.get("chance_of_rain", 0) / 100
        temperature = hour.get("temp_c", 22)
except Exception as e:
    print(f"Weather API error: {e}")
    rain_probability = 0
    temperature = 22

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
print(f"MAE on validation: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")

# ----------------------------
# Predict Bahrain GP (2025)
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
for feature in ["TeamPerformanceScore", "CleanAirRacePace (s)"]:
    qual_2025[feature].fillna(historical[feature].median(), inplace=True)

qual_2025["RainProbability"] = rain_probability
qual_2025["Temperature"] = temperature

X_pred = qual_2025[features].copy()
X_pred["QualifyingTime (s)"] *= QUALI_WEIGHT
X_pred = imputer.transform(X_pred)
qual_2025["PredictedPosition"] = model.predict(X_pred)
qual_2025 = qual_2025.sort_values("PredictedPosition")

print("\n🏁 Predicted Bahrain GP 2025 Results 🏁")
print(qual_2025[["Driver", "PredictedPosition"]])

print("\n🏆 Predicted Podium 🏆")
print(f"🥇 P1: {qual_2025.iloc[0]['Driver']}")
print(f"🥈 P2: {qual_2025.iloc[1]['Driver']}")
print(f"🥉 P3: {qual_2025.iloc[2]['Driver']}")

plt.figure(figsize=(8, 5))
plt.barh(features, model.feature_importances_, color="salmon")
plt.title("Feature Importance (Bahrain GP Race Prediction)")
plt.tight_layout()
plt.show()
