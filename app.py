import streamlit as st
import importlib
import os
import matplotlib.pyplot as plt
import pandas as pd # <-- CORRECTED: Pandas import added
import pages.team_analysis_page as team_analysis_page 
import pages.driver_vs_teammate as driver_vs_teammate
import pages.driver_analysis_page as driver_analysis_page
import pages.season_analysis_page as season_analysis_page

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🏎️ Formula 1 Race Predictor",
    page_icon="🏁",
    layout="wide",
)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "Race Prediction", 
    "Team Analysis",
    "Driver Analysis",
    "Driver vs Teammate",
    "Season Analysis"
])
st.markdown("---")

# --- DATA FILE CONFIGURATION (Used by all analysis pages) ---
DATA_FILE = "f1_feature_engineered.csv" 

# --- HELPER FUNCTION FOR DATA LOADING ---
@st.cache_data
def load_data(file_path):
    """Loads and caches the feature-engineered DataFrame."""
    if not os.path.exists(file_path):
        # Raise an error that the calling function can catch
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    return pd.read_csv(file_path)

# -------------------------------
# Main Content Dispatch Logic
# -------------------------------
if page == "Race Prediction":
    st.title("🏎️ Formula 1 Race Predictor Dashboard")
    st.markdown("---")

    # -------------------------------
    # Race Selection (Inside Prediction Page)
    # -------------------------------
    st.sidebar.header("Select Grand Prix")
    
    try:
        available_races = [
            f.replace("_gp.py", "")
            for f in os.listdir("Predictions")
            if f.endswith("_gp.py")
        ]
    except FileNotFoundError:
        available_races = []

    if not available_races:
        st.error("No race scripts found in the `Predictions/` folder. Please create it and add scripts.")
        run_button = False
        race_name = None
    else:
        race_name = st.sidebar.selectbox("Choose a race:", available_races)
        run_button = st.sidebar.button("🚀 Run Prediction")

    # -------------------------------
    # Run Prediction Logic
    # -------------------------------
    if run_button and race_name:
        st.info(f"Running prediction for **{race_name.capitalize()} Grand Prix**...")

        module_path = f"Predictions.{race_name}_gp"
        expected_file = os.path.join("Predictions", f"{race_name}_gp.py")
        cache_path = os.path.join("Predictions", f"cache_{race_name}")

        if not os.path.exists(expected_file):
            st.error(f"❌ `{expected_file}` not found.")
            st.stop()

        os.makedirs(cache_path, exist_ok=True)

        try:
            race_module = importlib.import_module(module_path)

            if hasattr(race_module, "run_prediction"):
                with st.spinner(f"Predicting {race_name.capitalize()} GP results..."):
                    try:
                        result = race_module.run_prediction(cache_path)
                    except TypeError:
                        result = race_module.run_prediction()

                st.success("✅ Prediction completed successfully!")

                if result is not None and not result.empty:
                    st.subheader("🏁 Predicted Race Results")
                    st.dataframe(result, use_container_width=True)

                    st.markdown("### 🏆 Predicted Podium")
                    podium_size = min(3, len(result)) 
                    if podium_size > 0:
                        podium = result.sort_values("PredictedPosition").head(podium_size)
                        cols = st.columns(podium_size)
                        medals = ["🥇", "🥈", "🥉"]
                        for i, col in enumerate(cols):
                            with col:
                                st.metric(label=f"{medals[i]} {podium.iloc[i]['Driver']}",
                                          value=f"P{i+1}")
                    
                    st.subheader("📊 Model Feature Importance")
                    try:
                        if hasattr(race_module, "plt"):
                             st.pyplot(race_module.plt.gcf())
                        else:
                             st.info("Feature importance plot not available or was not captured.")
                    except Exception as e:
                        st.warning(f"Could not render plot: {e}")

                else:
                    st.info("ℹ️ No prediction results returned — check the race script.")
            else:
                st.warning(f"No `run_prediction()` function found in `{race_name}_gp.py`")

        except Exception as e:
            st.error(f"🚨 Error while running prediction: {e}")
            st.code(f"Details: {e}", language='python')
            
    elif not available_races:
        pass 
    else:
        st.info("👈 Select a Grand Prix from the sidebar and click **Run Prediction** to start.")

elif page == "Driver Analysis":
    driver_analysis_page.run_driver_analysis_dashboard(load_data, DATA_FILE) 

elif page == "Driver vs Teammate":
    driver_vs_teammate.run_comparison_dashboard(load_data, DATA_FILE)

elif page == "Season Analysis":
    season_analysis_page.run_season_analysis_dashboard(load_data, DATA_FILE)

elif page == "Team Analysis":
    team_analysis_page.run_team_analysis_dashboard(load_data, DATA_FILE)
