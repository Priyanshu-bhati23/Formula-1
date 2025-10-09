import streamlit as st
import importlib.util # Use importlib.util for safer dynamic imports
import os
import matplotlib.pyplot as plt
import pandas as pd 
import fastf1

# --- Attempt robust dynamic import for analysis pages ---
# Import analysis pages directly from 'pages' subdirectory.
try:
    # Attempt to import all pages as they were in the user's snippet
    import pages.team_analysis_page as team_analysis_page 
    import pages.driver_vs_teammate as driver_vs_teammate
    import pages.driver_analysis_page as driver_analysis_page
    import pages.season_analysis_page as season_analysis_page
    analysis_pages_loaded = True
except ImportError:
    st.error("🚨 Module Error: Could not import analysis pages from the 'pages' directory. Ensure the folder exists and files are present.")
    team_analysis_page = None
    driver_vs_teammate = None
    driver_analysis_page = None
    season_analysis_page = None
    analysis_pages_loaded = False


# Enable FastF1 cache for Streamlit Cloud
cache_dir = "/tmp/fastf1_cache" 
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

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

# --- HELPER FUNCTION FOR DATA LOADING (REQUIRED for analysis pages) ---
@st.cache_data
def load_data(file_path):
    """Loads and caches the feature-engineered DataFrame. Returns empty DataFrame on error."""
    try:
        # Check for existence of the file relative to the script directory
        if not os.path.exists(file_path):
            st.error(f"Data file not found at: {file_path}. Please place 'f1_feature_engineered.csv' in the root directory.")
            return pd.DataFrame()
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data from CSV: {e}")
        return pd.DataFrame()

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
        # List files in the subdirectory
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
            # Safer dynamic import using spec_from_file_location
            spec = importlib.util.spec_from_file_location(module_path, expected_file)
            race_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(race_module)

            if hasattr(race_module, "run_prediction"):
                with st.spinner(f"Predicting {race_name.capitalize()} GP results..."):
                    try:
                        # Attempt to call with cache_path first
                        result = race_module.run_prediction(cache_path)
                    except TypeError:
                        # Fallback call without cache_path argument if the function signature is old
                        result = race_module.run_prediction()

                st.success("✅ Prediction completed successfully!")

                # --- RESULT DISPLAY ---
                # Handle single DataFrame result OR the new structured tuple (DataFrame, model, features)
                
                # Check for structured tuple output
                if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], pd.DataFrame):
                    qual_2025, *other_results = result # Unpack the DataFrame and the rest of the tuple
                    display_result = qual_2025
                    
                # Check for single DataFrame output
                elif isinstance(result, pd.DataFrame):
                    display_result = result
                else:
                    display_result = pd.DataFrame()


                if not display_result.empty:
                    st.subheader("🏁 Predicted Race Results")
                    st.dataframe(display_result, use_container_width=True)

                    st.markdown("### 🏆 Predicted Podium")
                    
                    # Assume "PredictedPosition" is always the sorting key
                    if "PredictedPosition" in display_result.columns:
                        podium_df = display_result.sort_values("PredictedPosition").head(3)
                    else:
                        # Fallback if the column name is missing (e.g., if only 'Position' exists)
                        podium_df = display_result.head(3)

                    podium_size = len(podium_df)
                    if podium_size > 0:
                        cols = st.columns(podium_size)
                        medals = ["🥇", "🥈", "🥉"]
                        for i, col in enumerate(cols):
                            with col:
                                st.metric(label=f"{medals[i]} {podium_df.iloc[i]['Driver']}",
                                            value=f"P{i+1}")
                                
                    st.subheader("📊 Model Feature Importance")
                    try:
                        # Try to capture the plot figure shown by the race script
                        if hasattr(plt, 'gcf') and plt.gcf().axes:
                             st.pyplot(plt.gcf())
                        else:
                             st.info("Feature importance plot not available or was not generated.")
                    except Exception:
                        st.info("Feature importance plot not available or was not generated.")

                else:
                    st.info("ℹ️ No prediction results returned or the result was empty. Check the race script.")
            
            else:
                st.warning(f"No `run_prediction()` function found in `{race_name}_gp.py`")

        except Exception as e:
            st.error(f"🚨 Error while running race prediction.")
            st.exception(e)
            st.info("Tip: This might happen if FastF1 cannot fetch or cache data for this race, or if the race script contains internal Python errors.")
            
    elif not available_races:
        pass 
    else:
        st.info("👈 Select a Grand Prix from the sidebar and click **Run Prediction** to start.")

# --- ANALYSIS PAGES DISPATCH ---
# Enhanced check to ensure modules loaded successfully
elif page == "Team Analysis":
    if team_analysis_page and analysis_pages_loaded:
        team_analysis_page.run_team_analysis_dashboard(load_data, DATA_FILE)
    else:
        st.error("Team Analysis page module failed to load. See error logs above.")
        
elif page == "Driver Analysis":
    if driver_analysis_page and analysis_pages_loaded:
        driver_analysis_page.run_driver_analysis_dashboard(load_data, DATA_FILE)
    else:
        st.error("Driver Analysis page module failed to load.")

elif page == "Driver vs Teammate":
    if driver_vs_teammate and analysis_pages_loaded:
        driver_vs_teammate.run_comparison_dashboard(load_data, DATA_FILE)
    else:
        st.error("Driver vs Teammate page module failed to load.")

elif page == "Season Analysis":
    if season_analysis_page and analysis_pages_loaded:
        season_analysis_page.run_season_analysis_dashboard(load_data, DATA_FILE)
    else:
        st.error("Season Analysis page module failed to load.")
