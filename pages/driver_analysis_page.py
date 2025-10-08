import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as np

def run_driver_analysis_dashboard(load_data_func, data_file):
    """
    Displays the interactive Driver Performance Analysis dashboard.
    """
    st.title("🏎️ Driver Performance Analysis")
    st.caption("Explore how F1 drivers perform over seasons in terms of points, consistency, and pace.")
    st.markdown("---")

    # --- Load Data ---
    try:
        df = load_data_func(data_file)
    except Exception as e:
        st.error(f"🚨 Failed to load data for analysis: {e}")
        return

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filters")

    available_seasons = sorted(df["season"].unique())
    selected_season = st.sidebar.selectbox("Select Season:", options=["All Seasons"] + available_seasons)

    # Filter by season if not "All"
    if selected_season != "All Seasons":
        df = df[df["season"] == selected_season]

    # --- Data Processing ---
    driver_stats = df.groupby("Abbreviation").agg(
        Total_Points=("Points", "sum"),
        Races_Driven=("RacesDriven", "max"),
        Avg_Grid_Position=("GridPosition", "mean"),
        Avg_Finishing_Position=("Position", "mean"),
    ).reset_index()

    driver_stats.rename(columns={"Abbreviation": "Driver"}, inplace=True)
    driver_stats["Points_Per_Race"] = driver_stats["Total_Points"] / driver_stats["Races_Driven"]
    driver_stats.sort_values(by="Total_Points", ascending=False, inplace=True)

    # --- Sidebar Filters Continued ---
    driver_list = ["All Drivers"] + sorted(driver_stats["Driver"].unique())
    selected_driver = st.sidebar.selectbox("Select Driver for Detail:", driver_list)

    n_drivers = st.sidebar.slider("Top N Drivers for Charts:", 5, 20, 10)
    top_drivers = driver_stats.head(n_drivers)

    # --- Visualization 1: Total Points Bar Chart ---
    st.subheader("💰 Top Drivers by Total Points")
    st.markdown("Displays which drivers accumulated the most points during the selected seasons.")

    if not top_drivers.empty:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.barplot(x="Driver", y="Total_Points", data=top_drivers, palette="coolwarm")

        ax1.set_title(f"Top {n_drivers} Drivers by Total Points", fontsize=14, weight="bold")
        ax1.set_xlabel("Driver", fontsize=12)
        ax1.set_ylabel("Total Points", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)

        # Add value labels
        for index, row in top_drivers.iterrows():
            ax1.text(index, row["Total_Points"] + 2, f"{row['Total_Points']:.0f}", ha="center", fontsize=9)

        st.pyplot(fig1)
        plt.close(fig1)
    else:
        st.info("⚠️ Not enough data to show top drivers chart.")

    # --- Visualization 2: Avg Grid vs. Finishing Position ---
    st.subheader("🎯 Qualifying vs. Race Performance")
    st.markdown("Drivers below the diagonal line perform **better in races** compared to their qualifying positions.")

    if not driver_stats.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 8))
        ax2.scatter(
            driver_stats["Avg_Grid_Position"],
            driver_stats["Avg_Finishing_Position"],
            alpha=0.6,
            label="All Drivers",
            color="gray",
        )

        if selected_driver != "All Drivers":
            highlight_driver = driver_stats[driver_stats["Driver"] == selected_driver]
            if not highlight_driver.empty:
                ax2.scatter(
                    highlight_driver["Avg_Grid_Position"],
                    highlight_driver["Avg_Finishing_Position"],
                    s=200,
                    color="red",
                    label=selected_driver,
                    zorder=5,
                )
                ax2.annotate(
                    selected_driver,
                    (
                        highlight_driver["Avg_Grid_Position"].iloc[0] + 0.3,
                        highlight_driver["Avg_Finishing_Position"].iloc[0],
                    ),
                    fontsize=10,
                    color="black",
                    weight="bold",
                )

        # Equal performance line
        min_val = min(driver_stats["Avg_Grid_Position"].min(), driver_stats["Avg_Finishing_Position"].min())
        max_val = max(driver_stats["Avg_Grid_Position"].max(), driver_stats["Avg_Finishing_Position"].max())
        ax2.plot([min_val, max_val], [min_val, max_val], "b--", alpha=0.6, label="Equal Pace Line")

        ax2.set_xlabel("Average Grid Position (Lower = Better)", fontsize=12)
        ax2.set_ylabel("Average Finishing Position (Lower = Better)", fontsize=12)
        ax2.set_title("Qualifying vs. Race Day Comparison", fontsize=14, weight="bold")
        ax2.invert_yaxis()
        ax2.invert_xaxis()
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig2)
        plt.close(fig2)

        st.caption(
            "*Drivers below the blue dashed line show improved race performance compared to their qualifying positions.*"
        )
    else:
        st.warning("No qualifying or finishing data available.")

    # --- Metrics Summary Section ---
    st.markdown("---")
    st.subheader("🏁 Performance Summary")

    best_driver = driver_stats.iloc[0]["Driver"]
    best_points = driver_stats.iloc[0]["Total_Points"]
    avg_position = driver_stats["Avg_Finishing_Position"].mean()
    best_efficiency = driver_stats.loc[driver_stats["Points_Per_Race"].idxmax()]["Driver"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Top Scorer", best_driver, f"{best_points:.0f} pts")
    col2.metric("Best Efficiency", best_efficiency, "Most points per race")
    col3.metric("Avg Finishing Pos (All Drivers)", f"{avg_position:.2f}")

    # --- Detailed Stats for Selected Driver ---
    if selected_driver != "All Drivers":
        st.markdown("---")
        st.subheader(f"📋 Detailed Statistics for {selected_driver}")

        detail_df = driver_stats[driver_stats["Driver"] == selected_driver].transpose()
        detail_df.columns = [selected_driver]

        # Format numeric columns
        for col in ["Avg_Grid_Position", "Avg_Finishing_Position", "Points_Per_Race"]:
            if col in detail_df.index:
                detail_df.loc[col] = detail_df.loc[col].astype(float).map("{:.2f}".format)

        st.table(detail_df)

    st.success("✅ Driver Performance Dashboard Loaded Successfully!")
