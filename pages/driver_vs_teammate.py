# driver_vs_teammate.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def run_comparison_dashboard(load_data_func, data_file):
    """
    Displays the Driver vs. Teammate comparison analysis.
    """
    st.title("🤝 Driver vs. Teammate Performance")
    st.markdown("---")
    st.markdown(
        "Compare the performance of F1 drivers against their most recent teammates. "
        "Positive values indicate the driver finishes better than their teammate on average."
    )

    # --- Load Data ---
    try:
        df = load_data_func(data_file)
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return

    # --- Normalize column names ---
    df.columns = df.columns.str.strip().str.lower()

    # --- Check required columns ---
    required_columns = ['broadcastname', 'teamname', 'fullname', 'position', 'season']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return

    # --- Sidebar: Season filter ---
    all_seasons = sorted(df['season'].unique())
    selected_season = st.sidebar.selectbox(
        "Select Season:",
        options=all_seasons,
        index=len(all_seasons)-1
    )
    df_filtered = df[df['season'] == selected_season]

    if df_filtered.empty:
        st.warning("No data available for this season.")
        return

    # --- Map drivers to their most common teammate ---
    def safe_mode(series):
        return series.mode().iloc[0] if not series.mode().empty else None

    teammate_mapping = (
        df_filtered.groupby(['broadcastname', 'teamname'])['fullname']
        .agg(safe_mode)
        .reset_index()
    )
    teammate_mapping.rename(columns={'fullname': 'teammate', 'broadcastname': 'driver'}, inplace=True)
    teammate_mapping.drop(columns='teamname', inplace=True)

    # Merge teammate info back
    df_merged = pd.merge(df_filtered, teammate_mapping, on='driver', how='left', suffixes=('', '_teammate'))

    # --- Calculate average positions ---
    avg_position = (
        df_merged.groupby(['driver', 'teammate'])['position']
        .mean()
        .reset_index()
        .rename(columns={'position': 'avg_position'})
    )

    teammate_avg_position = (
        df_merged.groupby('teammate')['position']
        .mean()
        .reset_index()
        .rename(columns={'position': 'teammate_avg_position'})
    )

    comparison_df = pd.merge(avg_position, teammate_avg_position, on='teammate', how='left')
    comparison_df['position_advantage'] = comparison_df['teammate_avg_position'] - comparison_df['avg_position']
    comparison_df.dropna(subset=['avg_position', 'teammate_avg_position'], inplace=True)
    comparison_df.sort_values(by='position_advantage', ascending=False, inplace=True)

    # --- Visualization ---
    st.subheader("📊 Driver Performance Advantage Over Teammate")
    if not comparison_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if adv > 0 else 'red' for adv in comparison_df['position_advantage']]
        ax.barh(comparison_df['driver'], comparison_df['position_advantage'], color=colors)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel("Average Position Advantage (Higher = Better)")
        ax.set_ylabel("Driver")
        ax.set_title("Driver's Average Finishing Position Advantage Over Teammate")
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "*Positive values indicate the driver consistently finishes ahead of their teammate.*"
        )
    else:
        st.info("Insufficient data to perform Driver vs Teammate comparison.")

    # --- Summary Table ---
    st.subheader("🏁 Summary Table")
    summary_table = comparison_df[['driver', 'teammate', 'avg_position', 'teammate_avg_position', 'position_advantage']].copy()
    summary_table.rename(columns={
        'driver': 'Driver',
        'teammate': 'Teammate',
        'avg_position': 'Driver Avg Position',
        'teammate_avg_position': 'Teammate Avg Position',
        'position_advantage': 'Position Advantage'
    }, inplace=True)
    summary_table['Position Advantage'] = summary_table['Position Advantage'].round(2)
    st.table(summary_table)

    st.success("✅ Driver vs Teammate Dashboard Loaded Successfully!")
