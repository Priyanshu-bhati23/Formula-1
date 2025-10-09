# team_analysis_page.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def run_team_analysis_dashboard(load_data_func, data_file):
    """
    Displays the Team Performance Dashboard.
    """
    st.title("🏎️ Team Performance Dashboard")
    st.markdown("---")
    st.markdown("Explore F1 team stats, trends, and performance consistency from 2018–2024.")

    # --- Load Data ---
    try:
        df = load_data_func(data_file)
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filters")
    all_seasons = sorted(df['season'].unique())
    selected_season = st.sidebar.selectbox(
        "Select Season:",
        options=all_seasons,
        index=len(all_seasons)-1
    )

    all_teams = sorted(df['teamname'].unique())
    selected_teams = st.sidebar.multiselect(
        "Select Teams to Compare:",
        options=all_teams,
        default=all_teams[:5]
    )

    # Filter dataframe
    df_filtered = df[(df['season'] == selected_season) & (df['teamname'].isin(selected_teams))]

    if df_filtered.empty:
        st.info("No data available for this selection.")
        return

    # ===========================
    # 1️⃣ Team Wins Distribution
    # ===========================
    st.subheader("🏆 Team Wins Distribution")
    st.markdown("Visualize how wins are distributed among selected teams.")

    team_wins = df_filtered.groupby("teamname")["winner"].sum().sort_values(ascending=False)
    if not team_wins.empty:
        fig1, ax1 = plt.subplots(figsize=(7, 7))
        ax1.pie(team_wins, labels=team_wins.index, autopct='%1.1f%%', startangle=120,
                colors=sns.color_palette("Set2", len(team_wins)))
        ax1.set_title(f"Team Wins in {selected_season}", fontsize=14, weight='bold')
        st.pyplot(fig1)
        plt.close(fig1)
    else:
        st.info("No win data available for this filter.")

    st.markdown("---")

    # ===========================
    # 2️⃣ Average Points per Team
    # ===========================
    st.subheader("⭐ Average Points per Race")
    st.markdown("Compare the average points per race across selected teams.")

    team_points = df_filtered.groupby("teamname")["points"].mean().sort_values(ascending=False)
    if not team_points.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.barplot(x=team_points.values, y=team_points.index, palette="Blues_r", ax=ax2)
        ax2.set_xlabel("Avg Points per Race")
        ax2.set_ylabel("Team")
        ax2.grid(axis='x', linestyle='--', alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("No average points data available for this filter.")

    st.markdown("---")

    # ===========================
    # 3️⃣ Finishing Position Distribution
    # ===========================
    st.subheader("⚙️ Finishing Position Distribution")
    st.markdown("Check consistency and spread of finishing positions among selected teams.")

    if not df_filtered.empty:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.boxplot(data=df_filtered, x='teamname', y='position', palette="crest", ax=ax3)
        ax3.set_title(f"Finishing Positions ({selected_season})", fontsize=14, weight='bold')
        ax3.set_xlabel("Team")
        ax3.set_ylabel("Finishing Position (Lower = Better)")
        plt.xticks(rotation=45)
        ax3.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.info("No data available to generate the box plot.")

    st.markdown("---")

    # ===========================
    # 4️⃣ Correlation Heatmap
    # ===========================
    st.subheader("📊 Feature Correlation (Performance Indicators)")
    st.markdown("See how numerical features correlate with each other for selected teams.")

    numeric_cols = df_filtered.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) > 1:
        corr = df_filtered[numeric_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5, ax=ax4)
        ax4.set_title("Feature Correlation Heatmap", fontsize=13, weight='bold')
        st.pyplot(fig4)
        plt.close(fig4)
    else:
        st.info("Not enough numerical data available for correlation analysis.")

    st.success("✅ Team Analysis Dashboard Loaded Successfully!")
