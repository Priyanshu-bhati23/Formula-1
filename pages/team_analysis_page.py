# team_analysis_page.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def run_team_analysis_dashboard(load_data_func, data_file):
    """
    Multi-page Streamlit Team Analysis Dashboard.

    Args:
        load_data_func: Function to load dataset (e.g., load_data)
        data_file: Path to the CSV dataset
    """
    st.title("🏎️ Formula 1 Team Performance Dashboard")
    st.caption("Explore F1 team stats, trends, and performance consistency from 2018–2024")
    st.markdown("---")

    # --- LOAD DATA ---
    try:
        df = load_data_func(data_file)
    except FileNotFoundError as e:
        st.error(f"🚨 {e}. Please ensure the dataset is in your project root.")
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filters")
    selected_season = st.sidebar.selectbox(
        "Select Season:",
        options=sorted(df["season"].unique()),
        index=len(df["season"].unique()) - 1
    )

    available_teams = sorted(df["TeamName"].unique())
    selected_teams = st.sidebar.multiselect(
        "Select Teams to Compare:",
        options=available_teams,
        default=available_teams[:5]
    )

    df_season = df[df["season"] == selected_season]
    df_filtered = df_season[df_season["TeamName"].isin(selected_teams)]

    # ===========================
    # 1️⃣ Team Wins Distribution
    # ===========================
    st.subheader("🏆 Team Wins Distribution")
    st.markdown("**Visualize how wins are distributed among top teams in the selected season.**")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        team_wins = df_filtered.groupby("TeamName")["Winner"].sum().sort_values(ascending=False)
        if not team_wins.empty:
            fig1, ax1 = plt.subplots(figsize=(7, 7))
            ax1.pie(
                team_wins,
                labels=team_wins.index,
                autopct='%1.1f%%',
                startangle=120,
                colors=sns.color_palette("Set2", len(team_wins))
            )
            ax1.set_title(f"🏆 Team Wins in {selected_season}", fontsize=14, weight='bold')
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.info("No win data available for this filter.")
    
    with col2:
        st.metric("Most Winning Team", team_wins.idxmax() if not team_wins.empty else "N/A")
        st.metric("Total Wins Counted", int(team_wins.sum()) if not team_wins.empty else 0)

    st.markdown("---")

    # ===========================
    # 2️⃣ Average Points per Team
    # ===========================
    st.subheader("⭐ Average Points per Race (Selected Season)")
    st.markdown("**Compare the average points per race across selected teams.**")

    team_points = df_filtered.groupby("TeamName")["Points"].mean().sort_values(ascending=False)

    if not team_points.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.barplot(x=team_points.values, y=team_points.index, palette="Blues_r", ax=ax2)

        ax2.set_xlabel("Avg Points per Race", fontsize=12)
        ax2.set_ylabel("Team")
        ax2.grid(axis='x', linestyle='--', alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("No average points data available for the current selection.")

    st.markdown("---")

    # ===========================
    # 3️⃣ Position Distribution
    # ===========================
    st.subheader("⚙️ Finishing Position Distribution")
    st.markdown("**Check consistency and spread of finishing positions among top teams.**")

    if not df_filtered.empty:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.boxplot(
                data=df_filtered,
                x='TeamName',
                y='Position',
                palette="crest",
                ax=ax3
            )
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
    st.markdown("**See how numerical features correlate with each other for the selected teams and season.**")

    numeric_cols = df_filtered.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) > 1:
        corr = df_filtered[numeric_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5, ax=ax4)
        ax4.set_title("Feature Correlation Heatmap", fontsize=13, weight='bold')
        st.pyplot(fig4)
        plt.close(fig4)
    else:
        st.info("No numerical data available for correlation analysis.")

    st.success("✅ Dashboard Loaded Successfully!")
