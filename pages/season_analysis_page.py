import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def run_season_analysis_dashboard(load_data_func, data_file):
    """
    Displays the Season Trends Analysis (Interactive Version)
    """
    st.title("📈 Season Trends and Dominance Dashboard")
    st.markdown("---")
    st.markdown("Explore Formula 1 team performance trends, championship dominance, and team competitiveness interactively.")

    try:
        # Load data
        df = load_data_func(data_file)
    except Exception as e:
        st.error(f"❌ Failed to load data for analysis: {e}")
        return

    # --- Normalize column names ---
    df.columns = df.columns.str.strip().str.lower()

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filters")

    # Available seasons and teams
    all_seasons = sorted(df['season'].unique())
    all_teams = sorted(df['teamname'].unique())

    # Season range selector
    selected_range = st.sidebar.slider(
        "Select Season Range:",
        min_value=int(min(all_seasons)),
        max_value=int(max(all_seasons)),
        value=(int(min(all_seasons)), int(max(all_seasons))),
        step=1
    )

    # Team selection
    selected_teams = st.sidebar.multiselect(
        "Select Teams to Display:",
        options=all_teams,
        default=all_teams  # show all by default
    )

    # Apply filters
    df_filtered = df[(df['season'] >= selected_range[0]) & (df['season'] <= selected_range[1])]
    if selected_teams:
        df_filtered = df_filtered[df_filtered['teamname'].isin(selected_teams)]

    # --- Data Aggregation ---
    team_points_by_year = df_filtered.groupby(['season', 'teamname'])['points'].sum().reset_index()
    total_points_by_year = df_filtered.groupby('season')['points'].sum().reset_index()
    total_points_by_year.rename(columns={'points': 'total_season_points'}, inplace=True)

    merged_df = pd.merge(team_points_by_year, total_points_by_year, on='season')
    merged_df['dominance_ratio'] = merged_df['points'] / merged_df['total_season_points']
    dominant_teams = merged_df.loc[merged_df.groupby('season')['dominance_ratio'].idxmax()]

    # --- Visualization 1: Stacked Bar Chart ---
    st.subheader("📊 Total Points per Team by Season")
    st.markdown("Breakdown of total points scored by selected teams each season.")

    if not team_points_by_year.empty:
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        pivot_df = team_points_by_year.pivot(index='season', columns='teamname', values='points').fillna(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pivot_df.plot(kind='bar', stacked=True, colormap='Spectral', ax=ax1)

        ax1.set_title("Total Points per Team by Season")
        ax1.set_xlabel("Season")
        ax1.set_ylabel("Total Points Scored")
        ax1.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        st.pyplot(fig1)
        plt.close(fig1)
    else:
        st.warning("No data available for the selected filters.")

    # --- Visualization 2: Dominance Ratio ---
    st.subheader("🏁 Dominance Trends (Constructor Championship)")
    st.markdown("Shows how dominant the leading team was each season based on total points ratio.")

    if not dominant_teams.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(dominant_teams['season'], dominant_teams['dominance_ratio'] * 100,
                 marker='o', linestyle='-', color='darkred')

        for i in dominant_teams.index:
            ax2.annotate(dominant_teams.loc[i, 'teamname'],
                         (dominant_teams.loc[i, 'season'],
                          dominant_teams.loc[i, 'dominance_ratio'] * 100 + 1),
                         fontsize=9, ha='center')

        ax2.set_title("Dominant Team's Championship Ratio by Season")
        ax2.set_xlabel("Season")
        ax2.set_ylabel("Dominance Ratio (%)")
        ax2.set_ylim(0, 100)
        ax2.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig2)
        plt.close(fig2)

    # --- Visualization 3: Top 3 Teams Year-over-Year Trends ---
    st.subheader("📈 Top 3 Teams Performance Trends")
    st.markdown("Tracks how the top 3 teams’ points evolved each season within your selected range.")

    if not team_points_by_year.empty:
        # Find top 3 based on total points (within selected range)
        top3_teams = (
            team_points_by_year.groupby('teamname')['points']
            .sum()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )

        top3_df = team_points_by_year[team_points_by_year['teamname'].isin(top3_teams)]

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=top3_df, x='season', y='points', hue='teamname', marker='o', linewidth=2.5, ax=ax3)

        ax3.set_title("Top 3 Teams – Points Progression by Season")
        ax3.set_xlabel("Season")
        ax3.set_ylabel("Total Points")
        ax3.grid(True, linestyle='--', alpha=0.4)
        ax3.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig3)
        plt.close(fig3)

    # --- Summary Table ---
    st.subheader("🏆 Season Champions & Dominance Summary")
    summary_table = dominant_teams[['season', 'teamname', 'dominance_ratio']].copy()
    summary_table.rename(columns={
        'teamname': 'Dominant Team',
        'dominance_ratio': 'Dominance Ratio',
        'season': 'Season'
    }, inplace=True)
    summary_table['Dominance Ratio'] = (summary_table['Dominance Ratio'] * 100).map('{:.2f}%'.format)

    st.table(summary_table.set_index('Season'))
