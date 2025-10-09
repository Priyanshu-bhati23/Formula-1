import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

def run_comparison_dashboard(load_data_func, data_file):
    """
    Displays the Driver vs. Teammate comparison analysis.
    """
    st.title("🤝 Driver vs. Teammate Performance")
    st.markdown("---")
    st.markdown("This analysis compares the performance of current F1 drivers against their most recent teammates, providing context for their race results.")

    try:
        df = load_data_func(data_file)
    except Exception as e:
        st.error(f"Failed to load data for analysis: {e}")
        return

    # --- Data Filtering and Processing ---
    # Find the most common teammate for each driver
    teammate_mapping = df.groupby(['Driver', 'TeamName'])['FullName'].agg(lambda x: x.mode()[0] if not x.empty else None).reset_index()
    teammate_mapping.rename(columns={'FullName': 'Teammate'}, inplace=True)
    teammate_mapping.drop(columns='TeamName', inplace=True)

    # Merge teammate info back to the main DataFrame
    df_merged = pd.merge(df, teammate_mapping, on='Driver', how='left', suffixes=('', '_Teammate'))
    
    # Calculate the average position for each driver/teammate pair
    avg_position = df_merged.groupby(['driver', 'Teammate'])['Position'].mean().reset_index()
    avg_position.rename(columns={'Position': 'Avg_Position'}, inplace=True)
    
    # Calculate the average position for the teammate
    teammate_avg_position = df_merged.groupby('Teammate')['Position'].mean().reset_index()
    teammate_avg_position.rename(columns={'Teammate': 'Teammate', 'Position': 'Teammate_Avg_Position'}, inplace=True)

    # Combine data
    comparison_df = pd.merge(avg_position, teammate_avg_position, on='Teammate', how='left')
    comparison_df['Position_Advantage'] = comparison_df['Teammate_Avg_Position'] - comparison_df['Avg_Position']
    
    # Sort and filter for visualization
    comparison_df.sort_values(by='Position_Advantage', ascending=False, inplace=True)
    
    # Drop rows where data is incomplete (often due to filtering/joins)
    comparison_df.dropna(subset=['Avg_Position', 'Teammate_Avg_Position'], inplace=True)

    # --- Visualization ---
    st.subheader("📊 Driver Performance Advantage Over Teammate")

    if not comparison_df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color based on advantage
        colors = ['green' if adv > 0 else 'red' for adv in comparison_df['Position_Advantage']]

        ax.barh(comparison_df['Driver'], comparison_df['Position_Advantage'], color=colors)
        
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel("Average Position Advantage (Higher = Better)")
        ax.set_ylabel("Driver")
        ax.set_title("Driver's Average Finishing Position Advantage Over Teammate")
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        st.pyplot(fig)
        plt.close(fig)
        
        st.caption("*Note: A positive value indicates the driver consistently finishes in a better position than their teammate's average position.*")

    else:
        st.info("Insufficient data to perform Driver vs Teammate comparison.")