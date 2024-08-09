import pandas as pd
import numpy as np
import streamlit as st

# Step 1: Apply Projections to Teams
def apply_projections_to_teams(draft_results_df, projections_df):
    # Prepare an empty DataFrame to store team results
    team_results = pd.DataFrame(index=draft_results_df['Team'].unique())

    # Iterate over each simulation column
    for sim in projections_df.columns:
        # Sum projections for each player in the team
        team_results[f'{sim}'] = draft_results_df.apply(
            lambda row: projections_df.loc[row.filter(regex='^Player_\d+_Name$')].sum(), axis=1)
    
    return team_results.reset_index()

# Streamlit UI
st.title('Fantasy Football Projection Application')

# File upload for draft results
uploaded_draft_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])

# File upload for simulated projections
uploaded_projections_file = st.file_uploader("Upload your simulated projections CSV file", type=["csv"])

if uploaded_draft_file and uploaded_projections_file:
    # Load the draft results and simulated projections
    draft_results_df = pd.read_csv(uploaded_draft_file)
    projections_df = pd.read_csv(uploaded_projections_file, index_col=0)

    st.write("Draft Results Data Preview:")
    st.dataframe(draft_results_df.head())

    st.write("Simulated Projections Data Preview:")
    st.dataframe(projections_df.head())

    # Apply projections to teams
    team_results = apply_projections_to_teams(draft_results_df, projections_df)

    # Display the results
    st.write("Team Results:")
    st.dataframe(team_results)

    # Download link for the results
    csv = team_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Team Results",
        data=csv,
        file_name='team_results.csv',
        mime='text/csv',
    )
