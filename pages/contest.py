import streamlit as st
import pandas as pd
import numpy as np

# Function to apply simulated projections to teams
def apply_projections_to_teams(draft_results_df, simulated_projections_df, num_simulations):
    teams = draft_results_df['Team'].unique()
    team_results = {team: [] for team in teams}

    for sim in range(num_simulations):
        for team in teams:
            team_players = draft_results_df[draft_results_df['Team'] == team]['Player'].values
            team_score = simulated_projections_df.loc[team_players, sim].sum()
            team_results[team].append(team_score)

    # Calculate average score for each team
    team_avg_scores = {team: np.mean(scores) for team, scores in team_results.items()}

    return pd.DataFrame.from_dict(team_avg_scores, orient='index', columns=['Average Score'])

# Streamlit UI
st.title('Team Projections Application')

# File upload for draft results
uploaded_draft_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])

# File upload for simulated projections
uploaded_projections_file = st.file_uploader("Upload your simulated projections CSV file", type=["csv"])

if uploaded_draft_file is not None and uploaded_projections_file is not None:
    # Load the draft results
    draft_results_df = pd.read_csv(uploaded_draft_file)
    st.write("Draft Results Data Preview:")
    st.dataframe(draft_results_df.head())

    # Load the simulated projections
    simulated_projections_df = pd.read_csv(uploaded_projections_file, index_col=0)
    st.write("Simulated Projections Data Preview:")
    st.dataframe(simulated_projections_df.head())

    # Number of simulations
    num_simulations = simulated_projections_df.shape[1]

    if st.button("Apply Projections to Teams"):
        try:
            team_results_df = apply_projections_to_teams(draft_results_df, simulated_projections_df, num_simulations)
            st.write("Team Results:")
            st.dataframe(team_results_df)

            # Download link for the team results
            csv = team_results_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Download Team Results",
                data=csv,
                file_name='team_results.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"An error occurred while applying projections: {str(e)}")
