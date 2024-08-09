import streamlit as st
import pandas as pd
import numpy as np

# Function to gather all player names from columns like Player_1_Name, Player_2_Name, etc.
def gather_team_players(draft_results_df):
    team_players = {}
    for _, row in draft_results_df.iterrows():
        team = row['Team']
        if team not in team_players:
            team_players[team] = []
        for col in row.index:
            if col.startswith('Player_') and col.endswith('_Name'):
                player_name = row[col]
                if pd.notna(player_name):
                    team_players[team].append(player_name)
    return team_players

# Function to apply simulated projections to teams
def apply_projections_to_teams(draft_results_df, simulated_projections_df, num_simulations):
    team_players = gather_team_players(draft_results_df)
    team_results = {team: [] for team in team_players.keys()}

    for sim in range(num_simulations):
        for team, players in team_players.items():
            try:
                # Check if all player names exist in the projections
                missing_players = [player for player in players if player not in simulated_projections_df.index]
                if missing_players:
                    st.error(f"Missing players in projections for simulation {sim}: {missing_players}")
                    raise KeyError(f"Missing players: {missing_players}")

                # Sum the projections for all players in the team for this simulation
                team_score = simulated_projections_df.loc[players, str(sim)].sum()
                team_results[team].append(team_score)
            except KeyError as e:
                st.error(f"Player {str(e)} not found in projections.")
                raise
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                raise

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

    # Ensure the column names in simulated projections are strings (in case they were read as integers)
    simulated_projections_df.columns = simulated_projections_df.columns.astype(str)

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
