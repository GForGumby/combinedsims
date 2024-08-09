import pandas as pd
import numpy as np
import streamlit as st
from numba import jit

# JIT compiled function to generate projection
@jit(nopython=True)
def generate_projection(median, std_dev):
    fluctuation = np.random.uniform(-0.01, 0.01) * median
    return max(0, np.random.normal(median, std_dev) + fluctuation)

# JIT compiled function to get payout based on rank
@jit(nopython=True)
def get_payout(rank):
    if rank == 1:
        return 50000.00
    elif rank == 2:
        return 10000.00
    elif rank == 3:
        return 5000.00
    elif rank == 4:
        return 2500.00
    elif rank == 5:
        return 2000.00
    elif rank == 6:
        return 1500.00
    elif rank in [7, 8]:
        return 1000.00
    elif rank in [9, 10]:
        return 500.00
    elif rank in range(11, 15):
        return 400.00
    elif rank in range(15, 20):
        return 300.00
    elif rank in range(20, 26):
        return 250.00
    elif rank in range(26, 36):
        return 200.00
    elif rank in range(36, 51):
        return 150.00
    elif rank in range(51, 76):
        return 100.00
    elif rank in range(76, 126):
        return 75.00
    elif rank in range(126, 251):
        return 60.00
    elif rank in range(251, 491):
        return 50.00
    elif rank in range(491, 1526):
        return 40.00
    else:
        return 0

# Function to prepare draft results in numpy array format
def prepare_draft_results(draft_results_df):
    teams = draft_results_df['Team'].unique()
    num_teams = len(teams)
    draft_results = np.empty((num_teams, 6), dtype='U50')

    for idx, team in enumerate(teams):
        team_players = draft_results_df[draft_results_df['Team'] == team]
        for i in range(1, 7):
            player_column = f'Player_{i}_Name'
            if player_column in team_players.columns:
                draft_results[idx, i - 1] = f"{team_players.iloc[0][player_column]}"
            else:
                draft_results[idx, i - 1] = "N/A"  # Placeholder for missing players

    return draft_results, teams 

# Function to create a projection lookup dictionary from the CSV
def create_projection_lookup(projections_df):
    projection_lookup = {}
    for _, row in projections_df.iterrows():
        player_name = row['player_name']
        proj = row['proj']
        projsd = row['projsd']
        projection_lookup[player_name] = (proj, projsd)
    return projection_lookup

# Function to simulate team projections from draft results
def simulate_team_projections(draft_results, projection_lookup, num_simulations):
    num_teams = draft_results.shape[0]
    total_payouts = np.zeros(num_teams)

    for sim in range(num_simulations):
        total_points = np.zeros(num_teams)
        for i in range(num_teams):
            for j in range(6):  # Loop through all 6 players
                player_name = draft_results[i, j]
                if player_name in projection_lookup:
                    proj, projsd = projection_lookup[player_name]
                    simulated_points = generate_projection(proj, projsd)
                    total_points[i] += simulated_points

        # Rank teams
        ranks = total_points.argsort()[::-1].argsort() + 1

        # Assign payouts and accumulate them
        payouts = np.array([get_payout(rank) for rank in ranks])
        total_payouts += payouts

    # Calculate average payout per team
    avg_payouts = total_payouts / num_simulations
    return avg_payouts

# Streamlit app to handle file uploads and run simulations
st.title('Fantasy Football Payout Simulator')

# Upload projections CSV file
uploaded_projections_file = st.file_uploader("Upload your Projections CSV file", type=["csv"])

# Upload draft results CSV file
uploaded_draft_file = st.file_uploader("Upload your Draft Results CSV file", type=["csv"])

if uploaded_projections_file is not None and uploaded_draft_file is not None:
    projections_df = pd.read_csv(uploaded_projections_file)
    draft_results_df = pd.read_csv(uploaded_draft_file)

    # Create projection lookup dictionary
    projection_lookup = create_projection_lookup(projections_df)

    # Prepare draft results
    draft_results, teams = prepare_draft_results(draft_results_df)

    # Run simulations
    num_simulations = st.number_input("Number of simulations", min_value=1, value=1000)
    if st.button("Run Simulation"):
        avg_payouts = simulate_team_projections(draft_results, projection_lookup, num_simulations)
        
        # Prepare final results
        final_results = pd.DataFrame({
            'Team': teams,
            'Average_Payout': avg_payouts
        })

        # Display and download results
        st.write(final_results)
        csv = final_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Payout Results",
            data=csv,
            file_name='payout_results.csv',
            mime='text/csv',
        )
