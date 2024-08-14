import streamlit as st
import pandas as pd
import numpy as np
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
        return 100000.00
    elif rank == 2:
        return 35000.00
    elif rank == 3:
        return 20000.00
    elif rank == 4:
        return 10000.00
    elif rank == 5:
        return 4000.00
    elif rank == 6:
        return 2000.00
    elif rank == 7:
        return 1500.00
    elif rank in [8, 9]:
        return 1000.00
    elif rank in [10, 11]:
        return 750.00
    elif rank in range(12, 15):
        return 600.00
    elif rank in range(15, 18):
        return 500.00
    elif rank in range(18, 22):
        return 400.00
    elif rank in range(22, 26):
        return 350.00
    elif rank in range(26, 36):
        return 300.00
    elif rank in range(36, 56):
        return 250.00
    elif rank in range(56, 76):
        return 200.00
    elif rank in range(76, 106):
        return 150.00
    elif rank in range(106, 156):
        return 120.00
    elif rank in range(156, 216):
        return 100.00
    elif rank in range(216, 316):
        return 90.00
    elif rank in range(316, 416):
        return 80.00
    elif rank in range(416, 516):
        return 70.00
    elif rank in range(516, 666):
        return 60.00
    elif rank in range(666, 866):
        return 50.00
    elif rank in range(866, 1166):
        return 45.00
    elif rank in range(1166, 1671):
        return 40.00
    elif rank in range(1671, 2571):
        return 35.00
    elif rank in range(2571, 5271):
        return 30.00
    else:
        return 0.00

# Function to prepare draft results in numpy array format
def prepare_draft_results(draft_results_df):
    teams = draft_results_df['Team'].unique()
    num_teams = len(teams)
    draft_results = np.empty((num_teams, 6), dtype='U50')

    for idx, team in enumerate(teams):
        team_players = draft_results_df[draft_results_df['Team'] == team]
        for i in range(1, 7):
            if i <= len(team_players):
                draft_results[idx, i - 1] = f"{team_players.iloc[i - 1]['G']}"
            else:
                draft_results[idx, i - 1] = "N/A"  # Placeholder for missing players

    return draft_results, teams

# Function to simulate team projections
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

def run_parallel_simulations(num_simulations, draft_results_df, projection_lookup):
    draft_results, teams = prepare_draft_results(draft_results_df)
    avg_payouts = simulate_team_projections(draft_results, projection_lookup, num_simulations)
    
    # Prepare final results
    final_results = pd.DataFrame({
        'Team': teams,
        'Average_Payout': avg_payouts
    })
    
    return final_results

# Streamlit application
st.title('Fantasy Sports Simulation')

# Upload draft results CSV
draft_file = st.file_uploader("Choose a draft results file (CSV)", type="csv")
projection_file = st.file_uploader("Choose a projections file (CSV)", type="csv")

if draft_file is not None and projection_file is not None:
    draft_results_df = pd.read_csv(draft_file)
    projections_df = pd.read_csv(projection_file)
    
    # Ensure the projections file has the necessary columns
    if 'DFS_ID' in projections_df.columns and 'proj' in projections_df.columns and 'projsd' in projections_df.columns:
        # Create a projection lookup dictionary for quick access
        projection_lookup = {
            str(row['DFS_ID']): (row['proj'], row['projsd']) 
            for index, row in projections_df.iterrows()
        }

        # Number of simulations
        num_simulations = st.number_input("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000)

        # Run simulations
        if st.button('Run Simulations'):
            with st.spinner('Simulating...'):
                final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)
            
            st.success('Simulations complete!')

            # Display the final results
            st.write(final_results)

            # Option to download the results as a CSV
            csv = final_results.to_csv(index=False)
            st.download_button("Download Results", data=csv, file_name='simulation_results.csv', mime='text/csv')
    else:
        st.error("Projections file must contain 'DFS_ID', 'proj', and 'projsd' columns.")
