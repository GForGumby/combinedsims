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

# Function to simulate team projections from draft results without using numba
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

def run_simulations(num_simulations, draft_results_df, projection_lookup):
    draft_results, teams = prepare_draft_results(draft_results_df)
    avg_payouts = simulate_team_projections(draft_results, projection_lookup, num_simulations)
    
    # Prepare final results
    final_results = pd.DataFrame({
        'Team': teams,
        'Average_Payout': avg_payouts
    })
    
    return final_results

# Streamlit app
st.title('Fantasy Football Projections and Payout Simulator')

# Upload for draft results
uploaded_draft_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])

# Define your projections here or load them from a file if needed
projections = {
    "Christian McCaffrey": {'proj': 30, 'projsd': 9},
    # Add more players as needed...
}

if uploaded_draft_file:
    draft_results_df = pd.read_csv(uploaded_draft_file)

    # Create a projection lookup dictionary for quick access
    projection_lookup = {
        name: (projections[name]['proj'], projections[name]['projsd'])
        for name in projections
    }

    # Number of simulations input
    num_simulations = st.number_input("Number of simulations", min_value=1, value=1000, step=1)

    if st.button("Run Simulations"):
        with st.spinner('Running simulations...'):
            try:
                final_results = run_simulations(num_simulations, draft_results_df, projection_lookup)
                st.success("Simulations completed!")

                # Display results
                st.write("Average Payouts per Team")
                st.dataframe(final_results)

                # Download results as CSV
                csv = final_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name='simulation_results.csv',
                    mime='text/csv'
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
