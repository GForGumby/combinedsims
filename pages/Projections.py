import streamlit as st
import pandas as pd
import numpy as np
from scipy.linalg import cholesky

# Function to load data from the uploaded file
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to simulate projections for each player independently
def simulate_individual_projections(projection_lookup, num_simulations):
    simulations = {}
    for player, (mean, std_dev) in projection_lookup.items():
        simulations[player] = np.random.normal(mean, std_dev, num_simulations)
    return simulations

# Function to calculate team scores by summing player projections
def calculate_team_scores(draft_results, player_simulations, num_simulations):
    num_teams = draft_results.shape[0]
    team_scores = np.zeros((num_teams, num_simulations))

    for i in range(num_teams):
        team_player_names = draft_results[i]
        for player in team_player_names:
            team_scores[i] += player_simulations[player]

    avg_scores = np.mean(team_scores, axis=1)
    return avg_scores

def run_simulation(num_simulations, draft_results_df, projection_lookup):
    # Prepare draft results for each team
    draft_results, player_positions, player_teams, teams = prepare_draft_results(draft_results_df)

    # Simulate individual player projections
    player_simulations = simulate_individual_projections(projection_lookup, num_simulations)

    # Calculate team scores based on player simulations
    avg_scores = calculate_team_scores(draft_results, player_simulations, num_simulations)

    final_results = pd.DataFrame({
        'Team': teams,
        'Average Points': avg_scores
    })

    return final_results

# Streamlit app
st.title('Fantasy Football Projection Simulator')

uploaded_draft_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])
uploaded_projections_file = st.file_uploader("Upload your custom projections CSV file", type=["csv"])

if uploaded_draft_file and uploaded_projections_file:
    draft_results_df = load_data(uploaded_draft_file)
    custom_projections_df = load_data(uploaded_projections_file)

    st.write("Draft Results Data Preview:")
    st.dataframe(draft_results_df.head())

    st.write("Custom Projections Data Preview:")
    st.dataframe(custom_projections_df.head())

    projection_lookup = {
        row['player_name']: (row['proj'], row['projsd'])
        for _, row in custom_projections_df.iterrows()
    }

    num_simulations = st.number_input("Number of simulations", min_value=1, value=1000)

    if st.button("Run Projection Simulation"):
        try:
            st.write("Running simulations...")
            final_results = run_simulation(num_simulations, draft_results_df, projection_lookup)
            st.write("Simulation completed. Displaying results...")
            st.dataframe(final_results)

            csv = final_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Projection Results",
                data=csv,
                file_name='projection_results.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"An error occurred during simulation: {e}")
else:
    st.write("Please upload both the draft results and custom projections CSV files.")
