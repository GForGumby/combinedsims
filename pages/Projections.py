import streamlit as st
import pandas as pd
import numpy as np
from scipy.linalg import cholesky

# Function to load data from the uploaded file
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to simulate projections with correlations
def simulate_correlated_projections(draft_results, projection_lookup, num_simulations):
    player_names = [player for player in projection_lookup.keys()]
    means = np.array([projection_lookup[player][0] for player in player_names])
    std_devs = np.array([projection_lookup[player][1] for player in player_names])

    # Create the correlation matrix for all players
    correlation_matrix = np.identity(len(player_names))
    for i in range(len(player_names)):
        for j in range(i + 1, len(player_names)):
            if player_names[i] in player_positions and player_names[j] in player_positions:
                correlation_matrix[i, j] = correlation_matrix[j, i] = 0.35

    # Covariance matrix
    covariance_matrix = np.outer(std_devs, std_devs) * correlation_matrix

    # Cholesky decomposition
    L = cholesky(covariance_matrix, lower=True)

    # Generate correlated projections for all players
    random_normals = np.random.normal(size=(len(player_names), num_simulations))
    correlated_projections = means[:, None] + np.dot(L, random_normals)

    return {player_names[i]: correlated_projections[i] for i in range(len(player_names))}

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

    # Simulate correlated player projections
    player_simulations = simulate_correlated_projections(draft_results, projection_lookup, num_simulations)

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
