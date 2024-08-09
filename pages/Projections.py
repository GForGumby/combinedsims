import streamlit as st
import pandas as pd
import numpy as np
from scipy.linalg import cholesky

# Function to load data from the uploaded file
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to prepare draft results in numpy array format
def prepare_draft_results(draft_results_df):
    teams = draft_results_df['Team'].unique()
    num_teams = len(teams)
    draft_results = np.empty((num_teams, 6), dtype='U50')
    player_positions = np.empty((num_teams, 6), dtype='U3')
    player_teams = np.empty((num_teams, 6), dtype='U50')

    for idx, team in enumerate(teams):
        team_players = draft_results_df[draft_results_df['Team'] == team]
        for i in range(1, 7):
            draft_results[idx, i - 1] = f"{team_players.iloc[0][f'Player_{i}_Name']}"
            player_positions[idx, i - 1] = f"{team_players.iloc[0][f'Player_{i}_Position']}"
            player_teams[idx, i - 1] = f"{team_players.iloc[0][f'Player_{i}_Team']}"

    return draft_results, player_positions, player_teams, teams

# Function to create a simplified correlation matrix based on real-life NFL teams and positions
def create_correlation_matrix(player_teams, player_positions):
    num_players = player_teams.size
    correlation_matrix = np.identity(num_players)

    for i in range(num_players):
        for j in range(i + 1, num_players):
            if player_teams.flat[i] == player_teams.flat[j]:
                if player_positions.flat[i] == 'QB':
                    if player_positions.flat[j] == 'WR':
                        correlation_matrix[i, j] = 0.35
                        correlation_matrix[j, i] = 0.35
                    elif player_positions.flat[j] == 'TE':
                        correlation_matrix[i, j] = 0.25
                        correlation_matrix[j, i] = 0.25
                    elif player_positions.flat[j] == 'RB':
                        correlation_matrix[i, j] = 0.1
                        correlation_matrix[j, i] = 0.1
                elif player_positions.flat[j] == 'QB':
                    if player_positions.flat[i] == 'WR':
                        correlation_matrix[i, j] = 0.35
                        correlation_matrix[j, i] = 0.35
                    elif player_positions.flat[i] == 'TE':
                        correlation_matrix[i, j] = 0.25
                        correlation_matrix[j, i] = 0.25
                    elif player_positions.flat[i] == 'RB':
                        correlation_matrix[i, j] = 0.1
                        correlation_matrix[j, i] = 0.1

    return correlation_matrix

# Function to generate correlated projections
def generate_correlated_projections(player_names, player_positions, player_teams, projection_lookup, correlation_matrix):
    num_players = len(player_names)
    mean = np.array([projection_lookup[name][0] for name in player_names])
    std_dev = np.array([projection_lookup[name][1] for name in player_names])

    cov_matrix = np.outer(std_dev, std_dev) * correlation_matrix
    L = cholesky(cov_matrix, lower=True)

    random_normals = np.random.normal(size=num_players)
    correlated_normals = np.dot(L, random_normals)
    correlated_projections = mean + correlated_normals

    return correlated_projections

# Function to simulate team projections from draft results
def simulate_team_projections(draft_results, player_positions, player_teams, projection_lookup, num_simulations):
    num_teams = draft_results.shape[0]
    total_points = np.zeros((num_teams, num_simulations))

    for sim in range(num_simulations):
        for i in range(num_teams):
            team_player_names = draft_results[i]
            team_player_positions = player_positions[i]
            team_player_teams = player_teams[i]
            correlation_matrix = create_correlation_matrix(team_player_teams, team_player_positions)
            correlated_projections = generate_correlated_projections(team_player_names, team_player_positions, team_player_teams, projection_lookup, correlation_matrix)
            total_points[i, sim] = np.sum(correlated_projections)

    avg_points = np.mean(total_points, axis=1)
    return avg_points

def run_simulation(num_simulations, draft_results_df, projection_lookup):
    draft_results, player_positions, player_teams, teams = prepare_draft_results(draft_results_df)
    avg_points = simulate_team_projections(draft_results, player_positions, player_teams, projection_lookup, num_simulations)

    final_results = pd.DataFrame({
        'Team': teams,
        'Average Points': avg_points
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
