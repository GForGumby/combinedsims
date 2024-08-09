import streamlit as st
import pandas as pd
import numpy as np
from numba import jit
from scipy.linalg import cholesky

# Define the streamlit app
st.title('Fantasy Football Projection Simulator')

# File upload for draft results
uploaded_draft_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])

# File upload for custom projections
uploaded_projections_file = st.file_uploader("Upload your custom projections CSV file", type=["csv"])

projections = {"Christian McCaffrey": (6.0, 4.0),
"CeeDee Lamb": (6.0, 4.0),
"Tyreek Hill": (6.0, 4.0),
"Ja'Marr Chase": (6.0, 4.0),
"Justin Jefferson": (6.0, 4.0),
"Amon-Ra St. Brown": (6.0, 4.0),
"Bijan Robinson": (6.0, 4.0),
"Breece Hall": (6.0, 4.0),
"A.J. Brown": (6.0, 4.0),
"Puka Nacua": (6.0, 4.0),
"Garrett Wilson": (6.0, 4.0),
"Jahmyr Gibbs": (6.0, 4.0),
"Marvin Harrison": (6.0, 4.0),
"Drake London": (6.0, 4.0),
"Jonathan Taylor": (6.0, 4.0),
"Nico Collins": (6.0, 4.0),
"Chris Olave": (6.0, 4.0),
"Deebo Samuel": (6.0, 4.0),
"Saquon Barkley": (6.0, 4.0),
"Jaylen Waddle": (6.0, 4.0),
"Davante Adams": (6.0, 4.0),
"Brandon Aiyuk": (6.0, 4.0),
"De'Von Achane": (6.0, 4.0),
"Mike Evans": (6.0, 4.0),
"DeVonta Smith": (6.0, 4.0),
"DK Metcalf": (6.0, 4.0),
"Malik Nabers": (6.0, 4.0),
"Cooper Kupp": (6.0, 4.0),
"Kyren Williams": (6.0, 4.0),
"Derrick Henry": (6.0, 4.0),
"DJ Moore": (6.0, 4.0),
"Stefon Diggs": (6.0, 4.0),
"Michael Pittman Jr.": (6.0, 4.0),
"Tank Dell": (6.0, 4.0),
"Sam LaPorta": (6.0, 4.0),
"Zay Flowers": (6.0, 4.0),
"Josh Allen": (6.0, 4.0),
"Travis Kelce": (6.0, 4.0),
"George Pickens": (6.0, 4.0),
"Isiah Pacheco": (6.0, 4.0),
"Amari Cooper": (6.0, 4.0),
"Jalen Hurts": (6.0, 4.0),
"Tee Higgins": (6.0, 4.0),
"Travis Etienne Jr.": (6.0, 4.0),
"Patrick Mahomes": (6.0, 4.0),
"Christian Kirk": (6.0, 4.0),
"Trey McBride": (6.0, 4.0),
"Lamar Jackson": (6.0, 4.0),
"Mark Andrews": (6.0, 4.0),
"Terry McLaurin": (6.0, 4.0),
"Dalton Kincaid": (6.0, 4.0),
"Josh Jacobs": (6.0, 4.0),
"Hollywood Brown": (6.0, 4.0),
"Keenan Allen": (6.0, 4.0),
"James Cook": (6.0, 4.0),
"Anthony Richardson": (6.0, 4.0),
"Jayden Reed": (6.0, 4.0),
"Calvin Ridley": (6.0, 4.0),
"Chris Godwin": (6.0, 4.0),
"Rashee Rice": (6.0, 4.0),
"Keon Coleman": (6.0, 4.0),
"Kyler Murray": (6.0, 4.0),
"Aaron Jones": (6.0, 4.0),
"DeAndre Hopkins": (6.0, 4.0),
"Rhamondre Stevenson": (6.0, 4.0),
"James Conner": (6.0, 4.0),
"Najee Harris": (6.0, 4.0),
"Jameson Williams": (6.0, 4.0),
"Jake Ferguson": (6.0, 4.0),
"Jordan Addison": (6.0, 4.0),
"Curtis Samuel": (6.0, 4.0),
"Jaylen Warren": (6.0, 4.0),
"Zamir White": (6.0, 4.0),
"Joe Burrow": (6.0, 4.0),
"Jonathon Brooks": (6.0, 4.0),
"D'Andre Swift": (6.0, 4.0),
"Raheem Mostert": (6.0, 4.0),
"Dak Prescott": (6.0, 4.0),
"Courtland Sutton": (6.0, 4.0),
"Brock Bowers": (6.0, 4.0),
"Jordan Love": (6.0, 4.0),
"Zack Moss": (6.0, 4.0),
"Joshua Palmer": (6.0, 4.0),
"David Njoku": (6.0, 4.0),
"Tony Pollard": (6.0, 4.0),
"Jayden Daniels": (6.0, 4.0),
"Brian Robinson Jr.": (6.0, 4.0),
"Romeo Doubs": (6.0, 4.0),
"Rashid Shaheed": (6.0, 4.0),
"Tyler Lockett": (6.0, 4.0),
"Tyjae Spears": (6.0, 4.0),
"Chase Brown": (6.0, 4.0),
"Devin Singletary": (6.0, 4.0),
"Khalil Shakir": (6.0, 4.0),
"Brock Purdy": (6.0, 4.0),
"Javonte Williams": (6.0, 4.0),
"Caleb Williams": (6.0, 4.0),
"Dontayvion Wicks": (6.0, 4.0),
"Brandin Cooks": (6.0, 4.0),
"Dallas Goedert": (6.0, 4.0),
"Trey Benson": (6.0, 4.0),
"Trevor Lawrence": (6.0, 4.0),
"Gus Edwards": (6.0, 4.0),
"Jakobi Meyers": (6.0, 4.0),
"Blake Corum": (6.0, 4.0),
"Ezekiel Elliott": (6.0, 4.0),
"Jerry Jeudy": (6.0, 4.0),
"Tua Tagovailoa": (6.0, 4.0),
"Jared Goff": (6.0, 4.0),
"Adonai Mitchell": (6.0, 4.0),
"Jerome Ford": (6.0, 4.0),
"Nick Chubb": (6.0, 4.0),
"Ja'Lynn Polk": (6.0, 4.0),
"Pat Freiermuth": (6.0, 4.0),
"Austin Ekeler": (6.0, 4.0),
"Dalton Schultz": (6.0, 4.0),
              }

if uploaded_draft_file is not None:
    draft_results_df = pd.read_csv(uploaded_draft_file)
    st.write("Draft Results Data Preview:")
    st.dataframe(draft_results_df.head())

    if uploaded_projections_file is not None:
        custom_projections_df = pd.read_csv(uploaded_projections_file)
        st.write("Custom Projections Data Preview:")
        st.dataframe(custom_projections_df.head())

        # Create a projection lookup dictionary from the custom projections
        projection_lookup = {}
        for _, row in custom_projections_df.iterrows():
            player_name = row['player_name']
            proj = row['proj']
            projsd = row['projsd']
            projection_lookup[player_name] = (proj, projsd)

        # Number of simulations for projection
        num_simulations = st.number_input("Number of simulations", min_value=1, value=1000)

        if st.button("Run Projection Simulation"):
           # Function to run parallel simulations
  def run_parallel_simulations(num_simulations, draft_results_df, projection_lookup):
     draft_results, player_positions, player_teams, teams = prepare_draft_results(draft_results_df)
    avg_payouts = simulate_team_projections(draft_results, player_positions, player_teams, projection_lookup, num_simulations)

          # Run simulations
            final_results = run_parallel_simulations(num_simulations, draft_results_df, projection_lookup)

            # Display the results
            st.dataframe(final_results)

            # Download link for the results
            csv = final_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Projection Results",
                data=csv,
                file_name='projection_results.csv',
                mime='text/csv',
            )

# Below is the code for projection simulations that you originally had in projections_sim.py

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
    total_payouts = np.zeros(num_teams)

    for sim in range(num_simulations):
        total_points = np.zeros(num_teams)
        for i in range(num_teams):
            team_player_names = draft_results[i]
            team_player_positions = player_positions[i]
            team_player_teams = player_teams[i]
            correlation_matrix = create_correlation_matrix(team_player_teams, team_player_positions)
            correlated_projections = generate_correlated_projections(team_player_names, team_player_positions, team_player_teams, projection_lookup, correlation_matrix)
            total_points[i] = np.sum(correlated_projections)

        # Rank teams
        ranks = total_points.argsort()[::-1].argsort() + 1

        # Assign payouts and accumulate them
        payouts = np.array([get_payout(rank) for rank in ranks])
        total_payouts += payouts

    # Calculate average payout per team
    avg_payouts = total_payouts / num_simulations
    return avg_payouts

    # Prepare final results
    final_results = pd.DataFrame({
        'Team': teams,
        'Average_Payout': avg_payouts
    })

    return final_results
