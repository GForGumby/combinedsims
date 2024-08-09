import numpy as np
from scipy.linalg import cholesky, LinAlgError
from scipy.linalg import eigh
from numpy import sqrt, diag

def nearest_positive_definite(A):
    """Find the nearest positive-definite matrix to A."""
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def is_positive_definite(x):
    """Check if a matrix is positive definite."""
    try:
        _ = cholesky(x)
        return True
    except LinAlgError:
        return False

# Function to simulate projections with correlations
def simulate_correlated_projections(draft_results, player_positions, projection_lookup, num_simulations):
    player_names = [player for player in projection_lookup.keys()]
    means = np.array([projection_lookup[player][0] for player in player_names])
    std_devs = np.array([projection_lookup[player][1] for player in player_names])

    # Create the correlation matrix for all players
    correlation_matrix = np.identity(len(player_names))
    for i in range(len(player_names)):
        for j in range(i + 1, len(player_names)):
            if player_positions.flat[i] == 'QB' and player_positions.flat[j] in ['WR', 'TE']:
                correlation_matrix[i, j] = correlation_matrix[j, i] = 0.35
            elif player_positions.flat[i] in ['WR', 'TE'] and player_positions.flat[j] == 'QB':
                correlation_matrix[i, j] = correlation_matrix[j, i] = 0.35
            elif player_positions.flat[i] == player_positions.flat[j] == 'RB':
                correlation_matrix[i, j] = correlation_matrix[j, i] = 0.1

    # Covariance matrix
    covariance_matrix = np.outer(std_devs, std_devs) * correlation_matrix

    # Ensure the covariance matrix is positive definite
    covariance_matrix = nearest_positive_definite(covariance_matrix)

    # Cholesky decomposition
    L = cholesky(covariance_matrix, lower=True)

    # Generate correlated projections for all players
    random_normals = np.random.normal(size=(len(player_names), num_simulations))
    correlated_projections = means[:, None] + np.dot(L, random_normals)

    return {player_names[i]: correlated_projections[i] for i in range(len(player_names))}

# The rest of your code remains unchanged.

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
    player_simulations = simulate_correlated_projections(draft_results, player_positions, projection_lookup, num_simulations)

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
