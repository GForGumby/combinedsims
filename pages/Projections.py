import streamlit as st
import pandas as pd
import numpy as np
from scipy.linalg import cholesky, LinAlgError

# Function to ensure the covariance matrix is positive definite
def nearest_positive_definite(A):
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
    try:
        _ = cholesky(x)
        return True
    except LinAlgError:
        return False

# Function to simulate projections
def simulate_projections(player_names, means, std_devs, num_simulations):
    correlation_matrix = np.identity(len(player_names))  # Identity matrix (no correlation)
    covariance_matrix = np.outer(std_devs, std_devs) * correlation_matrix
    covariance_matrix = nearest_positive_definite(covariance_matrix)
    L = cholesky(covariance_matrix, lower=True)

    random_normals = np.random.normal(size=(len(player_names), num_simulations))
    correlated_projections = means[:, None] + np.dot(L, random_normals)

    return pd.DataFrame(correlated_projections, index=player_names)

# Streamlit UI
st.title('Projection Simulator')

# File upload for player projections
uploaded_file = st.file_uploader("Upload your player projections CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)
    st.write("Player Projections Data Preview:")
    st.dataframe(df.head())

    # Assuming the CSV contains 'player_name', 'proj', and 'projsd' columns
    player_names = df['player_name'].values
    means = df['proj'].values
    std_devs = df['projsd'].values

    # Number of simulations
    num_simulations = st.number_input("Number of simulations", min_value=1, value=1000)

    if st.button("Run Simulation"):
        try:
            simulated_data = simulate_projections(player_names, means, std_devs, num_simulations)
            st.write("Simulated Projections:")
            st.dataframe(simulated_data)

            # Download link for the simulated projections
            csv = simulated_data.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Download Simulated Projections",
                data=csv,
                file_name='simulated_projections.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"An error occurred during simulation: {str(e)}")
