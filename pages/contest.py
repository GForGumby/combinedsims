import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor

# Define this function at the top level of the script
def calculate_simulation_payouts(sim, draft_results_df, projections_df):
    total_projections = []

    for team in draft_results_df['Team'].unique():
        team_data = draft_results_df[draft_results_df['Team'] == team]
        team_projection = team_data.apply(lambda row: projections_df.loc[row.filter(regex='^Player_\d+_Name$'), sim].sum(), axis=1).sum()
        total_projections.append((team, team_projection))

    total_projections.sort(key=lambda x: x[1], reverse=True)
    
    payouts = {}
    for rank, (team, _) in enumerate(total_projections):
        if rank == 0:
            payout = 1000
        elif 1 <= rank < 10:
            payout = 100
        else:
            payout = 0
        payouts[team] = payouts.get(team, 0) + payout

    return payouts

# Function to apply payouts concurrently using threads
def apply_payouts_concurrently(draft_results_df, projections_df):
    team_payouts = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(calculate_simulation_payouts, sim, draft_results_df, projections_df): sim for sim in projections_df.columns}
        for future in futures:
            sim_payouts = future.result()
            for team, payout in sim_payouts.items():
                team_payouts[team] = team_payouts.get(team, 0) + payout
    
    return pd.DataFrame(list(team_payouts.items()), columns=['Team', 'Total_Payout'])

# Streamlit UI
st.title('Fantasy Football Payout Application')

# File upload for draft results
uploaded_draft_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])

# File upload for simulated projections
uploaded_projections_file = st.file_uploader("Upload your simulated projections CSV file", type=["csv"])

if uploaded_draft_file and uploaded_projections_file:
    draft_results_df = pd.read_csv(uploaded_draft_file)
    projections_df = pd.read_csv(uploaded_projections_file, index_col=0)

    st.write("Draft Results Data Preview:")
    st.dataframe(draft_results_df.head())

    st.write("Simulated Projections Data Preview:")
    st.dataframe(projections_df.head())

    st.write("Starting payout calculation...")  # Log message
    try:
        team_payouts = apply_payouts_concurrently(draft_results_df, projections_df)

        st.write("Team Payout Results:")
        st.dataframe(team_payouts)

        csv = team_payouts.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Team Payout Results",
            data=csv,
            file_name='team_payout_results.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"An error occurred while applying payouts: {e}")
