import pandas as pd
import streamlit as st

# Step 1: Apply Projections and Payouts to Teams
def apply_payouts_to_teams(draft_results_df, projections_df):
    st.write("Applying payouts...")  # Log message
    team_payouts = {}

    for sim in projections_df.columns:
        st.write(f"Processing simulation: {sim}")  # Log message
        total_projections = []

        for team in draft_results_df['Team'].unique():
            st.write(f"Processing team: {team}")  # Log message
            team_data = draft_results_df[draft_results_df['Team'] == team]

            team_projection = team_data.apply(lambda row: projections_df.loc[row.filter(regex='^Player_\d+_Name$'), sim].sum(), axis=1).sum()
            total_projections.append((team, team_projection))

        total_projections.sort(key=lambda x: x[1], reverse=True)

        for rank, (team, _) in enumerate(total_projections):
            if rank == 0:
                payout = 1000
            elif 1 <= rank < 10:
                payout = 100
            else:
                payout = 0

            if team not in team_payouts:
                team_payouts[team] = 0
            team_payouts[team] += payout
    
    return pd.DataFrame(list(team_payouts.items()), columns=['Team', 'Total_Payout'])

# Streamlit UI
st.title('Fantasy Football Payout Application')

# File upload for draft results
uploaded_draft_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])

# File upload for simulated projections
uploaded_projections_file = st.file_uploader("Upload your simulated projections CSV file", type=["csv"])

if uploaded_draft_file and uploaded_projections_file:
    st.write("Loading files...")  # Log message
    draft_results_df = pd.read_csv(uploaded_draft_file)
    projections_df = pd.read_csv(uploaded_projections_file, index_col=0)

    st.write("Draft Results Data Preview:")
    st.dataframe(draft_results_df.head())

    st.write("Simulated Projections Data Preview:")
    st.dataframe(projections_df.head())

    st.write("Starting payout calculation...")  # Log message
    try:
        team_payouts = apply_payouts_to_teams(draft_results_df, projections_df)

        st.write("Displaying results...")  # Log message
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
