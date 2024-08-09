import pandas as pd
import streamlit as st

# Step 1: Apply Projections and Payouts to Teams
def apply_payouts_to_teams(draft_results_df, projections_df):
    # Initialize a dictionary to store team payouts
    team_payouts = {}

    # Iterate over each simulation
    for sim in projections_df.columns:
        # Initialize a list to store total projections for each team
        total_projections = []

        # Iterate over each team in the draft results
        for team in draft_results_df['Team'].unique():
            team_data = draft_results_df[draft_results_df['Team'] == team]
            
            # Sum the projections for the team in the current simulation
            team_projection = team_data.apply(lambda row: projections_df.loc[row.filter(regex='^Player_\d+_Name$'), sim].sum(), axis=1).sum()
            total_projections.append((team, team_projection))

        # Sort teams by their total projections for the current simulation
        total_projections.sort(key=lambda x: x[1], reverse=True)
        
        # Assign payouts based on the ranking
        for rank, (team, _) in enumerate(total_projections):
            if rank == 0:
                payout = 1000  # First place gets $1000
            elif 1 <= rank < 10:
                payout = 100  # Ranks 2-10 get $100
            else:
                payout = 0  # No payout for ranks 11 and beyond

            # Add the payout to the team's total payout across all simulations
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
    # Load the draft results and simulated projections
    draft_results_df = pd.read_csv(uploaded_draft_file)
    projections_df = pd.read_csv(uploaded_projections_file, index_col=0)

    st.write("Draft Results Data Preview:")
    st.dataframe(draft_results_df.head())

    st.write("Simulated Projections Data Preview:")
    st.dataframe(projections_df.head())

    # Apply payouts to teams
    try:
        team_payouts = apply_payouts_to_teams(draft_results_df, projections_df)

        # Display the results
        st.write("Team Payout Results:")
        st.dataframe(team_payouts)

        # Download link for the results
        csv = team_payouts.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Team Payout Results",
            data=csv,
            file_name='team_payout_results.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"An error occurred while applying payouts: {e}")
