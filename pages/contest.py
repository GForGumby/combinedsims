import pandas as pd
import streamlit as st

# Step 1: Apply Projections to Teams
def apply_projections_to_teams(draft_results_df, projections_df):
    # Initialize a dictionary to store team projection totals
    team_results = {}

    # Iterate over each team in the draft results
    for team in draft_results_df['Team'].unique():
        team_data = draft_results_df[draft_results_df['Team'] == team]
        total_projections = []

        # Sum the projections for each player on the team across all simulations
        for _, row in team_data.iterrows():
            player_projections = projections_df.loc[row.filter(regex='^Player_\d+_Name$')].sum()
            total_projections.append(player_projections)

        # Sum the projections for all players in the team
        team_results[team] = pd.concat(total_projections, axis=1).sum(axis=1)
    
    return pd.DataFrame(team_results)

# Streamlit UI
st.title('Fantasy Football Projection Application')

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

    # Apply projections to teams
    try:
        team_results = apply_projections_to_teams(draft_results_df, projections_df)

        # Display the results
        st.write("Team Results:")
        st.dataframe(team_results.T)

        # Download link for the results
        csv = team_results.T.to_csv().encode('utf-8')
        st.download_button(
            label="Download Team Results",
            data=csv,
            file_name='team_results.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"An error occurred while applying projections: {e}")
