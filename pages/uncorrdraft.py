import streamlit as st
import pandas as pd
import numpy as np

# Function to simulate a single draft
def simulate_draft(df, starting_team_num, num_teams=6, num_rounds=6):
    df_copy = df.copy()
    df_copy['Simulated ADP'] = np.random.normal(df_copy['adp'], df_copy['adpsd'])
    df_copy.sort_values('Simulated ADP', inplace=True)
    
    # Initialize the teams
    teams = {f'Team {i + starting_team_num}': [] for i in range(num_teams)}
    
    # Snake draft order
    for round_num in range(num_rounds):
        draft_order = list(range(num_teams)) if round_num % 2 == 0 else list(range(num_teams))[::-1]
        for pick_num in draft_order:
            if not df_copy.empty:
                selected_player = df_copy.iloc[0]
                teams[f'Team {pick_num + starting_team_num}'].append(selected_player)
                df_copy = df_copy.iloc[1:]
    
    return teams

# Function to run multiple simulations
def run_simulations(df, num_simulations=1124, num_teams=6, num_rounds=6):
    all_drafts = []

    for sim_num in range(num_simulations):
        starting_team_num = sim_num * num_teams + 1
        draft_result = simulate_draft(df, starting_team_num, num_teams, num_rounds)
        all_drafts.append(draft_result)
    
    return all_drafts

# Streamlit application
st.title("Fantasy Basketball Draft Simulation")

# File upload
uploaded_file = st.file_uploader("Upload your fantasy basketball players CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Create a unique identifier for each player if not already in the CSV
    if 'player_id' not in df.columns:
        df['player_id'] = df.index
    
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Simulation parameters
    num_simulations = st.number_input("Number of simulations", min_value=1, value=1124)
    num_teams = st.number_input("Number of teams", min_value=2, value=6)
    num_rounds = st.number_input("Number of rounds", min_value=1, value=6)
    
    if st.button("Run Simulation"):
        with st.spinner('Running simulations...'):
            all_drafts = run_simulations(df, num_simulations, num_teams, num_rounds)
        
        st.success('Simulations complete!')

        # Save the draft results to a DataFrame
        draft_results = []
        for sim_num, draft in enumerate(all_drafts):
            for team, players in draft.items():
                result_entry = {
                    'Simulation': sim_num + 1,
                    'Team': team,
                }
                for i, player in enumerate(players):
                    result_entry.update({
                        f'name': player['name'],
                        f'Player_{i+1}_Simulated_ADP': player['Simulated ADP']
                    })
                draft_results.append(result_entry)

        draft_results_df = pd.DataFrame(draft_results)
        
        # Display the first few rows of the results
        st.dataframe(draft_results_df.head())

        # Download link for the results
        csv = draft_results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Draft Results",
            data=csv,
            file_name='draft_results.csv',
            mime='text/csv',
        )
