import streamlit as st
import pandas as pd
import numpy as np

# Function to generate projection
def generate_projection(median, std_dev):
    fluctuation = np.random.uniform(-0.01, 0.01) * median
    return max(0, np.random.normal(median, std_dev) + fluctuation)

# Combined function to simulate a single draft and projections
def simulate_draft_and_projections(df, num_teams=6, num_rounds=6, team_bonus=0.95):
    df_copy = df.copy()
    df_copy['Simulated ADP'] = np.random.normal(df_copy['adp'].values, df_copy['adpsd'].values)
    df_copy.sort_values('Simulated ADP', inplace=True)
    
    # Initialize the teams
    teams = {f'Team {i+1}': [] for i in range(num_teams)}
    team_positions = {f'Team {i+1}': {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "FLEX": 0} for i in range(num_teams)}
    teams_stack = {f'Team {i+1}': [] for i in range(num_teams)}
    
    # Snake draft order
    for round_num in range(num_rounds):
        draft_order = list(range(num_teams)) if round_num % 2 == 0 else list(range(num_teams))[::-1]
        for pick_num in draft_order:
            if len(df_copy) > 0:
                team_name = f'Team {pick_num+1}'
                
                # Filter players based on positional requirements
                draftable_positions = []
                if team_positions[team_name]["QB"] < 1:
                    draftable_positions.append("QB")
                if team_positions[team_name]["RB"] < 1:
                    draftable_positions.append("RB")
                if team_positions[team_name]["WR"] < 2:
                    draftable_positions.append("WR")
                if team_positions[team_name]["TE"] < 1:
                    draftable_positions.append("TE")
                if team_positions[team_name]["FLEX"] < 1 and (team_positions[team_name]["RB"] + team_positions[team_name]["WR"] < 5):
                    draftable_positions.append("FLEX")
                
                df_filtered = df_copy.loc[
                    df_copy['position'].isin(draftable_positions) | 
                    ((df_copy['position'].isin(['RB', 'WR'])) & ('FLEX' in draftable_positions))
                ].copy()
                
                if len(df_filtered) == 0:
                    continue
                
                # Adjust Simulated ADP based on team stacking
                df_filtered['Adjusted ADP'] = df_filtered.apply(
                    lambda x: x['Simulated ADP'] * team_bonus 
                    if x['team'] in teams_stack[team_name] else x['Simulated ADP'],
                    axis=1
                )
                
                df_filtered.sort_values('Adjusted ADP', inplace=True)
                
                selected_player = df_filtered.iloc[0]
                simulated_points = generate_projection(selected_player['proj'], selected_player['projsd'])
                teams[team_name].append((selected_player, simulated_points))
                teams_stack[team_name].append(selected_player['team'])
                
                position = selected_player['position']
                if position in ["RB", "WR"]:
                    if team_positions[team_name][position] < {"RB": 1, "WR": 2}[position]:
                        team_positions[team_name][position] += 1
                    else:
                        team_positions[team_name]["FLEX"] += 1
                else:
                    team_positions[team_name][position] += 1
                df_copy = df_copy.loc[df_copy['player_id'] != selected_player['player_id']]
    
    return teams

# Function to run multiple simulations and organize results by draft position
def run_simulations(df, num_simulations=10, num_teams=6, num_rounds=6, team_bonus=0.95):
    all_drafts = {f'Team {i+1}': [] for i in range(num_teams)}

    for sim_num in range(num_simulations):
        draft_result = simulate_draft_and_projections(df, num_teams, num_rounds, team_bonus)
        for team, players in draft_result.items():
            for player, points in players:
                all_drafts[team].append({
                    'Simulation': sim_num + 1,
                    'Player_Name': player['name'],
                    'Position': player['position'],
                    'Team': player['team'],
                    'Simulated_Points': points
                })
    
    return all_drafts

# Streamlit app
st.title('Fantasy Sports Draft Simulation with Projections')

# File upload
uploaded_file = st.file_uploader("Upload your ADP CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check if player_id exists, if not, create it
    if 'player_id' not in df.columns:
        df['player_id'] = df.index
    
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    # Parameters for the simulation
    num_simulations = st.number_input("Number of simulations", min_value=1, value=10)
    num_teams = st.number_input("Number of teams", min_value=2, value=6)
    num_rounds = st.number_input("Number of rounds", min_value=1, value=6)
    team_bonus = st.number_input("Team stacking bonus", min_value=0.0, value=0.95)
    
    if st.button("Run Simulation"):
        all_drafts = run_simulations(df, num_simulations, num_teams, num_rounds, team_bonus)

        # Create a Pandas Excel writer to save multiple sheets
        with pd.ExcelWriter("draft_results_with_projections.xlsx", engine='xlsxwriter') as writer:
            for team, results in all_drafts.items():
                df_results = pd.DataFrame(results)
                df_results.to_excel(writer, sheet_name=team, index=False)
        
        # Provide download link for the Excel file
        with open("draft_results_with_projections.xlsx", "rb") as file:
            st.download_button(
                label="Download Draft Results",
                data=file,
                file_name="draft_results_with_projections.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
