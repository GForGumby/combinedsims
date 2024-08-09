import pandas as pd

def apply_projections_to_teams(draft_results_df, projections_df):
    # Initialize a DataFrame to store results
    team_results = pd.DataFrame()

    for simulation in range(projections_df.shape[1]):
        sim_results = []
        for team in draft_results_df['Team'].unique():
            team_players = draft_results_df[draft_results_df['Team'] == team].filter(regex='^Player_\d+_Name$').values.flatten()
            team_projection = projections_df.loc[team_players, simulation].sum()
            sim_results.append({'Team': team, f'Simulation_{simulation}': team_projection})
        
        # Convert to DataFrame and merge
        sim_results_df = pd.DataFrame(sim_results)
        team_results = pd.merge(team_results, sim_results_df, on='Team', how='outer') if not team_results.empty else sim_results_df
    
    return team_results

# Example usage:
draft_results_df = pd.read_csv('path_to_draft_results.csv')
projections_df = pd.read_csv('path_to_projections.csv', index_col=0)

team_results = apply_projections_to_teams(draft_results_df, projections_df)

print(team_results.head())
