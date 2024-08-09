import streamlit as st
import pandas as pd
from projections_sim import prepare_draft_results, run_parallel_simulations

st.title('Fantasy Football Projection Simulator')

# File upload for draft results
uploaded_draft_file = st.file_uploader("Upload your draft results CSV file", type=["csv"])

# File upload for custom projections
uploaded_projections_file = st.file_uploader("Upload your custom projections CSV file (optional)", type=["csv"])

if uploaded_draft_file is not None:
    draft_results_df = pd.read_csv(uploaded_draft_file)

    st.write("Draft Results Data Preview:")
    st.dataframe(draft_results_df.head())

    # Define default projections
    default_projections = {
  "Christian McCaffrey": {'proj': 30, 'projsd': 9},
    "CeeDee Lamb": {'proj': 29, 'projsd': 9},
    "Tyreek Hill": {'proj': 28, 'projsd': 9},
    "Ja'Marr Chase": {'proj': 27, 'projsd': 9},
    "Justin Jefferson": {'proj': 26, 'projsd': 9},
    "Amon-Ra St. Brown": {'proj': 25, 'projsd': 8},
    "Bijan Robinson": {'proj': 24, 'projsd': 8},
    "Breece Hall": {'proj': 23, 'projsd': 8},
    "A.J. Brown": {'proj': 22, 'projsd': 8},
    "Puka Nacua": {'proj': 21, 'projsd': 8},
    "Garrett Wilson": {'proj': 20, 'projsd': 7},
    "Jahmyr Gibbs": {'proj': 19, 'projsd': 7},
    "Marvin Harrison": {'proj': 18, 'projsd': 7},
    "Drake London": {'proj': 17, 'projsd': 7},
    "Jonathan Taylor": {'proj': 16, 'projsd': 7},
    "Nico Collins": {'proj': 15, 'projsd': 7},
    "Chris Olave": {'proj': 14, 'projsd': 6},
    "Deebo Samuel": {'proj': 13, 'projsd': 6},
    "Saquon Barkley": {'proj': 12, 'projsd': 6},
    "Jaylen Waddle": {'proj': 11, 'projsd': 6},
    "Davante Adams": {'proj': 10, 'projsd': 6},
    "Brandon Aiyuk": {'proj': 9, 'projsd': 6},
    "De'Von Achane": {'proj': 8, 'projsd': 5},
    "Mike Evans": {'proj': 7, 'projsd': 5},
    "DeVonta Smith": {'proj': 6, 'projsd': 5},
    "DK Metcalf": {'proj': 6, 'projsd': 5},
    "Malik Nabers": {'proj': 6, 'projsd': 4},
    "Cooper Kupp": {'proj': 6, 'projsd': 4},
    "Kyren Williams": {'proj': 6, 'projsd': 4},
    "Derrick Henry": {'proj': 6, 'projsd': 4},
    "DJ Moore": {'proj': 6, 'projsd': 3},
    "Stefon Diggs": {'proj': 6, 'projsd': 3},
    "Michael Pittman Jr.": {'proj': 6, 'projsd': 3},
    "Tank Dell": {'proj': 6, 'projsd': 3},
    "Sam LaPorta": {'proj': 6, 'projsd': 3},
    "Zay Flowers": {'proj': 6, 'projsd': 3},
    "Josh Allen": {'proj': 6, 'projsd': 3},
    "Travis Kelce": {'proj': 6, 'projsd': 3},
    "George Pickens": {'proj': 6, 'projsd': 3},
    "Isiah Pacheco": {'proj': 6, 'projsd': 3},
    "Amari Cooper": {'proj': 6, 'projsd': 3},
    "Jalen Hurts": {'proj': 6, 'projsd': 3},
    "Tee Higgins": {'proj': 6, 'projsd': 3},
    "Travis Etienne Jr.": {'proj': 6, 'projsd': 3},
    "Patrick Mahomes": {'proj': 6, 'projsd': 3},
    "Christian Kirk": {'proj': 6, 'projsd': 3},
    "Trey McBride": {'proj': 6, 'projsd': 3},
    "Lamar Jackson": {'proj': 6, 'projsd': 3},
    "Mark Andrews": {'proj': 6, 'projsd': 3},
    "Terry McLaurin": {'proj': 6, 'projsd': 3},
    "Dalton Kincaid": {'proj': 6, 'projsd': 3},
    "Josh Jacobs": {'proj': 6, 'projsd': 3},
    "Hollywood Brown": {'proj': 6, 'projsd': 3},
    "Keenan Allen": {'proj': 6, 'projsd': 3},
    "James Cook": {'proj': 6, 'projsd': 3},
    "Anthony Richardson": {'proj': 6, 'projsd': 3},
    "Jayden Reed": {'proj': 6, 'projsd': 3},
    "Calvin Ridley": {'proj': 6, 'projsd': 3},
    "Chris Godwin": {'proj': 6, 'projsd': 3},
    "Rashee Rice": {'proj': 6, 'projsd': 3},
    "Keon Coleman": {'proj': 6, 'projsd': 3},
    "Kyler Murray": {'proj': 6, 'projsd': 3},
    "Aaron Jones": {'proj': 6, 'projsd': 3},
    "DeAndre Hopkins": {'proj': 6, 'projsd': 3},
    "Rhamondre Stevenson": {'proj': 6, 'projsd': 3},
    "James Conner": {'proj': 6, 'projsd': 3},
    "Najee Harris": {'proj': 6, 'projsd': 3},
    "Jameson Williams": {'proj': 6, 'projsd': 3},
    "Jake Ferguson": {'proj': 6, 'projsd': 3},
    "Jordan Addison": {'proj': 6, 'projsd': 3},
    "Curtis Samuel": {'proj': 6, 'projsd': 3},
    "Jaylen Warren": {'proj': 6, 'projsd': 3},
    "Zamir White": {'proj': 6, 'projsd': 3},
    "Joe Burrow": {'proj': 6, 'projsd': 3},
    "Jonathon Brooks": {'proj': 6, 'projsd': 3},
    "D'Andre Swift": {'proj': 6, 'projsd': 3},
    "Raheem Mostert": {'proj': 6, 'projsd': 3},
    "Dak Prescott": {'proj': 6, 'projsd': 3},
    "Courtland Sutton": {'proj': 6, 'projsd': 3},
    "Brock Bowers": {'proj': 6, 'projsd': 3},
    "Jordan Love": {'proj': 6, 'projsd': 3},
    "Zack Moss": {'proj': 6, 'projsd': 3},
    "Joshua Palmer": {'proj': 6, 'projsd': 3},
    "David Njoku": {'proj': 6, 'projsd': 3},
    "Tony Pollard": {'proj': 6, 'projsd': 3},
    "Jayden Daniels": {'proj': 6, 'projsd': 3},
    "Brian Robinson Jr.": {'proj': 6, 'projsd': 3},
    "Romeo Doubs": {'proj': 6, 'projsd': 3},
    "Rashid Shaheed": {'proj': 6, 'projsd': 3},
    "Tyler Lockett": {'proj': 6, 'projsd': 3},
    "Tyjae Spears": {'proj': 6, 'projsd': 3},
    "Chase Brown": {'proj': 6, 'projsd': 3},
    "Devin Singletary": {'proj': 6, 'projsd': 3},
    "Khalil Shakir": {'proj': 6, 'projsd': 3},
    "Brock Purdy": {'proj': 6, 'projsd': 3},
    "Javonte Williams": {'proj': 6, 'projsd': 3},
    "Caleb Williams": {'proj': 6, 'projsd': 3},
    "Dontayvion Wicks": {'proj': 6, 'projsd': 3},
    "Brandin Cooks": {'proj': 6, 'projsd': 3},
    "Dallas Goedert": {'proj': 6, 'projsd': 3},
    "Trey Benson": {'proj': 6, 'projsd': 3},
    "Trevor Lawrence": {'proj': 6, 'projsd': 3},
    "Gus Edwards": {'proj': 6, 'projsd': 3},
    "Jakobi Meyers": {'proj': 6, 'projsd': 3},
    "Blake Corum": {'proj': 6, 'projsd': 3},
    "Ezekiel Elliott": {'proj': 6, 'projsd': 3},
    "Jerry Jeudy": {'proj': 6, 'projsd': 3},
    "Tua Tagovailoa": {'proj': 6, 'projsd': 3},
    "Jared Goff": {'proj': 6, 'projsd': 3},
    "Adonai Mitchell": {'proj': 6, 'projsd': 3},
    "Jerome Ford": {'proj': 6, 'projsd': 3},
    "Nick Chubb": {'proj': 6, 'projsd': 3},
    "Ja'Lynn Polk": {'proj': 6, 'projsd': 3},
    "Pat Freiermuth": {'proj': 6, 'projsd': 3},
    "Austin Ekeler": {'proj': 6, 'projsd': 3},
    "Dalton Schultz": {'proj': 6, 'projsd': 3}
}

    if uploaded_projections_file is not None:
        custom_projections_df = pd.read_csv(uploaded_projections_file)
        st.write("Custom Projections Data Preview:")
        st.dataframe(custom_projections_df.head())

        # Create a projection lookup dictionary from the custom projections
        projection_lookup = {}
        for _, row in custom_projections_df.iterrows():
            player_name = row['player_name']
            proj = row['proj']
            projsd = row.get('projsd', default_projections.get(player_name, {}).get('projsd', 6))  # Default projsd = 6 if not specified
            projection_lookup[player_name] = (proj, projsd)
    else:
        # Create a projection lookup dictionary from the default projections
        projection_lookup = {
            name: (default_projections[name]['proj'], default_projections[name]['projsd'])
            for name in default_projections
        }

    # Number of simulations for projection
    num_simulations = st.number_input("Number of simulations", min_value=1, value=1000)

    if st.button("Run Projection Simulation"):
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
