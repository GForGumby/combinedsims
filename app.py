import streamlit as st

st.set_page_config(
    page_title="Multipage App",
)

st.title("Main Page")
st.sidebar.success("Select a page above.")

st.text("Instructions: Navigate to Draft tab and download the adp sheet. In there you will find the adp template. Download this and make changes if needed (specifically to adpsd). Upload into the simulator.")
st.text("Wait for the simulation to finish, (takes approx 2 mins per 1000 drafts). Make sure you download the results")
st.text("Navigate to the Projections page, upload your draft results. Upload your projections. Must be in the format of player_name, proj in column A and B of your csv")
