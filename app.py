import streamlit as st

import streamlit as st
from st_paywall import add_auth

st.set_page_config(layout="wide")
st.title("🎈 Tyler's Subscription app POC 🎈")
st.balloons()

add_auth(
    required=True,
    login_button_text="Login with Google",
    login_button_color="#FD504D",
    login_sidebar=True,
)

st.write("Congrats, you are subscribed!")
st.write("the email of the user is " + str(st.session_state.email))

st.set_page_config(
    page_title="Multipage App",
)

st.title("Main Page")
st.sidebar.success("Select a page above.")

st.markdown("Instructions: Navigate to Draft tab and download the adp sheet. In there you will find the adp template. Download this and make changes if needed (specifically to adpsd). Upload into the simulator.")
st.markdown("Wait for the simulation to finish, (takes approx 2 mins per 1000 drafts). Make sure you download the results")
st.write("Navigate to the Projections page, upload your draft results. Upload your projections. Must be in the format of player_name, proj in column A and B of your csv")
