import streamlit as st
import streamlit as st
from authlib.integrations.requests_client import OAuth2Session
from authlib.integrations.base_client import OAuthError
import os

# Load your Google OAuth credentials from the JSON file
GOOGLE_CLIENT_ID = os.getenv('405134594415-8a9l8kliu9v2ineuhg6i6gadc8qjgr3a.apps.googleusercontent.com')
GOOGLE_CLIENT_SECRET = os.getenv('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')

# Redirect URI should match the one set in the Google Cloud Console
REDIRECT_URI = 'https://gumbysims1.streamlit.app/'

# OAuth 2.0 endpoints for Google
AUTHORIZATION_URL = 'https://accounts.google.com/o/oauth2/auth'
TOKEN_URL = 'https://accounts.google.com/o/oauth2/token'
USERINFO_URL = 'https://www.googleapis.com/oauth2/v1/userinfo'

# Initialize OAuth session
oauth = OAuth2Session(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
                      redirect_uri=REDIRECT_URI, scope='openid email profile')

if 'token' not in st.session_state:
    # Redirect to Google's OAuth 2.0 server
    authorization_url, state = oauth.authorization_url(AUTHORIZATION_URL,
                                                       access_type='offline',
                                                       prompt='select_account')

    st.write('Click [here](%s) to log in with your Google account.' % authorization_url)
else:
    # Fetch user information
    token = st.session_state['token']
    try:
        oauth.token = token
        userinfo = oauth.get(USERINFO_URL).json()
        st.write(f"Hello, {userinfo['name']}! You are logged in.")
    except OAuthError as e:
        st.error("OAuth authentication failed. Please try again.")

# Handle the callback from Google's OAuth 2.0 server
if 'code' in st.experimental_get_query_params():
    try:
        token = oauth.fetch_token(TOKEN_URL, authorization_response=st.experimental_get_query_params())
        st.session_state['token'] = token
        st.experimental_rerun()
    except OAuthError as e:
        st.error("OAuth authentication failed. Please try again.")

st.set_page_config(
    page_title="Multipage App",
)
st.title("Main Page")
st.sidebar.success("Select a page above.")

st.markdown("Instructions: Navigate to Draft tab and download the adp sheet. In there you will find the adp template. Download this and make changes if needed (specifically to adpsd). Upload into the simulator.")
st.markdown("Wait for the simulation to finish, (takes approx 2 mins per 1000 drafts). Make sure you download the results")
st.write("Navigate to the Projections page, upload your draft results. Upload your projections. Must be in the format of player_name, proj in column A and B of your csv")

st.write("google-site-verification content=45GIE4pR1aI7CMZtcceMzubFU42Y3jwBP5gFyAIaiuc")

