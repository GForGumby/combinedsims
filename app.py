import streamlit as st
import pandas as pd
import numpy as np
from numba import jit
from scipy.linalg import cholesky
from page1 import page1
from page2 import page2

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Page 1", "Page 2"])

# Display the selected page
if page == "Page 1":
    page1()
elif page == "Page 2":
    page2()
