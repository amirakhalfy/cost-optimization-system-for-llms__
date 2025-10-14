import streamlit as st
from front.helpers import health_check

def display():
    """ Display the health check results. """
    st.markdown("###  Verify System Status")
    if health_check():
        st.success(" System is fully operational: Database connection and model data are available.")
    else:
        st.error(" System check failed: Issues detected with database or model data.")