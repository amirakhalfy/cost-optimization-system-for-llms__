import streamlit as st

PRIMARY_COLOR = "#1E88E5" 
SECONDARY_COLOR = "#4CAF50"  
TEXT_COLOR = "#263238" 
BACKGROUND_COLOR = "#F5F7FA"  
ACCENT_COLOR = "#E53935" 
CHART_COLOR = "#42A5F5"  
CARD_COLOR = "#FFFFFF"  
INPUT_TOKEN_COLOR = "#FF69B4" 
OUTPUT_TOKEN_COLOR = "#1E88E5"  

CSS = f"""
    <style>
    /* General Styling */
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Inter', sans-serif;
    }}
    .css-1d391kg {{ /* Sidebar */
        background-color: #FFFFFF;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: background-color 0.3s;
    }}
    .stButton>button:hover {{
        background-color: #1565C0;
    }}
    .stSelectbox, .stSlider, .stTextInput, .stTextArea {{
        background-color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        padding: 8px;
    }}
    .stMetric {{
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    /* Card Styling */
    .card {{
        background-color: {CARD_COLOR};
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }}
    .card-title {{
        font-size: 20px;
        font-weight: 600;
        color: {TEXT_COLOR};
        margin-bottom: 10px;
    }}
    /* Section Headers */
    h1, h2, h3 {{
        color: {TEXT_COLOR};
        font-weight: 600;
    }}
    /* Sidebar Menu */
    .css-1v3fvcr .css-1qrvfrg {{ /* Option Menu */
        background-color: #FFFFFF;
    }}
    .css-1v3fvcr .css-1qrvfrg .css-1v0m6n0 {{ /* Selected Item */
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
    /* Response Card */
    .response-card {{
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }}
    .response-title {{
        font-size: 18px;
        font-weight: 600;
        color: {TEXT_COLOR};
        margin-bottom: 15px;
    }}
    </style>
"""

def apply_css():
    st.markdown(CSS, unsafe_allow_html=True)