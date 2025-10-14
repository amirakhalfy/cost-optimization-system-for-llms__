import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from front import dashboard,guide, forecasting, health_check, llm_invocation, usage_history as usage_history_module
import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import uuid
from sqlalchemy import text
from sqlalchemy.orm import joinedload
from streamlit_option_menu import option_menu
import math
import requests

from app.db.models import Model, Pricing, Provider, ProjectBudget
from app.db.models.user_project_budget import UserProjectBudget
from app.db.models.model_usage_log import ModelUsageLog
from app.db.models.user import User
from app.db.db_setup import SessionLocal

from front.config import apply_css
from front.helpers import *
from front import simulation

st.set_page_config(
    page_title="LLM Budgeting Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_css()

with st.sidebar:
    selected_page = option_menu(
        menu_title="LLM Budgeting Platform",
        options=[
            "Dashboard", 
            "Forecasting", 
            "Usage History", 
            "LLM Invocation", 
            "Health Check",
            "Guide"
        ],
        icons=[
            "house", 
            "gear", 
            "graph-up-arrow", 
            "clock-history", 
            "robot", 
            "heart-pulse"
        ],
        menu_icon="calculator",
        default_index=0,
        styles={
            "container": {"padding": "10px", "background-color": "#FFFFFF"},
            "icon": {"color": "#263238", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "2px",
                "--hover-color": "#E3F2FD",
                "color": "#263238",
            },
            "nav-link-selected": {"background-color": "#1E88E5", "color": "white"},
            "menu-title": {"font-size": "18px", "font-weight": "600", "color": "#263238"},
        }
    )

    st.header("User Configuration", divider="blue")
    user_email = st.text_input("Email", value="", placeholder="Enter email for personalized data", help="Required only for LLM invocation and usage history")
    weeks = st.slider("Forecast Weeks", 4, 12, 8, help="Number of weeks for forecasting")
    use_demo_mode = st.checkbox("Demo Mode (No DB)", value=False, help="Use generated data instead of database")

    st.header("Simulation Parameters", divider="blue")
    growth_rate = st.slider("Weekly Growth Rate (%)", 0.0, 5.0, 2.0, step=0.5, help="Expected weekly usage growth") / 100
    variance = st.slider("Usage Variance (±%)", 5.0, 20.0, 10.0, step=2.5, help="Random variation in usage") / 100
    usage_multiplier = st.slider("Usage Multiplier", 0.5, 2.0, 1.0, step=0.1, help="Scale historical usage")
    var_confidence = st.slider("VaR Confidence Level", 0.80, 0.99, 0.95, step=0.01, format="%.2f", help="Confidence level for Value at Risk")
    show_token_estimation = st.checkbox("Show Token Estimate", value=True, help="Display estimated tokens within budget")

session = SessionLocal()
try:
    if user_email and not use_demo_mode:
        ensure_user_exists(session, user_email)
    project_budget = get_user_budget(session, user_email) if user_email and not use_demo_mode else create_default_budget()

    raw_usage_history = get_historical_usage(session, user_email, weeks) if user_email and not use_demo_mode else get_user_usage_with_seasonality("default@example.com", weeks, growth_rate, variance)
    processed_usage_history = [(int(in_t * usage_multiplier), int(out_t * usage_multiplier)) for in_t, out_t in raw_usage_history]

    models = get_models_with_pricing(session)
    if not models:
        st.error("No models with pricing found in the database.")
        session.close()
        st.stop()

    if selected_page == "Health Check":
        if health_check():
            st.success(" System is fully operational: Database connection and model data are available.")
    elif selected_page == "Dashboard":
        st.title(" LLM Budgeting Dashboard")
        st.markdown(
            """
            Welcome to your intelligent LLM cost management hub! This dashboard provides a comprehensive overview of your budget, usage, and cost projections with real-time insights and actionable recommendations.
            """
        )
        st.markdown("###  Budget Overview")
        col1, col2 = st.columns(2)
        weekly_budget = get_weekly_budget(project_budget)
        with col1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">Weekly Budget</div>
                    <p style="font-size: 24px; font-weight: bold; color: #4CAF50;">{project_budget.currency} {weekly_budget:.2f}</p>
                    <p style="color: #78909C;">Per Week</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">{project_budget.name}</div>
                    <p style="font-size: 24px; font-weight: bold; color: #E53935;">{project_budget.currency} {project_budget.amount:.2f}</p>
                    <p style="color: #78909C;">{project_budget.period.title()} Budget</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        dashboard.display(session, project_budget, processed_usage_history, models, weeks, show_token_estimation)
    elif selected_page == "Forecasting":
        forecasting.display(project_budget, processed_usage_history, models, weeks, var_confidence)
    elif selected_page == "Usage History":
        usage_history_module.display(session, user_email, project_budget, processed_usage_history, models, weeks)
    elif selected_page == "LLM Invocation":
        llm_invocation.display_llm_invocation(session, user_email, project_budget, processed_usage_history, models, weeks, use_demo_mode)
    elif selected_page == "Guide":
        guide.display()

finally:
    session.close()
