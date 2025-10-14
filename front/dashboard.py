import streamlit as st
import numpy as np
import pandas as pd
from front.helpers import *

def display(session, project_budget, usage_history, models, weeks, show_token_estimation):
    """ Display the dashboard for cost optimization analysis. """
    alert_levels = create_alert_levels()

    chosen_models = st.multiselect(
        "Select Models for Analysis",
        models,
        format_func=lambda m: m.name,
        help="Choose models to include in cost projections"
    )
    if not chosen_models:
        st.warning("Please select at least one model to proceed.")
        return

    pricing_data = []
    latest_pricings = {}
    for model in chosen_models:
        if model.pricings:
            latest_pricing = max(model.pricings, key=lambda p: p.updated_at)
            latest_pricings[model] = latest_pricing
            pricing_data.append({
                "Model": model.name,
                "Input Cost": f"{latest_pricing.input_cost:.2f}",
                "Output Cost": f"{latest_pricing.output_cost:.2f}",
                "Token Unit": latest_pricing.token_unit,
                "Currency": latest_pricing.currency
            })
        else:
            st.warning(f"Model {model.name} has no pricing data.")
    valid_models = [m for m in chosen_models if m in latest_pricings]
    if not valid_models:
        st.error("No selected models have valid pricing data.")
        return

    num_models = len(valid_models)
    avg_pricing = create_average_pricing([latest_pricings[m] for m in valid_models])
    weekly_budget = get_weekly_budget(project_budget)

    if show_token_estimation and avg_pricing:
        token_est = tokens_for_budget([avg_pricing], weekly_budget)
        st.info(f" Estimated tokens per week within budget: **{token_est:,}** tokens")

    st.markdown("###  Budget Alert Levels")
    alert_cols = st.columns(4)
    for i, (threshold, info) in enumerate(alert_levels.items()):
        with alert_cols[i]:
            st.markdown(
                f"""
                <div class="card" style="text-align: center;">
                    <div style="font-size: 24px; margin-bottom: 5px;">{info['emoji']}</div>
                    <div style="font-weight: 600; color: {info['color']};">{info['level']}</div>
                    <div style="font-size: 12px; color: #78909C;">{threshold*100:.0f}% of Budget</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    if pricing_data:
        st.markdown("###  Model Pricing")
        st.dataframe(
            pd.DataFrame(pricing_data),
            use_container_width=True,
            hide_index=True
        )

    total_weekly_costs = [
        sum(calculate_cost(latest_pricings[model], in_t / num_models, out_t / num_models) for model in valid_models)
        for in_t, out_t in usage_history
    ]
    total_cost = sum(total_weekly_costs)
    st.markdown("###  Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Estimated Cost", f"{project_budget.currency} {total_cost:.2f}")
    with col2:
        st.metric("Avg Weekly Cost", f"{project_budget.currency} {np.mean(total_weekly_costs):.2f}")
    with col3:
        utilization = (total_cost / (weekly_budget * weeks) * 100) if weekly_budget > 0 else 0
        st.metric("Budget Utilization", f"{utilization:.2f}%")
    with col4:
        trend = ((total_weekly_costs[-1] - total_weekly_costs[0]) / total_weekly_costs[0] * 100) if total_weekly_costs[0] > 0 else 0
        st.metric("Cost Trend", f"{trend:+.2f}%")