import streamlit as st
import numpy as np
import plotly.graph_objects as go
from front.helpers import *
from front.config import *

def display(project_budget, usage_history, models, weeks):
    st.title(" Forecast Simulation")
    st.markdown("Simulate future costs based on historical usage patterns and selected models.")

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
    weekly_budget = get_weekly_budget(project_budget)

    sim_weeks = st.slider("Simulation Weeks", 1, 12, 4, help="Number of weeks to simulate")
    sim_usage = [usage_history[week % weeks] for week in range(sim_weeks)]
    sim_costs = [
        sum(calculate_cost(latest_pricings[model], in_t / num_models, out_t / num_models) for model in valid_models)
        for in_t, out_t in sim_usage
    ]

    st.markdown("###  Simulated Cost Projection")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, sim_weeks + 1)),
        y=sim_costs,
        mode="lines+markers",
        name="Simulated Cost",
        line=dict(color=CHART_COLOR, width=3)
    ))
    fig.add_hline(
        y=weekly_budget,
        line_dash="dash",
        line_color=ACCENT_COLOR,
        annotation_text="Weekly Budget",
        annotation_position="top right"
    )
    fig.update_layout(
        title="Simulated Weekly Cost Projection",
        xaxis_title="Week",
        yaxis_title=f"Cost ({project_budget.currency})",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("###  A/B Budget Testing")
    with st.expander("Compare Budget Strategies"):
        col_a, col_b = st.columns(2)
        with col_a:
            budget_a = st.number_input(
                f"Budget A ({project_budget.currency}/week)",
                min_value=1.0,
                value=weekly_budget,
                key="budget_a",
                format="%.2f"
            )
        with col_b:
            budget_b = st.number_input(
                f"Budget B ({project_budget.currency}/week)",
                min_value=1.0,
                value=weekly_budget * 1.5,
                key="budget_b",
                format="%.2f"
            )

        if avg_pricing := create_average_pricing([latest_pricings[m] for m in valid_models]):
            allocated_usage = [(in_t / num_models, out_t / num_models) for in_t, out_t in sim_usage]
            metrics_a = compute_budget_metrics(budget_a, avg_pricing, allocated_usage)
            metrics_b = compute_budget_metrics(budget_b, avg_pricing, allocated_usage)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Budget A")
                st.metric("Daily Budget", f"{project_budget.currency} {metrics_a['daily_budget']:.2f}")
                st.metric("Max Tokens/Day", f"{metrics_a['max_tokens_per_day']:,}")
                st.metric("Avg Daily Cost", f"{project_budget.currency} {metrics_a['avg_daily_cost']:.2f}")
                st.metric("Days Covered", f"{metrics_a['days_covered']:.2f}")

            with col2:
                st.markdown("#### Budget B")
                st.metric("Daily Budget", f"{project_budget.currency} {metrics_b['daily_budget']:.2f}")
                st.metric("Max Tokens/Day", f"{metrics_b['max_tokens_per_day']:,}")
                st.metric("Avg Daily Cost", f"{project_budget.currency} {metrics_b['avg_daily_cost']:.2f}")
                st.metric("Days Covered", f"{metrics_b['days_covered']:.2f}")

            better_option = "A" if metrics_a["days_covered"] > metrics_b["days_covered"] else "B"
            st.success(f" **Budget {better_option}** is more sustainable based on simulated usage.")