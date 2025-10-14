import streamlit as st
import numpy as np
import plotly.graph_objects as go
from front.helpers import *
from front.config import *

def display(project_budget, usage_history, models, weeks, var_confidence):
    st.title(" Cost Forecasting")
    st.markdown(
        """
        Forecast future costs with enhanced seasonality, reduced variance, and multi-level risk analysis.
        Includes Monte Carlo simulations and Value at Risk (VaR) for robust budget planning.
        """
    )

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

    total_weekly_costs = [
        sum(calculate_cost(latest_pricings[model], in_t / num_models, out_t / num_models) for model in valid_models)
        for in_t, out_t in usage_history
    ]
    historical_weeks = np.arange(1, weeks + 1)

    slope_in, intercept_in = np.polyfit(historical_weeks, np.array([in_t for in_t, _ in usage_history]), 1)
    slope_out, intercept_out = np.polyfit(historical_weeks, np.array([out_t for _, out_t in usage_history]), 1)
    future_weeks = np.arange(weeks + 1, 2 * weeks + 1)
    pred_total_input = slope_in * future_weeks + intercept_in
    pred_total_output = slope_out * future_weeks + intercept_out

    simulations = 1000
    reduced_variance = 0.1
    future_total_input_samples = np.array([
        pred_total_input * np.random.uniform(1 - reduced_variance, 1 + reduced_variance, size=weeks)
        for _ in range(simulations)
    ])
    future_total_output_samples = np.array([
        pred_total_output * np.random.uniform(1 - reduced_variance, 1 + reduced_variance, size=weeks)
        for _ in range(simulations)
    ])

    future_total_weekly_costs = np.zeros((simulations, weeks))
    for sim in range(simulations):
        for w in range(weeks):
            total_in = future_total_input_samples[sim, w]
            total_out = future_total_output_samples[sim, w]
            allocated_in = total_in / num_models
            allocated_out = total_out / num_models
            week_cost = sum(
                calculate_cost(latest_pricings[model], allocated_in, allocated_out)
                for model in valid_models
            )
            future_total_weekly_costs[sim, w] = week_cost

    mean_forecast = np.mean(future_total_weekly_costs, axis=0)
    lower_ci = np.percentile(future_total_weekly_costs, 5, axis=0)
    upper_ci = np.percentile(future_total_weekly_costs, 95, axis=0)

    warnings = validate_projections(total_weekly_costs, mean_forecast, weekly_budget)
    if warnings:
        st.markdown("###  Projection Warnings")
        for warning in warnings:
            st.warning(warning)

    st.markdown("###  Cost Forecast")
    forecast_chart = go.Figure()
    forecast_chart.add_trace(go.Scatter(
        x=historical_weeks,
        y=total_weekly_costs,
        mode="lines+markers",
        name="Historical Cost",
        line=dict(color=PRIMARY_COLOR, width=3)
    ))
    forecast_chart.add_trace(go.Scatter(
        x=future_weeks,
        y=mean_forecast,
        mode="lines+markers",
        name="Predicted Cost (Mean)",
        line=dict(color=SECONDARY_COLOR, dash="dash", width=3)
    ))
    forecast_chart.add_trace(go.Scatter(
        x=future_weeks,
        y=upper_ci,
        mode="lines",
        name="95% CI",
        line=dict(color=SECONDARY_COLOR, width=0),
        showlegend=False
    ))
    forecast_chart.add_trace(go.Scatter(
        x=future_weeks,
        y=lower_ci,
        mode="lines",
        name="5% CI",
        line=dict(color=SECONDARY_COLOR, width=0),
        fill="tonexty",
        fillcolor="rgba(76,175,80,0.2)",
        showlegend=False
    ))
    for threshold, info in create_alert_levels().items():
        threshold_value = weekly_budget * threshold
        forecast_chart.add_trace(go.Scatter(
            x=[1, 2 * weeks + 1],
            y=[threshold_value] * 2,
            mode="lines",
            name=f"{info['level']} Alert ({threshold*100:.0f}%)",
            line=dict(color=info['color'], dash="dot", width=2)
        ))
    forecast_chart.add_vline(
        x=weeks,
        line_color="gray",
        line_dash="dot",
        annotation_text="Forecast Start",
        annotation_position="top left"
    )
    forecast_chart.update_layout(
        title="Weekly Cost Forecast with Alert Levels",
        xaxis_title="Week",
        yaxis_title=f"Cost ({project_budget.currency})",
        template="plotly_white",
        height=600
    )
    st.plotly_chart(forecast_chart, use_container_width=True)

    st.markdown("###  Monte Carlo Risk Analysis")
    total_future_costs = np.sum(future_total_weekly_costs, axis=1)
    total_budget = weekly_budget * weeks
    overrun_probs = monte_carlo_budget_overrun_enhanced(total_future_costs, total_budget, list(create_alert_levels().keys()))

    risk_cols = st.columns(4)
    for i, (threshold, prob) in enumerate(overrun_probs.items()):
        info = create_alert_levels()[threshold]
        with risk_cols[i]:
            st.markdown(
                f"""
                <div class="card" style="text-align: center;">
                    <div style="font-size: 24px; margin-bottom: 5px;">{info['emoji']}</div>
                    <div style="font-weight: 600; color: {info['color']};">{prob * 100:.2f}%</div>
                    <div style="font-size: 12px; color: gray;">Risk at {threshold * 100:.0f}% Budget</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("###  Cost Distribution")
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=total_future_costs,
        nbinsx=50,
        name="Simulated Total Costs",
        marker_color=CHART_COLOR,
        opacity=0.7
    ))
    for threshold, info in create_alert_levels().items():
        threshold_value = total_budget * threshold
        hist_fig.add_vline(
            x=threshold_value,
            line_color=info['color'],
            line_dash="dash",
            annotation_text=f"{info['level']}",
            annotation_position="top right"
        )
    hist_fig.update_layout(
        title="Distribution of Simulated Total Costs",
        xaxis_title=f"Total Cost ({project_budget.currency})",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    st.markdown("###  Value at Risk (VaR)")
    var_value = calculate_value_at_risk(total_future_costs, var_confidence)
    expected_shortfall = np.mean(total_future_costs[total_future_costs > var_value])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"{int(var_confidence * 100)}% VaR", f"{project_budget.currency} {var_value:.2f}")
    with col2:
        budget_ratio = var_value / total_budget if total_budget > 0 else 0
        st.metric("VaR as % of Budget", f"{budget_ratio * 100:.2f}%")
    with col3:
        st.metric("Expected Shortfall", f"{project_budget.currency} {expected_shortfall:.2f}")

    st.markdown("###  Model-Specific Forecasts")
    individual_forecasts = {}
    for model in valid_models:
        model_weekly_costs = [
            calculate_cost(latest_pricings[model], in_t / num_models, out_t / num_models)
            for in_t, out_t in usage_history
        ]
        slope_model, intercept_model = np.polyfit(historical_weeks, model_weekly_costs, 1)
        pred_model_cost = slope_model * future_weeks + intercept_model
        individual_forecasts[model] = pred_model_cost

    fig_individual = go.Figure()
    colors = ['#FF6347', '#4682B4', '#FFD700', '#32CD32', '#DA70D6']
    for i, model in enumerate(valid_models):
        color = colors[i % len(colors)]
        model_historical_costs = [
            calculate_cost(latest_pricings[model], in_t / num_models, out_t / num_models)
            for in_t, out_t in usage_history
        ]
        fig_individual.add_trace(go.Scatter(
            x=historical_weeks,
            y=model_historical_costs,
            mode="lines+markers",
            name=f"{model.name} (Historical)",
            line=dict(color=color, width=2)
        ))
        fig_individual.add_trace(go.Scatter(
            x=future_weeks,
            y=individual_forecasts[model],
            mode="lines+markers",
            name=f"{model.name} (Predicted)",
            line=dict(color=color, dash="dash", width=2)
        ))
    fig_individual.add_trace(go.Scatter(
        x=[1, 2 * weeks],
        y=[weekly_budget] * 2,
        mode="lines",
        name="Weekly Budget",
        line=dict(color=ACCENT_COLOR, dash="dash", width=3)
    ))
    fig_individual.add_vline(
        x=weeks,
        line_color="gray",
        line_dash="dot",
        annotation_text="Forecast Start",
        annotation_position="top left"
    )
    fig_individual.update_layout(
        title="Individual Model Cost Forecasts",
        xaxis_title="Week",
        yaxis_title=f"Cost ({project_budget.currency})",
        template="plotly_white",
        height=600
    )
    st.plotly_chart(fig_individual, use_container_width=True)

    if len(valid_models) >= 1:
        st.markdown("###  Total Token Distribution")
        labels = []
        values = []
        total_input = sum(in_t for in_t, _ in usage_history) / weeks / num_models
        total_output = sum(out_t for _, out_t in usage_history) / weeks / num_models
        for model in valid_models:
            labels.append(f"{model.name} (Input)")
            labels.append(f"{model.name} (Output)")
            values.append(total_input)
            values.append(total_output)
        token_fig = go.Figure(data=[
            go.Pie(labels=labels, values=values, hole=0.3)
        ])
        token_fig.update_layout(
            title="Input and Output Token Distribution",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(token_fig, use_container_width=True)