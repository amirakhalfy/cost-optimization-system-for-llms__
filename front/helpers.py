import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd
import math
import uuid
from sqlalchemy import text
from sqlalchemy.orm import joinedload
from app.db.models import (
    Model, Pricing, Provider, ProjectBudget,
    UserProjectBudget, ModelUsageLog, User
)
from app.db.db_setup import SessionLocal
import requests
   
def email_to_seed(email: str) -> int:
    "Convert email to a unique seed for random number generation."
    return abs(int(uuid.uuid5(uuid.NAMESPACE_DNS, email).int) % (2 ** 32))

def ensure_user_exists(session, user_email: str) -> bool:
    "Ensure the user and default budget exist in DB."
    if not user_email:
        return False
    try:
        user = session.query(User).filter_by(user_mail=user_email).first()
        if not user:
            user = User(user_mail=user_email)
            session.add(user)
            session.commit()

            default_budget = ProjectBudget(
                name="Default Project",
                amount=1000.0,
                alert_threshold=0.5,
                period="monthly",
                currency="USD",
                created_at=datetime.now()
            )
            session.add(default_budget)
            session.commit()

            user_proj = UserProjectBudget(
                user_mail=user_email,
                project_budget_id=default_budget.id
            )
            session.add(user_proj)
            session.commit()
        return True
    except Exception as e:
        session.rollback()
        st.error(f"Failed to create user or budget: {e}")
        return False

def get_user_budget(session, user_email: str) -> ProjectBudget:
    "Retrieve the user's budget or create a default one."
    if not user_email:
        return create_default_budget()
    try:
        user_proj = session.query(UserProjectBudget).filter_by(user_mail=user_email).first()
        if user_proj:
            project_budget = session.query(ProjectBudget).filter_by(id=user_proj.project_budget_id).first()
            if project_budget:
                project_budget.amount = float(project_budget.amount or 1000.0)
                project_budget.alert_threshold = float(project_budget.alert_threshold or 0.5)
                return project_budget
        ensure_user_exists(session, user_email)
        user_proj = session.query(UserProjectBudget).filter_by(user_mail=user_email).first()
        project_budget = session.query(ProjectBudget).filter_by(id=user_proj.project_budget_id).first()
        return project_budget
    except Exception as e:
        session.rollback()
        st.error(f"DB query failed: {e}")
    return create_default_budget()

def create_default_budget() -> ProjectBudget:
    "Create a default ProjectBudget instance."
    return ProjectBudget(
        name="Default Project",
        amount=1000.0,
        alert_threshold=0.5,
        period="monthly",
        currency="USD",
        created_at=datetime.now()
    )

def get_models_with_pricing(session):
    "Return list of Models with pricing and provider loaded."
    return (
        session.query(Model)
        .options(joinedload(Model.pricings), joinedload(Model.provider))
        .filter(Model.pricings.any())
        .all()
    )

def get_historical_usage(session, user_email: str, weeks: int):
    "Get usage logs for user over last N weeks or generate synthetic data."
    if not user_email:
        return get_user_usage_with_seasonality("default@example.com", weeks)
    try:
        usage_logs = (
            session.query(ModelUsageLog)
            .filter_by(user_mail=user_email)
            .order_by(ModelUsageLog.timestamp.desc())
            .limit(weeks * 7)
            .all()
        )
        weekly_usage = {}
        for log in usage_logs:
            if not log.timestamp:
                continue
            week_start = (log.timestamp - pd.Timedelta(days=log.timestamp.weekday())).strftime("%Y-%W")
            if week_start not in weekly_usage:
                weekly_usage[week_start] = {"input_tokens": 0, "output_tokens": 0}
            weekly_usage[week_start]["input_tokens"] += log.input_tokens or 0
            weekly_usage[week_start]["output_tokens"] += log.output_tokens or 0

        sorted_weeks = sorted(weekly_usage.keys(), reverse=True)[:weeks]
        usage = [(weekly_usage[w]["input_tokens"], weekly_usage[w]["output_tokens"]) for w in sorted_weeks]

        if len(usage) < weeks:
            usage.extend(get_user_usage_with_seasonality(user_email, weeks - len(usage)))
        return usage
    except Exception:
        return get_user_usage_with_seasonality(user_email, weeks)

def get_user_usage_with_seasonality(user_email: str, weeks: int, growth_rate=0.02, variance=0.1):
    "Generate synthetic weekly usage data with seasonality and growth."
    seed = email_to_seed(user_email)
    np.random.seed(seed)
    base_input = 5_000_000
    base_output = 2_500_000
    usage = []
    for i in range(weeks):
        seasonal = 1 + 0.2 * np.sin(2 * np.pi * i / 7)
        growth = (1 + growth_rate) ** i
        noise = np.random.uniform(1 - variance, 1 + variance)
        input_tokens = int(base_input * growth * seasonal * noise)
        output_tokens = int(base_output * growth * seasonal * noise)
        usage.append((input_tokens, output_tokens))
    return usage

def parse_token_unit(token_unit_str):
    "Always interpret token_unit as 1,000,000 tokens."
    return 1_000_000

def calculate_cost(pricing, input_tokens, output_tokens, use_cache=False) -> float:
    """
    Calculate the total cost given tokens and pricing.
    Uses cached_input cost if use_cache=True and cached_input price available.
    If pricing values are 0, they are used as is.
    Raises an error if pricing information is missing.
    """
    if not pricing:
        raise ValueError("Pricing information is missing.")

    token_unit = parse_token_unit(pricing.token_unit)
    if token_unit <= 0:
        token_unit = 1_000_000

    input_tokens = max(0, int(input_tokens or 0))
    output_tokens = max(0, int(output_tokens or 0))

    if pricing.input_cost is None or pricing.output_cost is None:
        raise ValueError("Input or output cost is missing in pricing.")

    input_cost = float(pricing.input_cost) 
    cached_input_cost = float(pricing.cached_input) if pricing.cached_input is not None else input_cost
    output_cost = float(pricing.output_cost) 

    input_cost_per_million = cached_input_cost if use_cache else input_cost
    output_cost_per_million = output_cost

    cost = (input_tokens / token_unit) * input_cost_per_million + (output_tokens / token_unit) * output_cost_per_million
    return round(cost, 6)

def calculate_tokens(text: str) -> int:
    "Approximate tokens from text length (assuming 4 chars/token)."
    return math.ceil(len(text) / 4)

def simulate_model_inference(prompt: str, max_tokens: int) -> (str, int, int):
    "Simulate model inference response."
    response = f"Simulated response for prompt: {prompt[:50]}... This is a simulated answer."
    input_tokens = calculate_tokens(prompt)
    output_tokens = min(calculate_tokens(response), max_tokens)
    if calculate_tokens(response) > max_tokens:
        response = response[:int(max_tokens * 4)]
    return response, input_tokens, output_tokens

def invoke_openai(prompt: str, max_tokens: int, api_key: str, model_name: str):
    "Invoke OpenAI API to get model completion."
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["text"], data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"]
    else:
        raise Exception(f"OpenAI API error: {response.text}")

def invoke_deepseek(prompt: str, max_tokens: int, api_key: str, model_name: str):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    response = requests.post("https://vertebrate.api.deepseek.com/v1/completions", headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["choices"][0]["text"], data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"]
    else:
        raise Exception(f"DeepSeek API error: {response.text}")

def monte_carlo_budget_overrun_enhanced(total_future_costs, total_budget, alert_thresholds):
    """monte_carlo_budget_overrun_enhanced calculates the probability of budget overruns
    along with detailed statistics for each threshold.
    """
    results = {}
    for threshold in alert_thresholds:
        prob_overrun = np.mean(total_future_costs > (total_budget * threshold))
        results[threshold] = prob_overrun
    return results

def calculate_value_at_risk(cost_samples, confidence_level):
    """Calculate the Value at Risk (VaR) for a given set of cost samples.

    Value at Risk is a statistical technique used to measure the risk of loss 
    on a portfolio. It estimates the maximum potential loss over a given 
    time period at a specified confidence level.

    Parameters
    ----------
    cost_samples : array-like
        A list or NumPy array of cost values (e.g., simulated or historical costs).
    confidence_level : float
        The confidence level for the VaR calculation, expressed as a decimal 
        between 0 and 1 (e.g., 0.95 for 95% confidence).

    Returns
    -------
    float
        The Value at Risk at the specified confidence level."""
    return np.percentile(cost_samples, 100 * confidence_level)

def get_weekly_budget(project_budget):
    """Get the weekly budget amount from a ProjectBudget instance."""
    if not project_budget:
        return 0.0
    amount = float(project_budget.amount or 0)
    period = getattr(project_budget, "period", "weekly").lower()
    if period == "monthly":
        return amount / 4.333
    if period == "daily":
        return amount * 7.0
    return amount

def validate_projections(historical_costs, future_costs, weekly_budget):
    """ Validate future cost projections against historical data and budget constraints.

    This function compares the average projected future costs to historical averages 
    and budget limits to detect any major deviations. It generates warnings if:
    - Future costs are significantly higher than historical trends.
    - Historical cost growth appears unsustainable.
    - Future costs exceed a multiple of the defined weekly budget.
  """
    warnings = []
    avg_historical = np.mean(historical_costs)
    avg_future = np.mean(future_costs)
    if avg_future > avg_historical * 3:
        warnings.append("Future costs are projected to be 3x higher than historical average")
    if len(historical_costs) > 1:
        growth_rate = (historical_costs[-1] - historical_costs[0]) / historical_costs[0]
        if growth_rate > 2.0:
            warnings.append("Historical growth rate appears unsustainable")
    if avg_future > weekly_budget * 5:
        warnings.append("Projected costs are 5x over weekly budget - consider budget revision")
    return warnings

def tokens_for_budget(pricings, budget):
    """Calculate the maximum number of tokens that can be used given a budget and pricing."""
    if not pricings:
        return 0
    total_avg_cost = 0
    for pricing in pricings:
        token_unit = parse_token_unit(pricing.token_unit)
        input_cost = float(pricing.input_cost)
        output_cost = float(pricing.output_cost)
        avg_cost_per_token = (input_cost + output_cost) / 2 / token_unit
        total_avg_cost += avg_cost_per_token
    avg_cost_per_token = total_avg_cost / len(pricings) if pricings else 0
    if avg_cost_per_token <= 0:
        return 0
    return int(budget / avg_cost_per_token)

def health_check():
    """Perform a health check to ensure the database connection is working and models are available."""
    try:
        session = SessionLocal()
        session.execute(text("SELECT 1"))
        models = get_models_with_pricing(session)
        session.close()
        return len(models) > 0
    except Exception:
        return False

def compute_budget_metrics(budget_value, pricing, usage_history):
    """  Compute key budget-related metrics based on pricing and usage history.

    This function calculates several metrics that help assess how a given weekly budget
    aligns with current usage patterns and pricing:
    - Daily budget
    - Maximum tokens that can be used per day given the pricing
    - Average daily cost based on usage history
    - Estimated number of days the budget will cover at current usage rate"""
    daily_budget = budget_value / 7
    max_tokens_per_day = tokens_for_budget([pricing], daily_budget)
    avg_daily_cost = np.mean([calculate_cost(pricing, in_t, out_t) for (in_t, out_t) in usage_history]) / 7
    days_covered = budget_value / (avg_daily_cost * 7) if avg_daily_cost > 0 else 0
    return {
        "weekly_budget": budget_value,
        "daily_budget": daily_budget,
        "max_tokens_per_day": max_tokens_per_day,
        "avg_daily_cost": avg_daily_cost,
        "days_covered": days_covered
    }

def create_alert_levels():
    return {
        0.25: {"color": "#4CAF50", "level": "Low", "emoji": "🟢"},
        0.50: {"color": "#FFCA28", "level": "Medium", "emoji": "🟡"},
        0.75: {"color": "#FB8C00", "level": "High", "emoji": "🟠"},
        0.90: {"color": "#E53935", "level": "Critical", "emoji": "🔴"}
    }

def create_average_pricing(pricings):
    if not pricings:
        return None
    input_costs = [float(p.input_cost) for p in pricings if p.input_cost is not None]
    output_costs = [float(p.output_cost) for p in pricings if p.output_cost is not None]
    token_units = [p.token_unit for p in pricings]
    currencies = [p.currency for p in pricings]
    if len(set(token_units)) > 1 or len(set(currencies)) > 1:
        st.warning("Selected models have different token units or currencies. Using first model's units.")
        token_unit = pricings[0].token_unit
        currency = pricings[0].currency
    else:
        token_unit = token_units[0]
        currency = currencies[0]
    avg_input_cost = np.mean(input_costs) if input_costs else None
    avg_output_cost = np.mean(output_costs) if output_costs else None
    if avg_input_cost is None or avg_output_cost is None:
        raise ValueError("Cannot create average pricing due to missing input or output costs.")
    return Pricing(input_cost=avg_input_cost, output_cost=avg_output_cost, token_unit=token_unit, currency=currency)

def get_usage_history_details(session, user_email, models, weeks):
    """
    Retrieve detailed usage logs including costs and cached input info.
    """
    if not user_email:
        return [], 0, {m.name: default_model_dict() for m in models}

    try:
        usage_logs = (
            session.query(ModelUsageLog)
            .filter_by(user_mail=user_email)
            .order_by(ModelUsageLog.timestamp.desc())
            .limit(weeks * 7)
            .all()
        )

        model_pricings = {
            m.name: max(m.pricings, key=lambda p: p.updated_at)
            for m in models if m.pricings
        }

        history_data, total_cost = [], 0
        model_data = {m.name: default_model_dict() for m in models}

        for log in usage_logs:
            pricing = model_pricings.get(log.model_name)
            use_cache = (log.cached_input or 0) > 0
            try:
                cost = log.cost if log.cost is not None else calculate_cost(pricing, log.input_tokens, log.output_tokens, use_cache=use_cache)
            except ValueError as e:
                st.error(f"Error calculating cost for log {log.id}: {e}")
                cost = 0  
            total_cost += cost

            model_data[log.model_name]['dates'].append(log.timestamp)
            model_data[log.model_name]['costs'].append(cost)
            model_data[log.model_name]['input_tokens'].append(log.input_tokens or 0)
            model_data[log.model_name]['cached_inputs'].append(log.cached_input or 0)
            model_data[log.model_name]['output_tokens'].append(log.output_tokens or 0)

            history_data.append({
                "Date": log.timestamp.strftime('%Y-%m-%d'),
                "Model": log.model_name,
                "Input Tokens": log.input_tokens,
                "Cached Input": log.cached_input or 0,
                "Output Tokens": log.output_tokens,
                "Cost (USD)": cost,
            })

        return history_data, total_cost, model_data

    except Exception as e:
        st.error(f"Failed to retrieve usage history: {e}")
        return [], 0, {m.name: default_model_dict() for m in models}

def default_model_dict():
    """Helper to create empty model data dictionary."""
    return {'dates': [], 'costs': [], 'input_tokens': [], 'cached_inputs': [], 'output_tokens': []}