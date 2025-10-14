import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from front.helpers import get_usage_history_details

def display(session, user_email, project_budget, usage_history, models, weeks):
    """
    Affiche l'historique d'utilisation des modèles LLM,
    avec un tableau global et un graphique unique combinant
    tous les modèles,
    où pour chaque modèle on a une barre empilée Input/Cached/Output par jour.

    Args:
        session: SQLAlchemy session pour accéder à la BDD.
        user_email (str): Email de l'utilisateur.
        project_budget: Objet contenant info budget (devise, etc).
        usage_history: Données brutes d'historique (non utilisées directement).
        models (list): Liste des modèles disponibles.
        weeks (int): Nombre de semaines à afficher.
    """
    if not user_email:
        st.error("Please enter a valid email address to view usage history.")
        st.stop()

    st.title("Usage History")
    st.markdown("Detailed token usage and costs over recent weeks, combined for all models.")

    history_data, total_cost, model_data = get_usage_history_details(session, user_email, models, weeks)

    for log in history_data:
        try:
            dt = datetime.strptime(log["Date"], "%b %d, %Y %H:%M")
        except ValueError:
            try:
                dt = datetime.strptime(log["Date"], "%Y-%m-%d")
            except ValueError:
                st.warning(f"Unrecognized date format: {log['Date']}")
                continue
        log["Date"] = dt.strftime("%b %d, %Y")

    if not history_data:
        st.info("No usage history available yet.")
        return

    st.markdown(f"### Total Usage Cost: `{project_budget.currency} {total_cost:,.6f}`")  

    df = pd.DataFrame(history_data)

    numeric_cols = ["Input Tokens", "Cached Input", "Output Tokens", "Cost (USD)"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    dates = sorted(df["Date"].unique())
    models_list = sorted(df["Model"].unique())

    fig = go.Figure()

    for model_name in models_list:
        model_df = df[df["Model"] == model_name]

        daily_input = []
        daily_cached = []
        daily_output = []
        daily_cost = []

        for date in dates:
            day_df = model_df[model_df["Date"] == date]
            if not day_df.empty:
                daily_input.append(day_df["Input Tokens"].sum())
                daily_cached.append(day_df["Cached Input"].sum())
                daily_output.append(day_df["Output Tokens"].sum())
                daily_cost.append(day_df["Cost (USD)"].sum())
            else:
                daily_input.append(0)
                daily_cached.append(0)
                daily_output.append(0)
                daily_cost.append(0)

        fig.add_trace(go.Bar(
            x=dates,
            y=daily_input,
            name=f"{model_name} - Input",
            marker_color="#66BB6A",
            offsetgroup=model_name,
            legendgroup=model_name,
            customdata=daily_cost,
            hovertemplate=f"Model: {model_name}<br>Date: %{{x}}<br>Input Tokens: %{{y}}<br>Cost: $%{{customdata:.6f}}<extra></extra>"  # Show cost with 6 decimal places in hover
        ))

        fig.add_trace(go.Bar(
            x=dates,
            y=daily_cached,
            name=f"{model_name} - Cached",
            marker_color="#BDBDBD",
            offsetgroup=model_name,
            base=daily_input,
            legendgroup=model_name,
            showlegend=False,
            hovertemplate=f"Model: {model_name}<br>Date: %{{x}}<br>Cached Tokens: %{{y}}<extra></extra>"
        ))

        base_output = [i + c for i, c in zip(daily_input, daily_cached)]
        fig.add_trace(go.Bar(
            x=dates,
            y=daily_output,
            name=f"{model_name} - Output",
            marker_color="#42A5F5",
            offsetgroup=model_name,
            base=base_output,
            legendgroup=model_name,
            showlegend=False,
            hovertemplate=f"Model: {model_name}<br>Date: %{{x}}<br>Output Tokens: %{{y}}<extra></extra>"
        ))

    fig.update_layout(
        barmode="stack",
        title="Daily Token Usage per Model (Input + Cached + Output)",
        xaxis_title="Date",
        yaxis_title="Tokens",
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        xaxis=dict(tickangle=45)
    )

    st.subheader("Usage Logs")
    st.dataframe(
        df.style.format({
            "Input Tokens": "{:,.0f}",
            "Cached Input": "{:,.0f}",
            "Output Tokens": "{:,.0f}",
            "Cost (USD)": "${:,.6f}"  
        }),
        use_container_width=True,
        hide_index=True
    )

    st.plotly_chart(fig, use_container_width=True)