import streamlit as st

EMOJIS = {
    "guide": 128218,          
    "costs": 128178,         
    "alerts": 128202,         
    "trend": 128200,          
    "monte_carlo": 127922,    
    "var": 128201,            
    "models": 128193,         
    "tokens": 128283,         
    "tips": 128161,           
    "congrats": 127942,       
    "info": 8505,             
    "green": 129001,          
    "yellow": 9888,           
    "orange": 128317,         
    "red": 10060              
}

def display() -> None:
    """
    Displays the user guide for understanding cost analyses on the platform.

    This guide explains in simple, business-oriented terms:
    - Total, average, and trending costs.
    - Color-coded alerts for budget monitoring.
    - Forecasts and confidence intervals.
    - Monte Carlo simulations for anticipating budget overruns.
    - Risk indicators (Value at Risk and Expected Shortfall).
    - Individual model predictions and resource usage.
    - Token input/output distribution explained in simple terms.
    - Practical tips to optimize model usage and control costs.

    This function takes no parameters and returns nothing. It is designed to be
    called from the main Streamlit app to display the guide.
    """

    st.title(f"{chr(EMOJIS['guide'])} User Guide – Understanding Cost Analyses")
    st.markdown("""
    Welcome to this guide!  
    This page helps you understand all charts, alerts, and indicators on the platform,
    without diving into technical details.
    """)

    st.header(f"{chr(EMOJIS['costs'])} Understanding Your Costs")
    st.markdown("""
    - **Total cost**: amount spent during the analyzed period.  
    - **Average weekly cost**: typical weekly expenditure.  
    - **Weekly budget**: spending limit set for each week.  
    - **Budget usage**: percentage of your budget already consumed.  
    - **Cost trend**: whether your spending is increasing, decreasing, or stable over time.
    """)

    st.header(f"{chr(EMOJIS['alerts'])} Alerts and Color Indicators")
    st.markdown(f"""
    Alerts help you quickly see if your budget is respected:  
    - {chr(EMOJIS['green'])} **Green**: within limits.  
    - {chr(EMOJIS['yellow'])} **Yellow**: caution, monitor spending.  
    - {chr(EMOJIS['orange'])} **Orange**: alert, reduce usage.  
    - {chr(EMOJIS['red'])} **Red**: urgent, risk of overrun, act immediately.
    """)

    st.header(f"{chr(EMOJIS['trend'])} Trend Charts")
    st.markdown("""
    - **Historical curve**: shows past costs.  
    - **Future forecasts**: estimates costs for upcoming weeks.  
    - **Confidence interval (5%-95%)**: area indicating expected cost variability.
    """)

    st.header(f"{chr(EMOJIS['monte_carlo'])} Monte Carlo Simulations")
    st.markdown("""
    - The platform simulates **1000+ scenarios** to predict future costs.  
    - Helps visualize the **possible range of variation** rather than a single prediction.  
    - Useful to anticipate budget overruns under uncertainty.
    """)

    st.header(f"{chr(EMOJIS['var'])} Value at Risk (VaR) and Expected Shortfall")
    st.markdown("""
    - **VaR**: maximum amount you risk exceeding with a certain confidence (e.g., 95%).  
    - **VaR as % of budget**: proportion relative to your total budget.  
    - **Expected Shortfall**: average of overruns when the budget is exceeded, to assess potential risk.
    """)

    st.header(f"{chr(EMOJIS['models'])} Individual Models and Forecasts")
    st.markdown("""
    - Each model has its own cost forecast.  
    - Compare models to understand which usage patterns are most costly.  
    - Visualize **historical** and **predicted** costs for each model.
    """)

    st.header(f"{chr(EMOJIS['tokens'])} Token Distribution Explained")
    st.markdown("""
    - **Input tokens**: the text or data you send to the model (think of it as your request).  
    - **Output tokens**: the text or data the model generates in response (think of it as the answer).  
    - Helps identify which model consumes the most resources.  
    - Useful for optimizing model usage according to your budget.
    """)

    st.header(f"{chr(EMOJIS['tips'])} Practical Tips")
    st.markdown("""
    **To better manage costs**:  
    - Regularly monitor alerts and charts.  
    - Check which models consume the most tokens.  
    - Adjust usage before reaching orange or red zones.  

    **To anticipate risks**:  
    - Consult Monte Carlo simulations to visualize possible variability.  
    - Check VaR to know the amount not to exceed with a certain confidence.
    """)

    st.success(f"{chr(EMOJIS['congrats'])} You now know how to read and interpret all platform indicators!")
    st.info(f"{chr(EMOJIS['info'])}{chr(65039)} Need help? Contact your technical team for questions specific to your organization.")


if __name__ == "__main__":
    display()
