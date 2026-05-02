import ssl
import streamlit as st #Python library for creating web apps
import pandas as pd #Python library for data manipulation and analysis
from prophet import Prophet #Facebook's library for time series forecasting
import plotly.graph_objects as go #Python library for creating interactive visualizations
import plotly.express as px #Python library for interactive data visualization 
from langchain_google_genai import ChatGoogleGenerativeAI #Langchain library for integrating Google Generative AI
from langchain_core.messages import HumanMessage #Langchain library for handling human messages

# --- CONFIGURATION ---
st.set_page_config(page_title="Malaysia Economic Forecaster", layout="wide",page_icon="🇲🇾")
# --- TITLE ---
st.title("🇲🇾 Malaysia Economic Forecaster")

#---Data Extraction---
@st.cache_data(ttl =3600,show_spinner="Fetching latest economic data from goverment portals...")
def load_data():
    #1. CPI Data(Monthly)
    df_cpi = pd.read_parquet('https://storage.dosm.gov.my/cpi/cpi_2d_inflation.parquet')
    df_cpi['date'] = pd.to_datetime(df_cpi['date'])
    df_cpi = df_cpi[df_cpi['division'] == 'overall'].copy()

    #2. Fuel Price (Weekly Data)
    df_fuel = pd.read_parquet('https://storage.data.gov.my/commodities/fuelprice.parquet')
    df_fuel['date'] = pd.to_datetime(df_fuel['date'])
    if 'series_type' in df_fuel.columns:
        df_fuel = df_fuel[df_fuel['series_type'] == 'level'].copy()

    #3. Electricity Price (Monthly)
    df_elec = pd.read_parquet('https://storage.data.gov.my/energy/electricity_consumption.parquet')
    df_elec['date'] = pd.to_datetime(df_elec['date'])
    df_elec = df_elec[df_elec['sector'] == 'total'].copy()
    
    return df_cpi, df_fuel, df_elec

#---Error handling for data loading---
try:
    df_cpi,df_fuel,df_elec = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

#---Preprocessing  fuel data---
#Resample weekly fuel to mothly average 
df_fuel_m =df_fuel.set_index('date').resample("MS").mean(numeric_only=True).reset_index()

# Merge datasets
# Note: Using 'inner' join ensures we only use dates where ALL data points exist
master_df = (
    df_cpi[["date", "inflation_yoy", "inflation_mom"]]
    .merge(df_fuel_m[["date", "ron95", "ron97", "diesel"]], on="date", how="inner")
    .merge(df_elec[["date", "consumption"]], on="date", how="inner")
)

#For prohet model we did not use our standard column for the model, we change the model based on 'ds', 'y' format
#Use the documentation for more details https://facebook.github.io/prophet/docs/quick_start.html#python-api
master_df = master_df.rename(columns={'date': 'ds', 'inflation_yoy': 'y', 'ron95': 'fuel', 'consumption': 'electricity'})

# --- THE FIX: HANDLE NaN VALUES ---
# Prophet will crash if NaNs are present. 
# We forward fill (carry last value) then backward fill (handle start of series).
master_df = master_df.ffill().bfill()#Do a self study on this method, it is a common method to handle missing data in time series

# Check if we have enough data after cleaning
if master_df.empty:
    st.error("The merged dataset is empty. Check if the date ranges of the 3 sources overlap.")
    st.stop()

st.sidebar.header("Forecast Settings")
horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 24, 6)
show_intervals = st.sidebar.checkbox("Show Confidence Intervals", value=True)
google_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
st.sidebar.markdown("[Get an API key here](https://aistudio.google.com/app/apikey)")
st.sidebar.divider()
st.sidebar.caption(f"Data covers **{master_df['ds'].min().strftime('%b %Y')}** – **{master_df['ds'].max().strftime('%b %Y')}**")
st.sidebar.caption(f"**{len(master_df)} monthly observations** after merging all sources.")


#KPI Summary Row 
latest = master_df.iloc[-1]#Get the latest data point for current value
year_ago = master_df[master_df["ds"] <= latest["ds"] - pd.DateOffset(months=12)].iloc[-1] \
    if len(master_df) > 12 else master_df.iloc[0]

col1, col2, col3, col4 = st.columns(4)#Create 4 columns for KPIs

with col1:
    delta_inf = latest["y"] - year_ago["y"]#The comparison of current inflation with 12 months ago
    st.metric(
        "Current Inflation (YoY)",
        f"{latest['y']:.1f}%",
        f"{delta_inf:+.1f} pp vs 12 months ago",
    )
with col2:
    avg_fuel = master_df["fuel"].mean()
    st.metric(
        "RON95 Fuel Price",
        f"RM {latest['fuel']:.2f} / litre",
        f"{latest['fuel'] - avg_fuel:+.2f} vs historical avg",
    )
with col3:
    delta_elec = ((latest["electricity"] - year_ago["electricity"]) / year_ago["electricity"]) * 100
    st.metric(
        "Electricity Demand",
        f"{latest['electricity']:,.0f} GWh",
        f"{delta_elec:+.1f}% YoY",
    )
with col4:
    mom_val = latest.get("inflation_mom", np.nan)
    st.metric(
        "Monthly Price Change",
        f"{mom_val:+.1f}%" if not np.isnan(mom_val) else "N/A",
        "Month-on-month (MoM)",
        delta_color="inverse",
    )

st.divider()
# --- MODELING (Prophet) ---
# We wrap this in a try-block just in case of remaining data inconsistencies

try:
    m = Prophet(
        changepoint_prior_scale=0.05,#Defines the sensitivity of the model.(0,01 is most sensitive, 0.1 is least sensitive)
        yearly_seasonality=True,#Normally will spike during festive seasons or there are a specific incident 
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.08)
    m.add_regressor('fuel')
    m.add_regressor('electricity')
    m.fit(master_df[['ds', 'y', 'fuel', 'electricity']])

    future = m.make_future_dataframe(periods = horizon,freq = 'MS')
    # Carry forward the last known values for regressors into the future
    future['fuel'] = master_df['fuel'].iloc[-1]
    future['electricity'] = master_df['electricity'].iloc[-1]
    forecast = m.predict(future)

except Exception as exc:
    st.error(f"Forecasting model error: {exc}")
    st.stop()

forecast_start = master_df["ds"].max()  
future_rows = forecast[forecast["ds"] > forecast_start]
pred_end = future_rows["yhat"].iloc[-1]
trend_word = "rise" if pred_end > latest["y"] else "fall"

# Risk label
if pred_end > 4.0:
    risk_icon, risk_label = "🔴", "High"
elif pred_end > 2.5:
    risk_icon, risk_label = "🟡", "Moderate"
else:
    risk_icon, risk_label = "🟢", "Low"


#Forcast Chart 
st.subheader("📈 Inflation Forecast")
st.caption(
    "Dots show actual historical inflation. The line is the model's projection. "
    "The shaded band shows the 80% range of likely outcomes."
)

fig_fc = go.Figure()

if show_intervals:
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% Confidence Range",
        hoverinfo="skip",
    ))

fig_fc.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    line=dict(color="#636EFA", width=2.5),
    name="Forecast",
))

fig_fc.add_trace(go.Scatter(
    x=master_df["ds"], y=master_df["y"],
    mode="markers",
    marker=dict(color="#EF553B", size=5),
    name="Actual Inflation",
))

fig_fc.add_vline(
    x=forecast_start.timestamp() * 1000,
    line_dash="dash",
    line_color="gray",
    annotation_text="  Forecast begins",
    annotation_position="top right",
)

fig_fc.update_layout(
    xaxis_title="Date",
    yaxis_title="Inflation Rate — Year-on-Year (%)",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.25),
    height=430,
    margin=dict(t=20),
)
st.plotly_chart(fig_fc, use_container_width=True)

# Plain-language risk summary
st.info(
    f"**What this means:** Over the next **{horizon} month(s)**, inflation is projected to "
    f"**{trend_word}** from **{latest['y']:.1f}%** to approximately **{pred_end:.1f}%** "
    f"(likely range: {future_rows['yhat_lower'].iloc[-1]:.1f}% – {future_rows['yhat_upper'].iloc[-1]:.1f}%). "
    f"Inflation outlook: {risk_icon} **{risk_label} Risk**"
)

st.divider()

# --- GEMINI INSIGHTS ---
st.subheader("🤖 Gemini Economic Analysis")

if google_api_key:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        
        curr_val = master_df['y'].iloc[-1]
        pred_val = forecast['yhat'].iloc[-1]
        
        prompt = f"""
        You are a Malaysian economic expert. 
        Current Inflation: {curr_val:.2f}%
        Predicted Inflation in {horizon} months: {pred_val:.2f}%
        
        Factors considered: RON95 fuel prices and industrial electricity consumption.
        Provide a professional summary of the outlook for Malaysian households.
        Include a mention of 'B40/M40' groups if relevant.
        """
        
        if st.button("Generate AI Insight"):
            with st.spinner("Analyzing..."):
                response = llm.invoke([HumanMessage(content=prompt)])
                st.info(response.content)
    except Exception as e:
        st.error(f"Gemini Error: {e}")
else:
    st.warning("Please enter your API key to enable AI analysis.")

# --- DATA PREVIEW ---
with st.expander("Explore Data Preview"):
    st.write("Last 5 months of merged data (Cleaned):")
    st.dataframe(master_df.tail())
