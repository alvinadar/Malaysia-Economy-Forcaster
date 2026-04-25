import ssl
import streamlit as st #Python library for creating web apps
import pandas as pd #Python library for data manipulation and analysis
from prophet import Prophet #Facebook's library for time series forecasting
import plotly.express as px #Python library for interactive data visualization 
from langchain_google_genai import ChatGoogleGenerativeAI #Langchain library for integrating Google Generative AI
from langchain_core.messages import HumanMessage #Langchain library for handling human messages

# --- CONFIGURATION ---
st.set_page_config(page_title="Malaysia Economic Forecaster", layout="wide",page_icon="🇲🇾")
# --- TITLE ---
st.title("🇲🇾 Malaysia Economic Forecaster")

#---Data Extraction---
def load_data():
    #1. CPI Data(Monthly)
    df_cpi = pd.read_parquet('https://storage.dosm.gov.my/cpi/cpi_2d_inflation.parquet')
    df_cpi['date'] = pd.to_datetime(df_cpi['date'])
    df_cpi = df_cpi[df_cpi['division'] == 'overall']

    #2. Fuel Price (Weekly Data)
    df_fuel = pd.read_parquet('https://storage.data.gov.my/commodities/fuelprice.parquet')
    df_fuel['date'] = pd.to_datetime(df_fuel['date'])
    if 'series_type' in df_fuel.columns:
        df_fuel = df_fuel[df_fuel['series_type'] == 'level']

    #3. Electricity Price (Monthly)
    df_elec = pd.read_parquet('https://storage.data.gov.my/energy/electricity_consumption.parquet')
    df_elec['date'] = pd.to_datetime(df_elec['date'])
    df_elec = df_elec[df_elec['sector'] == 'total']
    
    return df_cpi, df_fuel, df_elec

#---Error handling for data loading---
try:
    df_cpi,df_fuel,df_elec = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

#---Preprocessing  fuel data---
#Resample weekly fuel to mothly average 
df_fuel_m = df_fuel.set_index('date').resample("MS").mean(numeric_only=True).reset_index()

# Merge datasets
# Note: Using 'inner' join ensures we only use dates where ALL data points exist
master_df = df_cpi[['date', 'inflation_yoy']].merge(df_fuel_m[['date', 'ron95']], on='date', how='inner')
master_df = master_df.merge(df_elec[['date', 'consumption']], on='date', how='inner')

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
horizon = st.sidebar.slider("Forecast Horizon (Months)", 1, 12, 6)
google_api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
st.sidebar.markdown("[Get an API key here](https://aistudio.google.com/app/apikey)")
st.sidebar.divider()

# --- MODELING (Prophet) ---
# We wrap this in a try-block just in case of remaining data inconsistencies

try:
    m = Prophet(changepoint_prior_scale=0.05)
    m.add_regressor('fuel')
    m.add_regressor('electricity')
    m.fit(master_df)

    future = m.make_future_dataframe(periods = horizon,freq = 'MS')
    # Carry forward the last known values for regressors into the future
    future['fuel'] = master_df['fuel'].iloc[-1]
    future['electricity'] = master_df['electricity'].iloc[-1]
    forecast = m.predict(future)

    # --- VISUALIZATION ---
    fig = px.line(forecast, x='ds', y='yhat', title="Projected Inflation (YoY %)", 
                  labels={'yhat': 'Inflation (%)', 'ds': 'Date'})
    fig.add_scatter(x=master_df['ds'], y=master_df['y'], name="Historical Actual", mode='markers')
    st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Modeling Error: {e}")
