import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to sys.path to ensure energy_modeling is importable
sys.path.insert(0, os.path.abspath("src"))

from energy_modeling.config import load_settings
from energy_modeling.pipeline import run_training, load_and_prepare

st.set_page_config(page_title="Energy Forecasting System", layout="wide")

st.title("⚡ Energy Forecasting System")
st.markdown("""
Upload your energy consumption data (.xlsx) to train models and generate forecasts.
""")

# Sidebar settings
st.sidebar.header("Settings")
config_path = st.sidebar.text_input("Config Path", "config/default.yaml")

if not os.path.exists(config_path):
    st.error(f"Config file not found: {config_path}")
    st.stop()

settings = load_settings(config_path)

# File uploader
st.header("1. Data Upload")
uploaded_file = st.file_uploader("Choose a .xlsx file", type="xlsx")

if uploaded_file is not None:
    try:
        # Read the file into a dataframe
        # We need to know the sheet name from settings or let user choose
        data_cfg = settings.get("data")
        sheet_name = data_cfg.get("sheet_name", "Foglio1")
        
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")
        st.success("File uploaded successfully!")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Process data
        if st.button("Run Training and Forecast"):
            with st.spinner("Processing data and training models..."):
                try:
                    # Run the pipeline with the uploaded dataframe
                    results = run_training(settings, df=df)
                    
                    st.header("2. Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Model Metrics")
                        metrics_df = pd.read_csv(results["metrics"])
                        st.table(metrics_df)
                    
                    with col2:
                        st.subheader("Forecast Plot")
                        if os.path.exists(results["plot"]):
                            st.image(str(results["plot"]))
                        else:
                            st.warning("Plot not found.")
                            
                    st.subheader("Detailed Forecast Data")
                    forecast_df = pd.read_csv(results["forecast"])
                    st.dataframe(forecast_df)
                    
                    # Download buttons
                    st.download_button(
                        label="Download Forecast CSV",
                        data=forecast_df.to_csv(index=False),
                        file_name="forecast_results.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    st.exception(e)
                    
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.exception(e)
else:
    st.info("Please upload a .xlsx file to begin.")
