from pathlib import Path
import importlib.util
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "config" / "default.yaml"


def load_local_package_from_src() -> Path:
    candidate_package_dirs = [
        APP_DIR / "src" / "energy_modeling",
        Path.cwd() / "src" / "energy_modeling",
        APP_DIR.parent / "src" / "energy_modeling",
    ]
    package_dir = next((path for path in candidate_package_dirs if path.exists()), None)
    if package_dir is None:
        searched = "\n".join(str(path) for path in candidate_package_dirs)
        raise ImportError(f"Could not find local package directory.\nSearched:\n{searched}")

    if "energy_modeling" not in sys.modules:
        init_file = package_dir / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "energy_modeling",
            init_file,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {init_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["energy_modeling"] = module
        spec.loader.exec_module(module)

    return package_dir


try:
    PACKAGE_DIR = load_local_package_from_src()
    from energy_modeling.config import load_settings
    from energy_modeling.pipeline import run_training
except ImportError as exc:
    st.set_page_config(page_title="Energy Forecasting System", layout="wide")
    st.error("Failed to load the local `energy_modeling` package from `src/`.")
    st.code(
        "\n".join(
            [
                f"APP_DIR={APP_DIR}",
                f"CWD={Path.cwd()}",
                f"PACKAGE_DIR={locals().get('PACKAGE_DIR', 'not found')}",
                f"ImportError={exc}",
            ]
        )
    )
    st.stop()


def resolve_config_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path.resolve()


st.set_page_config(page_title="Energy Forecasting System", layout="wide")

st.title("Energy Forecasting System")
st.markdown(
    """
Upload your energy consumption data (`.xlsx`) to train models and generate forecasts.
This app is compatible with cloud hosting because it reads uploaded files directly
instead of requiring the workbook to exist on the server.
"""
)

st.sidebar.header("Settings")
config_path_input = st.sidebar.text_input("Config Path", str(DEFAULT_CONFIG_PATH.relative_to(APP_DIR)))
config_path = resolve_config_path(config_path_input)

if not config_path.exists():
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
        data_cfg = settings.get("data", default={}) or {}
        sheet_name = data_cfg.get("sheet_name", "Foglio1")
        
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine="openpyxl")
        st.success("File uploaded successfully!")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        if st.button("Run Training and Forecast"):
            with st.spinner("Processing data and training models..."):
                try:
                    results = run_training(settings, df=df, persist_artifacts=False)
                    metrics_df = results["metrics_df"]
                    forecast_df = results["forecast_df"]
                    
                    st.header("2. Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Model Metrics")
                        st.table(metrics_df)
                    
                    with col2:
                        st.subheader("Forecast Plot")
                        st.image(results["plot_bytes"])
                            
                    st.subheader("Detailed Forecast Data")
                    st.dataframe(forecast_df)
                    
                    st.download_button(
                        label="Download Forecast CSV",
                        data=forecast_df.to_csv(index=False),
                        file_name="forecast_results.csv",
                        mime="text/csv"
                    )
                    st.download_button(
                        label="Download Model Bundle",
                        data=results["model_bytes"],
                        file_name="trained_model.joblib",
                        mime="application/octet-stream"
                    )
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    st.exception(e)
                    
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.exception(e)
else:
    st.info("Please upload a .xlsx file to begin.")
