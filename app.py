from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "config" / "default.yaml"


def find_src_root() -> Path:
    search_bases = [APP_DIR, *APP_DIR.parents, Path.cwd(), *Path.cwd().parents]
    checked_paths: list[Path] = []
    seen: set[Path] = set()

    for base in search_bases:
        src_root = (base / "src").resolve()
        package_init = src_root / "energy_modeling" / "__init__.py"
        if src_root in seen:
            continue
        seen.add(src_root)
        checked_paths.append(src_root)
        if package_init.exists():
            return src_root

    searched = "\n".join(str(path) for path in checked_paths)
    raise ImportError(f"Could not find a valid src root containing energy_modeling.\nSearched:\n{searched}")


try:
    SRC_ROOT = find_src_root()
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    from energy_modeling.config import load_settings
    from energy_modeling.pipeline import run_training
except Exception as exc:
    st.set_page_config(page_title="Energy Forecasting System", layout="wide")
    st.error("Failed to load the local `energy_modeling` package from `src/`.")
    st.code(
        "\n".join(
            [
                f"APP_DIR={APP_DIR}",
                f"CWD={Path.cwd()}",
                f"SRC_ROOT={locals().get('SRC_ROOT', 'not found')}",
                f"sys.path[0:5]={sys.path[:5]}",
                f"{type(exc).__name__}={exc}",
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
