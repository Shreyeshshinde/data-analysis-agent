import streamlit as st
import pandas as pd
from modules.data_ingestion import load_csv
from modules.eda import perform_eda
from modules.preprocessing import preprocess_data
from modules.automl_module import automl_task
from modules.visualization import generate_report

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")

st.title("🤖 AI-Powered Data Analysis Agent")
st.markdown("Upload your dataset and let the agent handle **ingestion, preprocessing, AutoML, and visualization** automatically.")

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # Show dataset preview
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Show dataset info
    st.subheader("📈 Dataset Summary")
    st.write(df.describe(include="all"))

    # Choose target column
    st.subheader("🎯 Select Target Column")
    target = st.selectbox("Select the target column for prediction", df.columns)

    if target:
        # Perform EDA
        st.header("🔎 Exploratory Data Analysis (EDA)")
        perform_eda(df)   # Make sure perform_eda uses st instead of print

        # Preprocessing
        X, y, scaler = preprocess_data(df, target)

        # AutoML Training
        st.header("🤖 AutoML Training")
        best_model = automl_task(X, y)

        # Visualization
        st.header("📊 Visual Report")
        generate_report(df, target)  # Ensure generate_report also uses st + matplotlib/plotly

        # Model summary
        st.success(f"✅ Best model trained: {type(best_model).__name__}")
