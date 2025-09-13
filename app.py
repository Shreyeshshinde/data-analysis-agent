# app.py
import streamlit as st
import pandas as pd
import os
from modules.data_ingestion import load_csv
from modules.preprocessing import preprocess_data_streamlit
from modules.eda import perform_eda_streamlit
from modules.autolm_sklearn import automl_task_streamlit
from modules.visualization import generate_report_streamlit, feature_vs_target_streamlit
from modules.output_layer import output_dashboard_streamlit

st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("🤖 AI-Powered Data Analysis Agent")

st.markdown(
    """
    Upload a CSV, choose the target column, then run the pipeline.
    The app will show missing value info, preprocessing steps, EDA visualizations,
    feature vs target plots, model comparison (accuracy/R²) and let you download the best model.
    """
)

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
if uploaded_file:
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Load
    df = load_csv(uploaded_file)
    if df is None:
        st.error("Failed to load dataset.")
        st.stop()

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    st.subheader("📊 Raw Dataset (first 10 rows)")
    st.dataframe(df.head(10))
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Show columns and select target
    st.subheader("🎯 Select Target Column")
    target = st.selectbox("Choose target column for prediction", df.columns)

    # Show missing values
    st.subheader("❗ Missing Values (Before Preprocessing)")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing)

    # Option to run preprocessing
    if st.button("⚙️ Run Preprocessing & Show Updated Data"):
        with st.spinner("Running preprocessing..."):
            X_processed, y, df_cleaned, scaler = preprocess_data_streamlit(df.copy(), target)
        st.success("Preprocessing complete.")

        st.subheader("📋 Cleaned Dataset (first 10 rows)")
        st.dataframe(df_cleaned.head(10))
        st.write(f"Processed feature shape: {X_processed.shape}")

        # EDA
        st.header("🔎 Exploratory Data Analysis (EDA)")
        perform_eda_streamlit(df_cleaned, target)

        # Feature vs Target (all in streamlit)
        st.header("📈 Feature vs Target (All Features)")
        st.markdown("Scroll through the plots below.")
        feature_vs_target_streamlit(df_cleaned, target)

        # AutoML: train multiple models and show comparison
        st.header("🤖 AutoML — Model Comparison")
        with st.spinner("Training and comparing models..."):
            best_model, leaderboard_df = automl_task_streamlit(X_processed, y)

        if leaderboard_df is not None:
            st.subheader("Model leaderboard")

            # ✅ Format numeric columns for display (avoid Styler issues)
            formatted_df = leaderboard_df.copy()
            for col in formatted_df.select_dtypes(include=['float', 'float64', 'int']):
                formatted_df[col] = formatted_df[col].map(
                    lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
                )

            st.dataframe(formatted_df)

            st.subheader("Model Performance Chart")

            # ✅ Dynamically detect the model/algorithm column
            model_col = leaderboard_df.columns[0]  # assume first column = model name
            metric_cols = leaderboard_df.columns[1:]  # rest are numeric metrics

            try:
                st.bar_chart(leaderboard_df.set_index(model_col)[metric_cols])
                st.success(f"Best model: {leaderboard_df.iloc[0][model_col]}")
            except Exception as e:
                st.error(f"Could not plot model performance: {e}")

            # allow download of model
            model_path = os.path.join("models", "best_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    st.download_button("⬇️ Download Best Model", data=f, file_name="best_model.pkl")
        else:
            st.error("AutoML failed. See logs above.")
