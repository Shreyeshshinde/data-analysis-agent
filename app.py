# app.py
import streamlit as st
import pandas as pd
import os
import shap
import pickle
import io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay

# Import your module functions
from modules.data_ingestion import load_csv
from modules.preprocessing import preprocess_data_streamlit
from modules.eda import perform_eda_streamlit
from modules.autolm_sklearn import automl_task_streamlit
from modules.visualization import feature_vs_target_streamlit

# Streamlit page setup
st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("🤖 AI-Powered Data Analysis Agent")

st.markdown(
    """
    Upload a CSV, choose the target column, and explore your dataset through preprocessing,
    EDA, AutoML training, and explainability visualizations.

    **Features Added:**
    - 🧹 Preprocessing with download option  
    - 📊 EDA and feature-target plots  
    - 🤖 AutoML with training performance and model download  
    - 🧠 Explainability using SHAP + Evaluation Plots (ROC, Confusion Matrix)
    """
)

# ---------------------- MAIN APP ----------------------
tab1, tab2, tab3, tab4 = st.tabs(["📂 Data & Preprocessing", "📊 EDA", "🤖 AutoML", "🧠 Explainability"])

# ---------------------- TAB 1: Data & Preprocessing ----------------------
with tab1:
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
    if uploaded_file:
        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        df = load_csv(uploaded_file)
        if df is None:
            st.error("Failed to load dataset.")
            st.stop()

        df.columns = df.columns.astype(str).str.strip()
        st.subheader("📊 Raw Dataset (first 10 rows)")
        st.dataframe(df.head(10))
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.subheader("🎯 Select Target Column")
        target = st.selectbox("Choose target column for prediction", df.columns)

        st.subheader("❗ Missing Values (Before Preprocessing)")
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("No missing values detected.")
        else:
            st.dataframe(missing)

        if st.button("⚙️ Run Preprocessing"):
            with st.spinner("Running preprocessing..."):
                X_processed, y, df_cleaned, scaler = preprocess_data_streamlit(df.copy(), target)

            st.success("✅ Preprocessing complete.")
            st.subheader("📋 Cleaned Dataset (first 10 rows)")
            st.dataframe(df_cleaned.head(10))
            st.write(f"Processed feature shape: {X_processed.shape}")

            st.download_button(
                label="⬇️ Download Preprocessed Dataset (CSV)",
                data=df_cleaned.to_csv(index=False).encode("utf-8"),
                file_name="preprocessed_data.csv",
                mime="text/csv"
            )

            st.session_state["X_processed"] = X_processed
            st.session_state["y"] = y
            st.session_state["df_cleaned"] = df_cleaned
            st.session_state["target"] = target

# ---------------------- TAB 2: EDA ----------------------
with tab2:
    if "df_cleaned" in st.session_state:
        st.header("🔎 Exploratory Data Analysis (EDA)")
        df_cleaned = st.session_state["df_cleaned"]
        target = st.session_state["target"]

        perform_eda_streamlit(df_cleaned, target)

        st.header("📈 Feature vs Target (All Features)")
        st.markdown("Scroll through the plots below.")
        feature_vs_target_streamlit(df_cleaned, target)
    else:
        st.warning("Please complete preprocessing first in Tab 1.")

# ---------------------- TAB 3: AutoML ----------------------
with tab3:
    if "X_processed" in st.session_state:
        X_processed = st.session_state["X_processed"]
        y = st.session_state["y"]

        st.header("🤖 AutoML — Model Comparison")
        with st.spinner("Training and comparing models..."):
            best_model, leaderboard_df, training_history = automl_task_streamlit(X_processed, y)

        if leaderboard_df is not None:
            st.subheader("🏁 Model Leaderboard")
            formatted_df = leaderboard_df.copy()
            for col in formatted_df.select_dtypes(include=["float", "float64", "int"]):
                formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
            st.dataframe(formatted_df)

            model_col = leaderboard_df.columns[0]
            metric_cols = leaderboard_df.columns[1:]
            try:
                st.bar_chart(leaderboard_df.set_index(model_col)[metric_cols])
                st.success(f"🏆 Best model: {leaderboard_df.iloc[0][model_col]}")
            except Exception as e:
                st.error(f"Could not plot model performance: {e}")

            model_path = os.path.join("models", "best_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    st.download_button("⬇️ Download Best Model", data=f, file_name="best_model.pkl")

            if training_history:
                st.subheader("📈 Model Training Performance (Cross-validation)")
                for model_name, details in training_history.items():
                    st.markdown(f"**{model_name}** — Params: `{details.get('params', {})}`")
                    if "cv_results" in details and "mean_test_score" in details["cv_results"]:
                        mean_scores = details["cv_results"]["mean_test_score"]
                        param_labels = [str(p) for p in details["cv_results"]["params"]]
                        history_df = pd.DataFrame({
                            "Params": param_labels,
                            "Mean CV Score": mean_scores
                        })
                        st.line_chart(history_df.set_index("Params"))
                    else:
                        st.write("No detailed CV results available for this model.")

            # Save for Explainability
            st.session_state["best_model"] = best_model
            st.session_state["X_processed"] = X_processed
            st.session_state["y"] = y
        else:
            st.error("❌ AutoML failed. See logs above.")
    else:
        st.warning("Please complete preprocessing first in Tab 1.")

# ---------------------- TAB 4: Explainability ----------------------
with tab4:
    if "best_model" in st.session_state and "X_processed" in st.session_state:
        st.header("🧠 Model Interpretability & Evaluation")

        best_model = st.session_state["best_model"]
        X_processed = st.session_state["X_processed"]
        y = st.session_state["y"]

        # Confusion Matrix + ROC Curve
        try:
            st.subheader("📊 Model Evaluation Plots")
            y_pred = best_model.predict(X_processed)

            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            st.pyplot(fig)

            if hasattr(best_model, "predict_proba"):
                y_prob = best_model.predict_proba(X_processed)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc = auc(fpr, tpr)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
                ax2.plot([0, 1], [0, 1], linestyle="--")
                ax2.legend()
                st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Evaluation plots unavailable: {e}")

        # SHAP Explainability
        st.subheader("🔍 SHAP Feature Importance")
        try:
            explainer = shap.Explainer(best_model, X_processed)
            shap_values = explainer(X_processed)

            st.write("### Feature Importance Summary")
            fig3, ax3 = plt.subplots()
            shap.summary_plot(shap_values, X_processed, plot_type="bar", show=False)
            st.pyplot(fig3)

            st.write("### SHAP Dependence Plot (Top Feature)")
            top_feature = X_processed.columns[0]
            fig4, ax4 = plt.subplots()
            shap.dependence_plot(top_feature, shap_values.values, X_processed, show=False)
            st.pyplot(fig4)
        except Exception as e:
            st.warning(f"SHAP failed: {e}")
    else:
        st.warning("Please train a model first in Tab 3.")
