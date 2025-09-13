# modules/output_layer.py
import streamlit as st

def output_dashboard_streamlit():
    st.header("📦 Output Summary")
    st.write("- Best model saved in `models/best_model.pkl`")
    st.write("- Report images (if any) are in `reports/`")
    st.info("You can extend this panel to serve predictions or add model explainability charts.")
