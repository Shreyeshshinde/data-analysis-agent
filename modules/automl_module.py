import os
import pandas as pd
import numpy as np
import streamlit as st
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, save_model as cls_save_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, save_model as reg_save_model

def automl_task(X, y):
    """
    Automatically detect task (Classification/Regression) and run PyCaret AutoML inside Streamlit.
    """

    # Ensure X is a DataFrame
    if isinstance(X, (np.ndarray, list)):
        X = pd.DataFrame(X)
    
    # Reset index of y and keep as Series
    y = y.reset_index(drop=True)

    # Merge features and target
    df = pd.concat([X, y], axis=1)

    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    try:
        # Detect Classification vs Regression
        if y.nunique() <= 20 and y.dtype in ['int64', 'object', 'category']:
            st.info("🔎 Detected **Classification Task**")
            cls_setup(data=df, target=y.name, session_id=123, verbose=False)
            best_model = cls_compare()
            cls_save_model(best_model, "models/best_model")
        else:
            st.info("🔎 Detected **Regression Task**")
            reg_setup(data=df, target=y.name, session_id=123, verbose=False)
            best_model = reg_compare()
            reg_save_model(best_model, "models/best_model")

        st.success("✅ Best model trained and saved!")
        st.write("🏆 **Selected Model:**", best_model)

        return best_model

    except Exception as e:
        st.error(f"❌ AutoML Error: {e}")
        return None
