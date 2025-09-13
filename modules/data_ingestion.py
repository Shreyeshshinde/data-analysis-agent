import pandas as pd
import streamlit as st

def load_csv(uploaded_file):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        # ✅ Removed unsupported "errors" argument
        df = pd.read_csv(uploaded_file)

        # Strip column names
        df.columns = df.columns.astype(str).str.strip()
        st.success("✅ Dataset loaded successfully.")
        return df

    except pd.errors.EmptyDataError:
        st.error("❌ The file is empty.")
        return None
    except pd.errors.ParserError:
        st.error("❌ Parsing error: Check if the file is a valid CSV.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None
