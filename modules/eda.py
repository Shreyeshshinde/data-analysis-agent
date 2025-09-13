# modules/eda.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def perform_eda_streamlit(df, target_column):
    st.write("### Dataset Summary")
    st.write(df.describe(include="all").T)

    # Missing values
    st.write("### Missing values (post-preprocessing check)")
    ms = df.isnull().sum()
    if (ms > 0).any():
        st.dataframe(ms[ms > 0])
    else:
        st.success("✅ No missing values detected")

    # Correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        st.write("### Correlation heatmap (numeric features)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric features for correlation heatmap.")

    # Target distribution
    if target_column in df.columns:
        st.write(f"### Distribution of target: {target_column}")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        if df[target_column].dtype == "object" or str(df[target_column].dtype).startswith("category"):
            sns.countplot(x=df[target_column], ax=ax2)  # ✅ fixed
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        else:
            sns.histplot(df[target_column], kde=True, ax=ax2)
        st.pyplot(fig2)
    else:
        st.error(f"❌ Target column '{target_column}' not found in dataset")

    # Pairplot sample (avoid huge datasets)
    st.write("### Pairwise sample (first 6 numeric features)")
    numeric_cols = numeric_df.columns.tolist()[:6]
    if len(numeric_cols) > 1:
        try:
            sample = df[numeric_cols].sample(min(200, len(df)))
            pp = sns.pairplot(sample)
            st.pyplot(pp.fig)
            plt.close(pp.fig)
        except Exception:
            st.info("⚠️ Pairplot skipped due to size or plotting issues.")
    else:
        st.info("Not enough numeric columns for pairplot.")
