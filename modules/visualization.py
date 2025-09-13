import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def generate_report_streamlit(df, target_column):
    """
    Quick report: correlation heatmap + target distribution
    """
    st.subheader("📊 Quick Report")
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric features for correlation heatmap.")

    # Target distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    if pd.api.types.is_numeric_dtype(df[target_column]) and df[target_column].nunique() > 10:
        sns.histplot(df[target_column], kde=True, ax=ax)
    else:
        sns.countplot(x=target_column, data=df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)


def feature_vs_target_streamlit(df, target_column):
    """
    Plot feature vs target intelligently based on dtype.
    """
    features = [col for col in df.columns if col != target_column]

    st.markdown("Feature vs Target plots (numeric = scatter, categorical = boxplot/countplot)")

    for feature in features:
        st.write(f"### {feature} vs {target_column}")

        try:
            fig, ax = plt.subplots(figsize=(6, 4))

            # Detect categorical columns by dtype OR low unique values
            feature_is_cat = (not pd.api.types.is_numeric_dtype(df[feature])) or (df[feature].nunique() < 15)
            target_is_cat = (not pd.api.types.is_numeric_dtype(df[target_column])) or (df[target_column].nunique() < 15)

            if feature_is_cat and target_is_cat:
                # Both categorical → countplot grouped
                sns.countplot(x=df[feature].astype(str), hue=df[target_column].astype(str), ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            elif feature_is_cat and not target_is_cat:
                # Categorical feature vs numeric target → boxplot
                sns.boxplot(x=df[feature].astype(str), y=df[target_column], ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            elif not feature_is_cat and target_is_cat:
                # Numeric feature vs categorical target → boxplot
                sns.boxplot(x=df[target_column].astype(str), y=df[feature], ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            else:
                # Both numeric → scatter
                sns.scatterplot(x=df[feature], y=df[target_column], ax=ax)

            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Could not plot {feature}: {e}")
