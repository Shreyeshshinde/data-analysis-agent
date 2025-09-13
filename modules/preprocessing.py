# modules/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

def preprocess_data_streamlit(df, target_column, handle_outliers=True, remove_corr=True):
    """
    Preprocess dataset for ML:
      - Handle missing values
      - Encode categoricals
      - Remove duplicates
      - Detect & remove outliers
      - Drop low-variance & highly correlated features
      - Scale features
    Returns:
      X_processed (pd.DataFrame)
      y (pd.Series)
      df_cleaned (pd.DataFrame)
      scaler (StandardScaler)
    """
    st.write("### 1) Basic cleaning steps")

    # Drop completely empty rows/cols
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # Remove duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    st.write(f"Removed {before - df.shape[0]} duplicate rows")

    # Show missing before actions
    missing = df.isnull().sum()
    num_missing = missing[missing > 0]
    if not num_missing.empty:
        st.write("Columns with missing values (handled below):")
        st.dataframe(num_missing)

    # Fill numeric missing with mean
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].mean(), inplace=True)
            st.write(f"Filled numeric column **{c}** with mean")

    # Fill categorical missing with mode
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        if df[c].isnull().any():
            df[c].fillna(df[c].mode().iloc[0], inplace=True)
            st.write(f"Filled categorical column **{c}** with mode")

    # Encode categoricals
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Outlier detection & removal (IQR method)
    if handle_outliers:
        st.write("### 2) Outlier detection & removal")
        for col in num_cols:
            if col not in df_encoded.columns:
                continue
            Q1 = df_encoded[col].quantile(0.25)
            Q3 = df_encoded[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before = df_encoded.shape[0]
            df_encoded = df_encoded[(df_encoded[col] >= lower) & (df_encoded[col] <= upper)]
            removed = before - df_encoded.shape[0]
            if removed > 0:
                st.write(f"Removed {removed} outliers in column **{col}**")

    # Separate target
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found after cleaning.")
        return None, None, df, None

    y = df[target_column].loc[df_encoded.index]  # align with outlier removal
    X = df_encoded.drop(columns=[target_column], errors="ignore")

    # Drop near-constant columns
    st.write("### 3) Dropping low-variance features")
    low_var_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if low_var_cols:
        st.write(f"Dropped {len(low_var_cols)} low-variance features: {low_var_cols}")
        X = X.drop(columns=low_var_cols)

    # Drop highly correlated features
    if remove_corr and X.shape[1] > 1:
        st.write("### 4) Checking correlations")
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
        if drop_cols:
            st.write(f"Dropped {len(drop_cols)} highly correlated features: {drop_cols}")
            X = X.drop(columns=drop_cols)

    # Standard scaling
    scaler = StandardScaler()
    if X.shape[1] == 0:
        st.warning("No features left after preprocessing.")
        return X, y, df, None

    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    st.write("✅ Preprocessing finished. Cleaned, encoded, outliers removed, and scaled features ready.")
    return X_scaled, y.reset_index(drop=True), df.reset_index(drop=True), scaler
