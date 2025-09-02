from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, target_column):
    """
    Handle missing values, encode categorical columns, scale features
    """
    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna('Unknown')
    
    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        if col != target_column:
            df[col] = LabelEncoder().fit_transform(df[col])
    
    # Separate X and y
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[X.columns] = scaler.fit_transform(X)
    
    return X_scaled, y, scaler
