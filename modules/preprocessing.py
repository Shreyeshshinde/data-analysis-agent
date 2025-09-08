from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, target_column):
    # Normalize column names (lowercase)
    df.columns = df.columns.str.lower()
    target_column = target_column.lower()
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler
