import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def perform_eda(df):
    """
    Exploratory Data Analysis: stats, missing values, correlations
    """
    print("\n----- Basic Stats -----")
    print(df.describe(include='all'))  # include all columns
    
    print("\n----- Missing Values -----")
    print(df.isnull().sum())
    
    # 1️⃣ Correlation heatmap (numeric columns only)
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix (Numeric Columns)")
        plt.show()
    else:
        print("\nNo numeric columns for correlation heatmap.")
    
    # 2️⃣ Pairplot for first 5 numeric columns
    numeric_cols = numeric_df.columns[:5]
    if len(numeric_cols) > 1:
        sns.pairplot(df[numeric_cols])
        plt.show()
    
    # 3️⃣ Optional: Encode categorical columns and plot correlation
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        df_encoded = df.copy()
        for col in categorical_cols:
            df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix (All Columns Encoded)")
        plt.show()
