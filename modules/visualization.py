import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_report(df, target_column):
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    # Ensure target_column is valid
    if target_column not in df.columns:
        print(f"❌ Error: Target column '{target_column}' not found in dataframe columns.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Heatmap only on numeric data
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(reports_dir, "correlation_heatmap.png"))
    plt.close()

    # Distribution of target column
    if df[target_column].dtype == 'object' or str(df[target_column].dtype).startswith("category"):
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_column, data=df)
        plt.title(f"Distribution of {target_column}")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(reports_dir, f"{target_column}_distribution.png"))
        plt.close()
    else:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[target_column], kde=True)
        plt.title(f"Distribution of {target_column}")
        plt.savefig(os.path.join(reports_dir, f"{target_column}_distribution.png"))
        plt.close()

    print("📊 Reports generated and saved in 'reports/' folder")
