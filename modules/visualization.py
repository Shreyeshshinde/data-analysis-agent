import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_report(df, target_column):
    """
    Generate and save plots & reports
    """
    os.makedirs("reports", exist_ok=True)
    
    # Target distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_column, data=df)
    plt.title("Target Distribution")
    plt.savefig("reports/target_distribution.png")
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig("reports/correlation_matrix.png")
    plt.show()
