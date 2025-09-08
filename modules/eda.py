import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def perform_eda(df):
    """
    Exploratory Data Analysis: basic stats, missing values, correlations,
    and visualizations by species.
    """
    # Basic stats
    print("\n----- Basic Stats -----")
    print(df.describe(include='all'))
    
    # Missing values
    print("\n----- Missing Values -----")
    print(df.isnull().sum())
    
    # Correlation heatmap for numeric columns only
    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix (Numeric Features)")
        plt.show()
    
    # Pairplot colored by Species if exists
    if 'Species' in df.columns:
        numeric_cols_for_pairplot = df.select_dtypes(include='number').columns
        sns.pairplot(df[numeric_cols_for_pairplot.tolist() + ['Species']], hue='Species')
        plt.show()
    
    # Boxplots for each numeric feature by species
    if 'Species' in df.columns:
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x='Species', y=col, data=df)
            plt.title(f'{col} distribution by Species')
            plt.show()
    
    # Distribution plots for numeric features
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f'Distribution of {col}')
        plt.show()
