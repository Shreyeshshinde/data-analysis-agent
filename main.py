from modules.data_ingestion import load_csv
from modules.eda import perform_eda
from modules.preprocessing import preprocess_data
from modules.automl_module import automl_task
from modules.visualization import generate_report
from modules.output_layer import output_dashboard
import os

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    file_path = input("Enter CSV file path: ")
    target_column = input("Enter target column name: ")
    
    # 1. Data Ingestion
    df = load_csv(file_path)
    if df is None:
        return
    
    # 2. EDA
    perform_eda(df)
    
    # 3. Preprocessing
    X, y, scaler = preprocess_data(df, target_column)
    
    # 4. AutoML
    best_model = automl_task(X, y)
    
    # 5. Visualization & Reports
    generate_report(df, target_column)
    
    # 6. Output Dashboard
    output_dashboard()

if __name__ == "__main__":
    main()
