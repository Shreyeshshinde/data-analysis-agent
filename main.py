from modules.data_ingestion import load_csv
from modules.eda import perform_eda
from modules.preprocessing import preprocess_data
from modules.autolm_sklearn import automl_task
from modules.visualization import generate_report
from modules.output_layer import output_dashboard
import os

def main():
    # Create required directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1. Data Ingestion
    file_path = input("📂 Enter CSV file path: ").strip()
    df = load_csv(file_path)
    if df is None or df.empty:
        print("❌ Error: Dataframe is empty or file could not be loaded.")
        return

    # Normalize column names (strip spaces, lowercase)
    df.columns = df.columns.str.strip()
    print("\n✅ Available columns in dataset:")
    print(df.columns.tolist())

    # 2. Target column selection
    target_column = input("\n🎯 Enter target column name (choose from above): ").strip()
    matched_cols = [col for col in df.columns if col.lower() == target_column.lower()]
    if not matched_cols:
        print(f"❌ Error: Column '{target_column}' not found. Please choose from the list above.")
        return
    target_column = matched_cols[0]  # Correct column name from df

    # 3. Exploratory Data Analysis (EDA)
    print("\n🔍 Performing EDA...")
    perform_eda(df)

    # 4. Preprocessing
    print("\n⚙️ Preprocessing data...")
    X, y, scaler = preprocess_data(df, target_column)

    # 5. AutoML Training
    print("\n🤖 Running AutoML...")
    best_model = automl_task(X, y)

   # 6. Visualization & Reports
    print("\n📊 Showing visual reports...")
    generate_report(df, target_column, results_df)


    # 7. Output Dashboard
    print("\n📌 Launching output dashboard...")
    output_dashboard()

    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
