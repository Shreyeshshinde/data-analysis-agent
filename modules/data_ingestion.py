import pandas as pd

def load_csv(file_path):
    """
    Load CSV file and validate
    """
    try:
        df = pd.read_csv(file_path)
        print(f"CSV loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
