from pycaret.classification import setup as cls_setup, compare_models as cls_compare, save_model as cls_save_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, save_model as reg_save_model

def automl_task(X, y):
    """
    AutoML task detection (classification/regression)
    """
    import pandas as pd
    df = pd.concat([X, y], axis=1)
    
    if y.nunique() <= 20 and y.dtype in ['int64', 'object']:
        print("Detected Classification Task")
        s = cls_setup(data=df, target=y.name, silent=True, session_id=123)
        best_model = cls_compare()
        cls_save_model(best_model, 'models/best_model')
    else:
        print("Detected Regression Task")
        s = reg_setup(data=df, target=y.name, silent=True, session_id=123)
        best_model = reg_compare()
        reg_save_model(best_model, 'models/best_model')
    
    print(f"Best model saved to 'models/best_model.pkl'")
    return best_model
