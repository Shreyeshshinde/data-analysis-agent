import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def automl_task_streamlit(X, y):
    """
    AutoML with GridSearchCV for both Classification and Regression.
    Returns best model and leaderboard dataframe.
    """

    # Detect problem type
    problem_type = "classification"
    if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10:
        problem_type = "regression"

    st.write(f"🔍 Detected Task: **{problem_type.upper()}**")

    # Encode target if categorical
    if problem_type == "classification" and (y.dtype == "object" or str(y.dtype).startswith("category")):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define models
    if problem_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=500),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC()
        }

        param_grids = {
            "LogisticRegression": {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            },
            "RandomForestClassifier": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10]
            },
            "GradientBoostingClassifier": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "SVC": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            }
        }

    else:  # regression
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "SVR": SVR()
        }

        param_grids = {
            "LinearRegression": {},  # no hyperparams
            "RandomForestRegressor": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10]
            },
            "GradientBoostingRegressor": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "SVR": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            }
        }

    results = []
    metric_name = "Accuracy" if problem_type == "classification" else "R2"

    for name, model in models.items():
        try:
            st.write(f"🔄 Running GridSearch for **{name}**...")

            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grids.get(name, {}),
                scoring="accuracy" if problem_type == "classification" else "r2",
                cv=3,
                n_jobs=-1
            )
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            preds = best_model.predict(X_test)

            score = (
                accuracy_score(y_test, preds)
                if problem_type == "classification"
                else r2_score(y_test, preds)
            )

            results.append({
                "Model": name,
                metric_name: score,
                "BestParams": str(grid.best_params_)  # ✅ FIX applied
            })

            st.write(f"{name} {metric_name}: {score:.4f} | Best Params: {grid.best_params_}")

            # Save best model
            best_so_far = max([r[metric_name] for r in results]) if results else -np.inf
            if score == best_so_far:
                os.makedirs("models", exist_ok=True)
                joblib.dump(best_model, os.path.join("models", "best_model.pkl"))

        except Exception as e:
            st.error(f"{name} failed: {e}")

    if results:
        leaderboard_df = pd.DataFrame(results).sort_values(
            by=metric_name, ascending=False
        ).reset_index(drop=True)
        best_model = leaderboard_df.iloc[0]["Model"]

        # 📊 Safe plotting
        st.bar_chart(leaderboard_df.set_index("Model")[metric_name])

        return best_model, leaderboard_df
    else:
        return None, None
