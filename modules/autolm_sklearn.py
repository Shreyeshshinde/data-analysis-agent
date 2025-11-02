# modules/autolm_sklearn.py

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
import joblib
import os

def automl_task_streamlit(X, y):
    """
    Runs a simple AutoML loop with multiple models.
    Supports both regression and classification tasks.
    Returns:
        best_model: trained model
        leaderboard_df: pandas dataframe of results
        training_history: dictionary of model training progress
    """

    # Detect task type
    task_type = "classification" if y.nunique() < 20 and y.dtype != float else "regression"
    st.write(f"🔍 Detected Task: **{task_type.upper()}**")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = []
    if task_type == "classification":
        models = [
            ("LogisticRegression", LogisticRegression(max_iter=500)),
            ("RandomForestClassifier", RandomForestClassifier()),
            ("GradientBoostingClassifier", GradientBoostingClassifier()),
            ("SVC", SVC()),
        ]
    else:
        models = [
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor()),
            ("GradientBoostingRegressor", GradientBoostingRegressor()),
            ("SVR", SVR()),
        ]

    results = []
    training_history = {}

    for name, model in models:
        st.write(f"🔄 Running GridSearch for **{name}**...")
        time.sleep(0.5)

        # simple param grid for quick testing
        param_grid = {}
        if "RandomForest" in name:
            param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, None]}
        elif "GradientBoosting" in name:
            param_grid = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
        elif "SVC" in name:
            param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}
        elif "SVR" in name:
            param_grid = {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}

        try:
            grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=0)
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_

            y_pred = best_model.predict(X_test)

            if task_type == "classification":
                score = accuracy_score(y_test, y_pred)
            else:
                score = r2_score(y_test, y_pred)

            results.append((name, score))
            training_history[name] = {
                "params": grid.best_params_,
                "score": score,
                "cv_results": grid.cv_results_
            }

            st.success(f"{name} achieved score: {score:.4f}")

        except Exception as e:
            st.error(f"{name} failed: {e}")
            continue

    if not results:
        st.error("AutoML failed. No successful models.")
        return None, None, {}

    # create leaderboard
    leaderboard_df = pd.DataFrame(results, columns=["Model", "Score"])
    leaderboard_df = leaderboard_df.sort_values(by="Score", ascending=False).reset_index(drop=True)

    best_model_name = leaderboard_df.iloc[0]["Model"]
    st.success(f"🏆 Best model: {best_model_name}")

    # retrain best model on full data
    best_model = None
    for name, model in models:
        if name == best_model_name:
            if "GridSearchCV" in str(type(model)):
                best_model = model.best_estimator_
            else:
                model.fit(X, y)
                best_model = model
            break

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")

    return best_model, leaderboard_df, training_history
