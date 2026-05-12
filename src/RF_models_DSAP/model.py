import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

from .config import (
    FINAL_BINARY_PARAMS,
    FINAL_MULTICLASS_PARAMS,
    MODEL_RANDOM_STATE,
    RANDOM_FOREST_PARAM_GRID,
)


def tune_hyperparameters(X_train, y_train):
    """Run GridSearchCV to find the best hyperparameters."""
    rf = RandomForestClassifier(random_state=MODEL_RANDOM_STATE)
    grid = GridSearchCV(rf, RANDOM_FOREST_PARAM_GRID, cv=3, scoring="f1_macro")
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    return grid


def train_multi_model(X_train, y_train):
    """Train the final multiclass Random Forest model."""
    rf_multi_final = RandomForestClassifier(**FINAL_MULTICLASS_PARAMS)
    rf_multi_final.fit(X_train, y_train)

    return rf_multi_final


def train_binary_model(X_train, y_train):
    """Train the final binary Random Forest model."""
    rf_binary_final = RandomForestClassifier(**FINAL_BINARY_PARAMS)
    rf_binary_final.fit(X_train, y_train)

    return rf_binary_final


def evaluate_model(model, X_test, y_test):
    """Evaluate model and print accuracy and classification report."""
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return y_pred


def save_models(rf_multi_final, rf_binary_final, output_dir="models"):
    """Save trained models as .pkl files for later use (e.g. inference, deployment)."""
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(rf_multi_final, f"{output_dir}/model_multi.pkl")
    joblib.dump(rf_binary_final, f"{output_dir}/model_binary.pkl")

    print("Modelle gespeichert!")
