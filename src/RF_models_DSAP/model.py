"""
model.py

Model training, evaluation and persistence for the RF_models_DSAP pipeline.
Provides functions for hyperparameter tuning, training final Random Forest
models, evaluating performance and saving models to disk.
"""

import joblib
import os

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
    """Run GridSearchCV to find the best Random Forest hyperparameters.

    Uses 3-fold cross-validation with macro F1 score as the scoring
    metric to find the best combination of hyperparameters from
    RANDOM_FOREST_PARAM_GRID.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target labels.

    Returns:
        GridSearchCV: Fitted grid search object with best parameters
            accessible via grid.best_params_.
    """
    rf = RandomForestClassifier(random_state=MODEL_RANDOM_STATE)
    grid = GridSearchCV(rf, RANDOM_FOREST_PARAM_GRID, cv=3, scoring="f1_macro")
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    return grid


def train_multi_model(X_train, y_train):
    """Train the final multiclass Random Forest model.

    Uses the pre-defined FINAL_MULTICLASS_PARAMS for training.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Multiclass target labels.

    Returns:
        RandomForestClassifier: Trained multiclass Random Forest model.
    """
    rf_multi_final = RandomForestClassifier(**FINAL_MULTICLASS_PARAMS)
    rf_multi_final.fit(X_train, y_train)

    return rf_multi_final


def train_binary_model(X_train, y_train):
    """Train the final binary Random Forest model.

    Uses the pre-defined FINAL_BINARY_PARAMS for training.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Binary target labels.

    Returns:
        RandomForestClassifier: Trained binary Random Forest model.
    """
    rf_binary_final = RandomForestClassifier(**FINAL_BINARY_PARAMS)
    rf_binary_final.fit(X_train, y_train)

    return rf_binary_final


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and print accuracy and classification report.

    Args:
        model (RandomForestClassifier): Trained model to evaluate.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): True target labels.

    Returns:
        np.ndarray: Predicted labels for the test set.
    """
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return y_pred


def save_models(rf_multi_final, rf_binary_final, output_dir="models"):
    """Save trained models as .pkl files for later use.

    Creates the output directory if it does not exist, then saves
    both models using joblib serialization.

    Args:
        rf_multi_final (RandomForestClassifier): Trained multiclass model.
        rf_binary_final (RandomForestClassifier): Trained binary model.
        output_dir (str): Directory to save the models. Defaults to "models".

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(rf_multi_final, f"{output_dir}/model_multi.pkl")
    joblib.dump(rf_binary_final, f"{output_dir}/model_binary.pkl")

    print("Modelle gespeichert!")
