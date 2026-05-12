from sklearn.model_selection import train_test_split

from .config import TRAIN_SIZE, TRAIN_TEST_RANDOM_STATE
from .model import (
    evaluate_model,
    save_models,
    train_binary_model,
    train_multi_model,
    tune_hyperparameters,
)
from .preprocessing import (
    load_data,
    prepare_binary,
    prepare_multiclass,
    select_features,
)
from .visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_pca,
    plot_roc_curve,
)


def main():
    # ── Load & merge data ──────────────────────────────────────────
    cancer_df = load_data()

    # ── Prepare datasets ───────────────────────────────────────────
    X_multi, y_multi = prepare_multiclass(cancer_df)
    X_binary, y_binary = prepare_binary(cancer_df)

    # ── Exploratory plots ──────────────────────────────────────────
    plot_class_distribution(y_multi, "Class Distribution (CMS multiclass)", "CMS class")
    plot_class_distribution(y_binary, "Class Distribution (Binary)", "Class")
    plot_pca(X_multi, y_multi, "PCA - CMS classes (Multi class)")
    plot_pca(X_binary, y_binary, "PCA - CMS classes (binary)")
    plot_correlation_heatmap(X_multi, "Correlation heatmap (top Features - Multi Class)")
    plot_correlation_heatmap(X_binary, "Correlation heatmap (top Features - Binary)")

    # ── Train/test split ───────────────────────────────────────────
    train_X_multi, test_X_multi, train_y_multi, test_y_multi = train_test_split(
        X_multi, y_multi, train_size=TRAIN_SIZE, random_state=TRAIN_TEST_RANDOM_STATE
    )
    train_X_binary, test_X_binary, train_y_binary, test_y_binary = train_test_split(
        X_binary, y_binary, train_size=TRAIN_SIZE, random_state=TRAIN_TEST_RANDOM_STATE
    )

    # ── Feature selection ──────────────────────────────────────────
    X_train_var_multi, X_test_var_multi = select_features(train_X_multi, test_X_multi)
    X_train_var_binary, X_test_var_binary = select_features(
        train_X_binary, test_X_binary
    )

    # ── Hyperparameter tuning ──────────────────────────────────────
    tune_hyperparameters(X_train_var_multi, train_y_multi)
    tune_hyperparameters(X_train_var_binary, train_y_binary)

    # ── Train final models ─────────────────────────────────────────
    rf_multi_final = train_multi_model(X_train_var_multi, train_y_multi)
    rf_binary_final = train_binary_model(X_train_var_binary, train_y_binary)

    # ── Evaluate models ────────────────────────────────────────────
    y_pred_multi = evaluate_model(rf_multi_final, X_test_var_multi, test_y_multi)
    y_pred_binary = evaluate_model(rf_binary_final, X_test_var_binary, test_y_binary)

    # ── Visualize results ──────────────────────────────────────────
    plot_confusion_matrix(
        rf_multi_final, test_y_multi, y_pred_multi,
        "Confusion Matrix - Multi Class"
    )
    plot_confusion_matrix(
        rf_binary_final, test_y_binary, y_pred_binary,
        "Confusion Matrix - Binary"
    )
    plot_feature_importance(rf_multi_final, X_test_var_multi, "Top 10 Features (Multi)")
    plot_feature_importance(
        rf_binary_final, X_test_var_binary, "Top 10 Features (Binary)"
    )
    plot_roc_curve(rf_binary_final, X_test_var_binary, test_y_binary)

    # ── Save models ────────────────────────────────────────────────
    save_models(rf_multi_final, rf_binary_final)


if __name__ == "__main__":
    main()
