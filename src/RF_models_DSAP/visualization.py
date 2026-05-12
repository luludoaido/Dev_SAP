import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

from .config import TOP_FEATURE_COUNT, TOP_IMPORTANCE_COUNT


def plot_class_distribution(y, title, x_label):
    """Plot the distribution of class labels."""
    plt.figure()
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Number of samples")
    plt.show()


def plot_pca(X, y, title):
    """Plot the first two PCA components colored by class label."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


def plot_correlation_heatmap(X, title):
    """Plot a correlation heatmap of the top high-variance features."""
    top_features = X.var().sort_values(ascending=False).head(TOP_FEATURE_COUNT).index
    correlation_matrix = X[top_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm")
    plt.title(title)
    plt.show()


def plot_confusion_matrix(model, y_test, y_pred, title):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def plot_feature_importance(model, X_test, title):
    """Plot the top feature importances of a trained model."""
    feature_importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    top = feature_importance.head(TOP_IMPORTANCE_COUNT)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top, x="importance", y="feature", orient="h")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Gene Expression")

    for i, (imp, feat) in enumerate(zip(top["importance"], top["feature"])):
        plt.text(
            imp - 0.0002 * max(top["importance"]),
            i,
            f"{imp:.3f}",
            va="center",
            ha="left",
        )
    plt.tight_layout()
    plt.show()


def plot_roc_curve(model, X_test, y_test, pos_label="MSI"):
    """Plot the ROC curve and print the AUC score for a binary model."""
    test_probabilities = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, test_probabilities, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    print("ROC-AUC Value:", roc_auc)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkgreen", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
