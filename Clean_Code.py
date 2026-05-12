from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(".")
EXPRESSION_FILE = BASE_DIR / "data" / "TCGA-STAD_gene_expression_cpm.csv"
SUBTYPE_FILE = BASE_DIR / "data" / "TCGA-STAD_subtypes.csv"


TRAIN_SIZE = 0.7
TRAIN_TEST_RANDOM_STATE = 1
MODEL_RANDOM_STATE = 42
MIN_NON_ZERO_FRACTION = 0.1
VARIANCE_THRESHOLD = 0.01
TOP_FEATURE_COUNT = 20
TOP_IMPORTANCE_COUNT = 10


RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [None, 20, 50, 100],
    "max_features": ["sqrt"],
    "min_samples_leaf": [1, 2, 5, 10],
}

FINAL_MULTICLASS_PARAMS = {
    "n_estimators": 300,
    "bootstrap": True,
    "min_samples_leaf": 2,
    "max_depth": None,
    "max_features": "sqrt",
    "criterion": "gini",
    "random_state": MODEL_RANDOM_STATE,
}

FINAL_BINARY_PARAMS = {
    "n_estimators": 100,
    "bootstrap": True,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 10,
    "criterion": "gini",
    "random_state": MODEL_RANDOM_STATE,
}


def plot_class_distribution(y, title, x_label):
    plt.figure()
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Number of samples")
    plt.show()


def plot_pca(X, y, title):
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
    top_features = X.var().sort_values(ascending=False).head(TOP_FEATURE_COUNT).index
    correlation_matrix = X[top_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap="coolwarm")
    plt.title(title)
    plt.show()


def filter_non_expressed_features(train_X, test_X):
    min_samples = int(MIN_NON_ZERO_FRACTION * train_X.shape[0])
    expressed_features = (train_X > 0).sum(axis=0) >= min_samples

    return train_X.loc[:, expressed_features], test_X.loc[:, expressed_features]


def apply_variance_filter(train_X, test_X):
    variance_filter = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    variance_filter.fit(train_X)

    selected_features = train_X.columns[variance_filter.get_support()]

    return train_X[selected_features], test_X[selected_features]


def select_features(train_X, test_X):
    train_X, test_X = filter_non_expressed_features(train_X, test_X)
    train_X, test_X = apply_variance_filter(train_X, test_X)

    return train_X, test_X


expression_df = pd.read_csv(EXPRESSION_FILE)

print("Data shape Gene Expression:", expression_df.shape)

subtype_df = pd.read_csv(SUBTYPE_FILE)

print("Data shape Subtypes:", subtype_df.shape)


# Check overlap between expression and subtype sample IDs.
expression_ids = set(expression_df.index)
subtype_ids = set(subtype_df.index)

print(len(expression_ids))
print(len(subtype_ids))
print(len(expression_ids & subtype_ids))


# Keep only samples that have both subtype labels and expression data.
cancer_df = pd.merge(subtype_df, expression_df, on="submitter_id", how="inner")

print("Data shape whole DF:", cancer_df.shape)
print("\nThere are", cancer_df.isna().sum().sum(), "Na's in the Dataframe.")
print(
    "\nIn which features are the Na's present?\n",
    cancer_df.isna().sum().sort_values(ascending=False).head(),
)
print("\nThere are", cancer_df.index.duplicated().sum(), "duplicated rows.")
print("\nThere are", cancer_df.columns.duplicated().sum(), "duplicated Columns.")


cancer_df.select_dtypes(include="object").head()


# Prepare separate datasets for multiclass subtype and binary MSI classification.
df_multi = cancer_df.dropna(subset=["Molecular.Subtype"]).copy()

print(df_multi["Molecular.Subtype"].value_counts())

df_binary = cancer_df.dropna(subset=["MSI_phenotype"]).copy()


# Combine MSI-H and MSI-L into one MSI class for binary classification.
df_binary["MSI_binary"] = df_binary["MSI_phenotype"].replace(
    {
        "MSI-H": "MSI",
        "MSI-L": "MSI",
    }
)

print(df_binary["MSI_binary"].value_counts())


y_multi = df_multi["Molecular.Subtype"]
X_multi = df_multi.drop(columns=df_multi.select_dtypes(include="object").columns)

print(X_multi.dtypes.unique())


y_binary = df_binary["MSI_binary"]
X_binary = df_binary.drop(columns=df_binary.select_dtypes(include="object").columns)

print(X_binary.dtypes.unique())

# Plot class distributions for both target definitions.
plot_class_distribution(
    y_multi,
    title="Class Distribution (CMS multiclass)",
    x_label="CMS class",
)

plot_class_distribution(
    y_binary,
    title="Class Distribution (Binary)",
    x_label="Class",
)

# Visualize the first two PCA components for the multiclass target.
plot_pca(X_multi, y_multi, title="PCA - CMS classes (Multi class)")

# Visualize the first two PCA components for the binary target.
plot_pca(X_binary, y_binary, title="PCA - CMS classes (binary)")

# Plot correlations among the highest-variance features for the multiclass target.
plot_correlation_heatmap(
    X_multi,
    title="Correlation heatmap (top Features - Multi Class)",
)

# Plot correlations among the highest-variance features for the binary target.
plot_correlation_heatmap(
    X_binary,
    title="Correlation heatmap (top Features - Binary)",
)


train_X_multi, test_X_multi, train_y_multi, test_y_multi = train_test_split(
    X_multi,
    y_multi,
    train_size=TRAIN_SIZE,
    random_state=TRAIN_TEST_RANDOM_STATE,
)

train_X_binary, test_X_binary, train_y_binary, test_y_binary = train_test_split(
    X_binary,
    y_binary,
    train_size=TRAIN_SIZE,
    random_state=TRAIN_TEST_RANDOM_STATE,
)


# Compare class proportions in train and test splits.
train_y_multi = pd.Series(train_y_multi)
test_y_multi = pd.Series(test_y_multi)
print(
    "\nTesting to see if the training and test set is balanced (multi):\n (Count in %)"
)
print("train:", train_y_multi.value_counts(normalize=True) * 100)
print("\ntest:", test_y_multi.value_counts(normalize=True) * 100)


train_y_binary = pd.Series(train_y_binary)
test_y_binary = pd.Series(test_y_binary)
print(
    "\nTesting to see if the training and test set is balanced (binary):\n (Count in %)"
)
print("train:", train_y_binary.value_counts(normalize=True) * 100)
print("\ntest:", test_y_binary.value_counts(normalize=True) * 100)


# Select features separately for the multiclass and binary models.
X_train_var_multi, X_test_var_multi = select_features(
    train_X_multi,
    test_X_multi,
)

X_train_var_binary, X_test_var_binary = select_features(
    train_X_binary,
    test_X_binary,
)

print(
    X_train_var_multi.shape[1],
    X_test_var_multi.shape[1],
    X_train_var_binary.shape[1],
    X_test_var_binary.shape[1],
)


# Tune random forest hyperparameters for the multiclass model.
rf_multi = RandomForestClassifier(random_state=MODEL_RANDOM_STATE)
grid_multi = GridSearchCV(
    rf_multi,
    RANDOM_FOREST_PARAM_GRID,
    cv=3,
    scoring="f1_macro",
)
grid_multi.fit(X_train_var_multi, train_y_multi)

print("Best parameters:", grid_multi.best_params_)
print("Best CV score:", grid_multi.best_score_)


# Tune random forest hyperparameters for the binary model.
rf_binary = RandomForestClassifier(random_state=MODEL_RANDOM_STATE)
grid_binary = GridSearchCV(
    rf_binary,
    RANDOM_FOREST_PARAM_GRID,
    cv=3,
    scoring="f1_macro",
)
grid_binary.fit(X_train_var_binary, train_y_binary)

print("Best parameters:", grid_binary.best_params_)
print("Best CV score:", grid_binary.best_score_)

rf_multi_final = RandomForestClassifier(**FINAL_MULTICLASS_PARAMS)
rf_multi_final.fit(X_train_var_multi, train_y_multi)

rf_binary_final = RandomForestClassifier(**FINAL_BINARY_PARAMS)
rf_binary_final.fit(X_train_var_binary, train_y_binary)


# Evaluate the final multiclass model on the test set.
y_pred_multi = rf_multi_final.predict(X_test_var_multi)

print("Accuracy of multi class RF:", accuracy_score(test_y_multi, y_pred_multi))
print(classification_report(test_y_multi, y_pred_multi))

confusion_matrix_multi = confusion_matrix(test_y_multi, y_pred_multi)

plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix_multi,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=rf_multi_final.classes_,
    yticklabels=rf_multi_final.classes_,
)
plt.title("Confusion Matrix on Test Set (Final Evaluation - Multi)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# Inspect the most important features for the multiclass model.
feature_importance_multi = pd.DataFrame(
    {
        "feature": X_test_var_multi.columns,
        "importance": rf_multi_final.feature_importances_,
    }
).sort_values("importance", ascending=False)

top10_multi = feature_importance_multi.head(TOP_IMPORTANCE_COUNT)


plt.figure(figsize=(12, 8))
sns.barplot(
    data=feature_importance_multi.head(TOP_IMPORTANCE_COUNT),
    x="importance",
    y="feature",
    orient="h",
)
plt.title("Top 10 Feature Importances (Multi)")
plt.xlabel("Importance")
plt.ylabel("Gene Expression")

for i, (imp, feat) in enumerate(zip(top10_multi["importance"], top10_multi["feature"])):
    plt.text(
        imp - 0.0002 * max(top10_multi["importance"]),
        i,
        f"{imp:.3f}",
        va="center",
        ha="left",
    )
plt.tight_layout()
plt.show()


# Evaluate the final binary model on the test set.
y_pred_binary = rf_binary_final.predict(X_test_var_binary)

print("Accuracy of Binary class RF:", accuracy_score(test_y_binary, y_pred_binary))
print(classification_report(test_y_binary, y_pred_binary))

confusion_matrix_binary = confusion_matrix(test_y_binary, y_pred_binary)

plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix_binary,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=rf_binary_final.classes_,
    yticklabels=rf_binary_final.classes_,
)
plt.title("Confusion Matrix on Test Set (Final Evaluation - Binary)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()


# Compute ROC-AUC for the binary MSI model.
test_probabilities_binary = rf_binary_final.predict_proba(X_test_var_binary)[:, 1]
fpr_binary, tpr_binary, _ = roc_curve(
    test_y_binary,
    test_probabilities_binary,
    pos_label="MSI",
)
roc_auc_binary = auc(fpr_binary, tpr_binary)
print("ROC-AUC Value:", roc_auc_binary)


# Plot the ROC curve for the binary model.
plt.figure(figsize=(8, 6))
plt.plot(
    fpr_binary,
    tpr_binary,
    color="darkgreen",
    lw=2,
    label=f"Roc Curve (AUC = {roc_auc_binary:.2f})",
)
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Test Set")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Inspect the most important features for the binary model.
feature_importance_binary = pd.DataFrame(
    {
        "feature": X_test_var_binary.columns,
        "importance": rf_binary_final.feature_importances_,
    }
).sort_values("importance", ascending=False)

top10_binary = feature_importance_binary.head(TOP_IMPORTANCE_COUNT)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=feature_importance_binary.head(TOP_IMPORTANCE_COUNT),
    x="importance",
    y="feature",
    orient="h",
)
plt.title("Top 10 Feature Importances (Binary)")
plt.xlabel("Importance")
plt.ylabel("Gene Expression")

for i, (imp, feat) in enumerate(
    zip(top10_binary["importance"], top10_binary["feature"])
):
    plt.text(
        imp - 0.0002 * max(top10_binary["importance"]),
        i,
        f"{imp:.3f}",
        va="center",
        ha="left",
    )

plt.tight_layout()
plt.show()
