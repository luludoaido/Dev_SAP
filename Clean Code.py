import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# DATA LOADING

# Load gene expression matrix (samples x genes, CPM normalized)
expression_df = pd.read_csv(
    "/Users/luanadoaido/ZHAW/HS25/Track1/Mini Project/Track/Trackmodule_1_RF_TCGA-STAD/data/TCGA-STAD_gene_expression_cpm.csv",
    index_col=0
)

# Load subtype annotations for the same cohort
subtype_df = pd.read_csv(
    "/Users/luanadoaido/ZHAW/HS25/Track1/Mini Project/Track/Trackmodule_1_RF_TCGA-STAD/TCGA-STAD_subtypes.csv",
    index_col=0
)


# DATA CONSISTENCY CHECK

# Ensure that sample IDs overlap between expression and subtype tables
expr_ids = set(expression_df.index)
label_ids = set(subtype_df.index)

print("Number of expression samples:", len(expr_ids))
print("Number of subtype samples:", len(label_ids))
print("Overlapping samples:", len(expr_ids & label_ids))


# MERGING DATASETS

# Keep only samples present in both datasets
cancer_df = pd.merge(subtype_df, expression_df, on="submitter_id", how="inner")

print("Merged dataset shape:", cancer_df.shape)

# Check data quality
print("\nTotal missing values:", cancer_df.isna().sum().sum())
print("\nTop features with missing values:\n",
      cancer_df.isna().sum().sort_values(ascending=False).head())

print("\nDuplicated rows:", cancer_df.index.duplicated().sum())
print("Duplicated columns:", cancer_df.columns.duplicated().sum())


# DATASET PREPARATION

# Create two separate datasets:
# 1) Multiclass CMS classification
# 2) Binary MSI classification

df_multi = cancer_df.dropna(subset=["Molecular.Subtype"]).copy()
df_binary = cancer_df.dropna(subset=["MSI_phenotype"]).copy()

# Collapse MSI-H and MSI-L into one MSI class
# (biological decision: treat both as instability phenotype)
df_binary["MSI_binary"] = df_binary["MSI_phenotype"].replace({
    "MSI-H": "MSI",
    "MSI-L": "MSI"
})


# FEATURE / TARGET SPLIT

# Remove categorical columns to keep only numeric gene expression values

y_multi = df_multi["Molecular.Subtype"]
X_multi = df_multi.drop(columns=df_multi.select_dtypes(include="object").columns)

y_binary = df_binary["MSI_binary"]
X_binary = df_binary.drop(columns=df_binary.select_dtypes(include="object").columns)


# EXPLORATORY ANALYSIS

# Visualize class balance (important for model evaluation strategy)

sns.countplot(x=y_multi)
plt.title("Class Distribution (CMS multiclass)")
plt.show()

sns.countplot(x=y_binary)
plt.title("Class Distribution (Binary)")
plt.show()


# PCA VISUALIZATION

# PCA is used only for visualization to inspect potential class separation

scaler = StandardScaler()
X_scaled_multi = scaler.fit_transform(X_multi)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled_multi)

sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=y_multi)
plt.title("PCA - CMS classes (Multi class)")
plt.show()


scaler = StandardScaler()
X_scaled_binary = scaler.fit_transform(X_binary)

x_pca_binary = pca.fit_transform(X_scaled_binary)

sns.scatterplot(x=x_pca_binary[:, 0], y=x_pca_binary[:, 1], hue=y_binary)
plt.title("PCA - CMS classes (Binary)")
plt.show()


# FEATURE FILTERING

# Step 1: Remove genes expressed in less than 10% of training samples
# (low-information features)

train_X_multi, test_X_multi, train_y_multi, test_y_multi = train_test_split(
    X_multi, y_multi, train_size=0.7, random_state=1
)

train_X_binary, test_X_binary, train_y_binary, test_y_binary = train_test_split(
    X_binary, y_binary, train_size=0.7, random_state=1
)

min_samples_multi = int(0.1 * train_X_multi.shape[0])
mask_multi = (train_X_multi > 0).sum(axis=0) >= min_samples_multi

X_train_multi_filt = train_X_multi.loc[:, mask_multi]
X_test_multi_filt = test_X_multi.loc[:, mask_multi]


min_samples_binary = int(0.1 * train_X_binary.shape[0])
mask_binary = (train_X_binary > 0).sum(axis=0) >= min_samples_binary

X_train_binary_filt = train_X_binary.loc[:, mask_binary]
X_test_binary_filt = test_X_binary.loc[:, mask_binary]


# Step 2: Remove near-constant features (low variance)
vt_multi = VarianceThreshold(threshold=0.01)
vt_multi.fit(X_train_multi_filt)

selected_genes_multi = X_train_multi_filt.columns[vt_multi.get_support()]
X_train_var_multi = X_train_multi_filt[selected_genes_multi]
X_test_var_multi = X_test_multi_filt[selected_genes_multi]


vt_binary = VarianceThreshold(threshold=0.01)
vt_binary.fit(X_train_binary_filt)

selected_genes_binary = X_train_binary_filt.columns[vt_binary.get_support()]
X_train_var_binary = X_train_binary_filt[selected_genes_binary]
X_test_var_binary = X_test_binary_filt[selected_genes_binary]


# MODEL SELECTION (GRID SEARCH)

# Use f1_macro because of potential class imbalance

param_grid_multi = {
    "n_estimators": [100, 300],
    "max_depth": [None, 20, 50, 100],
    "max_features": ["sqrt"],
    "min_samples_leaf": [1, 2, 5, 10]
}

rf_multi = RandomForestClassifier(random_state=42)

grid_multi = GridSearchCV(
    rf_multi,
    param_grid_multi,
    cv=3,
    scoring="f1_macro"
)

grid_multi.fit(X_train_var_multi, train_y_multi)


param_grid_binary = {
    "n_estimators": [100, 300],
    "max_depth": [None, 20, 50, 100],
    "max_features": ["sqrt"],
    "min_samples_leaf": [1, 2, 5, 10]
}

rf_binary = RandomForestClassifier(random_state=42)

grid_binary = GridSearchCV(
    rf_binary,
    param_grid_binary,
    cv=3,
    scoring="f1_macro"
)

grid_binary.fit(X_train_var_binary, train_y_binary)


# FINAL MODEL TRAINING

# Train final models using best-performing hyperparameters

rfc_multi = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42
)

rfc_multi.fit(X_train_var_multi, train_y_multi)


rfc_binary = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=10,
    max_features="sqrt",
    random_state=42
)

rfc_binary.fit(X_train_var_binary, train_y_binary)


# EVALUATION

y_pred_multi = rfc_multi.predict(X_test_var_multi)

print("Accuracy (multi):", accuracy_score(test_y_multi, y_pred_multi))
print(classification_report(test_y_multi, y_pred_multi))


y_pred_binary = rfc_binary.predict(X_test_var_binary)

print("Accuracy (binary):", accuracy_score(test_y_binary, y_pred_binary))
print(classification_report(test_y_binary, y_pred_binary))


# ROC-AUC only meaningful for binary classification
test_prob_binary = rfc_binary.predict_proba(X_test_var_binary)[:, 1]

fpr_binary, tpr_binary, thresholds = roc_curve(
    test_y_binary,
    test_prob_binary,
    pos_label="MSI"
)

roc_auc_binary = auc(fpr_binary, tpr_binary)

plt.plot(fpr_binary, tpr_binary, label=f"AUC = {roc_auc_binary:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - Binary Classification")
plt.legend()
plt.show()
