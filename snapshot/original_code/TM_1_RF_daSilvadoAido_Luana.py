import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

expression_df = pd.read_csv("/Users/luanadoaido/ZHAW/HS25/Track1/Mini Project/Track/Trackmodule_1_RF_TCGA-STAD/data/TCGA-STAD_gene_expression_cpm.csv", index_col=0)
# 431 rows und 60616 columns
print("Data shape Gene Expression:", expression_df.shape)

subtype_df = pd.read_csv("/Users/luanadoaido/ZHAW/HS25/Track1/Mini Project/Track/Trackmodule_1_RF_TCGA-STAD/TCGA-STAD_subtypes.csv", index_col=0)
print("Data shape Subtypes:", subtype_df.shape)
# 411 rows und 4 columns

#Are all ID's represented?
expr_ids = set(expression_df.index)
label_ids = set(subtype_df.index)

print(len(expr_ids))
print(len(label_ids))
print(len(expr_ids & label_ids))  # Overlapping samples
# as only 411 samples overlap the merged df will be smalller

#merging

cancer_df = pd.merge(subtype_df, expression_df, on="submitter_id", how="inner")
print("Data shape whole DF:", cancer_df.shape)
#411 rows und 60620 Columns

#print(cancer_df.head(10))

print("\nThere are", cancer_df.isna().sum().sum(), "Na's in the Dataframe.") #157

print("\nIn which features are the Na's present?\n", cancer_df.isna().sum().sort_values(ascending=False).head())
#there are 157 na

print("\nThere are", cancer_df.index.duplicated().sum(), "duplicated rows.")
#no duplicated rows
print("\nThere are", cancer_df.columns.duplicated().sum(), "duplicated Columns.")
#no duplicated columns

#cat variablen rausnehmen, verändern
cancer_df.select_dtypes(include="object").head()

#as we have two different approaches we need to make two different Datasets which we need to clean
df_multi = cancer_df.dropna(subset=["Molecular.Subtype"]).copy()

print(df_multi["Molecular.Subtype"].value_counts())

df_binary = cancer_df.dropna(subset=["MSI_phenotype"]).copy()

df_binary["MSI_binary"] = df_binary["MSI_phenotype"].replace({
    "MSI-H": "MSI",
    "MSI-L": "MSI"
})

print(df_binary["MSI_binary"].value_counts())

#for the multi class approach
y_multi = df_multi["Molecular.Subtype"] #Targe Value
X_multi = df_multi.drop(columns=df_multi.select_dtypes(include="object").columns)

print(X_multi.dtypes.unique())   #should be only floeats


#for the binary class approach
y_binary = df_binary["MSI_binary"] #Targen value 2
X_binary = df_binary.drop(columns=df_binary.select_dtypes(include="object").columns)

print(X_binary.dtypes.unique())

#Class Distribution
#multi class

plt.Figure()
sns.countplot(x= y_multi)
plt.title("Class Distribution (CMS multiclass)")
plt.xlabel("CMS class")
plt.ylabel("Number of samples")
plt.show()

#binary

plt.Figure()
sns.countplot(x= y_binary)
plt.title("Class Distribution (Binary)")
plt.xlabel("Class")
plt.ylabel("Number of samples")
plt.show()

#PCA - Multi

scaler = StandardScaler()
X_scaled_multi = scaler.fit_transform(X_multi)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled_multi)

plt.figure()
sns.scatterplot(x = x_pca[:,0], y= x_pca[:,1], hue = y_multi)
plt.title("PCA - CMS classes (Multi class)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#PCA - binary

scaler = StandardScaler()
X_scaled_binary = scaler.fit_transform(X_binary)

pca = PCA(n_components=2)
x_pca_binary = pca.fit_transform(X_scaled_binary)

plt.figure()
sns.scatterplot(x = x_pca_binary[:,0], y= x_pca_binary[:,1], hue = y_binary)
plt.title("PCA - CMS classes (binary)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#heatmap with variance for multi
var_multi = X_multi.var()
top20_var = var_multi.sort_values(ascending=False).head(20)
feat_top = top20_var.index

#correlation Heatmap
plt.Figure(figsize=(10,8))
corr_multi = X_multi[feat_top].corr()
sns.heatmap(corr_multi, cmap = "coolwarm")
plt.title("Correlation heatmap (top Features - Multi Class)")
plt.show()

#heatmap with variance for multi
var_binary = X_binary.var()
top20_var_bin = var_binary.sort_values(ascending=False).head(20)
feat_top_bin = top20_var_bin.index

#correlation Heatmap
plt.Figure(figsize=(10,8))
corr_binary = X_binary[feat_top_bin].corr()
sns.heatmap(corr_binary, cmap = "coolwarm")
plt.title("Correlation heatmap (top Features - Binary)")
plt.show()

#for multi class approach
train_X_multi, test_X_multi, train_y_multi, test_y_multi = train_test_split(X_multi, y_multi, train_size= 0.7, random_state=1)

#for binary approach
train_X_binary, test_X_binary, train_y_binary, test_y_binary = train_test_split(X_binary, y_binary, train_size= 0.7, random_state=1)


#testing if the target value in the testing and training set is balanced
train_y_multi = pd.Series(train_y_multi)
test_y_multi = pd.Series(test_y_multi)
print("\nTesting to see if the training and test set is balanced (multi):\n (Count in %)")
print("train:", train_y_multi.value_counts(normalize=True) * 100)
print("\ntest:", test_y_multi.value_counts(normalize=True) * 100)


train_y_binary = pd.Series(train_y_binary)
test_y_binary = pd.Series(test_y_binary)
print("\nTesting to see if the training and test set is balanced (binary):\n (Count in %)")
print("train:", train_y_binary.value_counts(normalize=True) * 100)
print("\ntest:", test_y_binary.value_counts(normalize=True) * 100)

#for Multiclass Target - filter machen
min_samples_multi = int(0.1 *train_X_multi.shape[0])

#das eigentliche
mask_multi = (train_X_multi > 0).sum(axis=0) >= min_samples_multi

#Anwenden
X_train_multi_filt = train_X_multi.loc[:, mask_multi]
X_test_multi_filt = test_X_multi.loc[:, mask_multi]

#for Binary Target - filter machen
min_samples_binary = int(0.1 *train_X_binary.shape[0])

#das eigentliche
mask_binary = (train_X_binary > 0).sum(axis=0) >= min_samples_binary

#Anwenden
X_train_binary_filt = train_X_binary.loc[:, mask_binary]
X_test_binary_filt = test_X_binary.loc[:, mask_binary]


vt_multi = VarianceThreshold(threshold=0.01)
vt_multi.fit(X_train_multi_filt)

selected_genes_multi = X_train_multi_filt.columns[vt_multi.get_support()]

X_train_var_multi = X_train_multi_filt[selected_genes_multi]
X_test_var_multi  = X_test_multi_filt[selected_genes_multi]


vt_binary = VarianceThreshold(threshold=0.01)
vt_binary.fit(X_train_binary_filt)

selected_genes_binary = X_train_binary_filt.columns[vt_binary.get_support()]

X_train_var_binary = X_train_binary_filt[selected_genes_binary]
X_test_var_binary  = X_test_binary_filt[selected_genes_binary]

print(X_train_var_multi.shape[1],
X_test_var_multi.shape[1],
X_train_var_binary.shape[1],
X_test_var_binary.shape[1])



#Multi
param_grid_multi = {
    "n_estimators": [100, 300],
    "max_depth": [None, 20, 50, 100],
    "max_features": ["sqrt"],
    "min_samples_leaf": [1,2,5, 10]
}

rf_multi = RandomForestClassifier(random_state=42)
grid_mulit = GridSearchCV(rf_multi, param_grid_multi, cv=3, scoring="f1_macro")
grid_mulit.fit(X_train_var_multi , train_y_multi)

print("Best parameters:", grid_mulit.best_params_)
print("Best CV score:", grid_mulit.best_score_)


#binary
param_grid_binary = {
    "n_estimators": [100, 300],
    "max_depth": [None, 20, 50, 100],
    "max_features": ["sqrt"],
    "min_samples_leaf": [1,2,5, 10]
}

rf_binary = RandomForestClassifier(random_state=42)
grid_binary = GridSearchCV(rf_binary, param_grid_binary, cv=3, scoring="f1_macro")
grid_binary.fit(X_train_var_binary , train_y_binary)

print("Best parameters:", grid_binary.best_params_)
print("Best CV score:", grid_binary.best_score_)

rfc_multi = RandomForestClassifier(n_estimators=300, bootstrap=True, min_samples_leaf= 2, max_depth=None, max_features='sqrt', criterion='gini', random_state=42)
rfc_multi.fit(X_train_var_multi , train_y_multi)



rfc_binary = RandomForestClassifier(n_estimators=100, bootstrap=True, max_depth=None, max_features='sqrt', min_samples_leaf=10, criterion='gini', random_state=42)
rfc_binary.fit(X_train_var_binary , train_y_binary)

#Predicting
y_pred_multi = rfc_multi.predict(X_test_var_multi)

#metrics
#accuracy score
print("Accuracy of multi class RF:", accuracy_score(test_y_multi, y_pred_multi))
print(classification_report(test_y_multi, y_pred_multi))

#Confusionmatrix:
multi_test = confusion_matrix(test_y_multi, y_pred_multi)

plt.figure(figsize=(8, 6))
sns.heatmap(multi_test, annot=True, fmt="d", cmap='Greens',
            xticklabels=rfc_multi.classes_,
            yticklabels=rfc_multi.classes_)
plt.title('Confusion Matrix on Test Set (Final Evaluation - Multi)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance
feature_importance_multi = pd.DataFrame({
    "feature": X_test_var_multi.columns,
    "importance": rfc_multi.feature_importances_
}).sort_values("importance", ascending=False)

top10_multi = feature_importance_multi.head(10)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance_multi.head(10), x="importance", y="feature", orient="h")
plt.title("Top 10 Feature Importances (Multi)")
plt.xlabel("Importance")
plt.ylabel("Gene Expression")

for i, (imp, feat) in enumerate(zip(top10_multi["importance"], top10_multi["feature"])):
    plt.text(
        imp - 0.0002 * max(top10_multi["importance"]),
        i,
        f"{imp:.3f}",
        va="center",
        ha="left"
    )

plt.tight_layout()
plt.show()

#Predicting
y_pred_binary = rfc_binary.predict(X_test_var_binary)

#metrics
#accuracy score
print("Accuracy of Binary class RF:", accuracy_score(test_y_binary, y_pred_binary))
print(classification_report(test_y_binary, y_pred_binary))

#Confusionmatrix:
binary_test = confusion_matrix(test_y_binary, y_pred_binary)

plt.figure(figsize=(8, 6))
sns.heatmap(binary_test, annot=True, fmt="d", cmap="Greens",
            xticklabels=rfc_binary.classes_,
            yticklabels=rfc_binary.classes_)


plt.title("Confusion Matrix on Test Set (Final Evaluation - Binary)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()



#ROC-AUC
#predict probabilities for MSS
test_prob_binary = rfc_binary.predict_proba(X_test_var_binary)[:,1] #for posiive variable MSI
fpr_binary, tpr_binary, thresholds = roc_curve(test_y_binary, test_prob_binary, pos_label="MSI")
roc_auc_binary = auc(fpr_binary,tpr_binary)
print("ROC-AUC Value:", roc_auc_binary)
#Roc curve visualization:

plt.Figure(figsize=(8,6))
plt.plot(fpr_binary,tpr_binary, color ="darkgreen", lw = 2,
         label=f"Roc Curve (AUC = {roc_auc_binary:.2f})")

plt.plot([0,1],[0,1], color = "gray", lw= 1, linestyle = "--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Test Set")
plt.legend(loc = "lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Feature importance
feature_importance_binary = pd.DataFrame({
    "feature": X_test_var_binary.columns,
    "importance": rfc_binary.feature_importances_
}).sort_values("importance", ascending=False)


top10_binary = feature_importance_binary.head(10)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance_binary.head(10), x="importance", y="feature", orient="h")
plt.title("Top 10 Feature Importances (Binary)")
plt.xlabel("Importance")
plt.ylabel("Gene Expression")

for i, (imp, feat) in enumerate(zip(top10_binary["importance"], top10_binary["feature"])):
    plt.text(
        imp - 0.0002 * max(top10_binary["importance"]),
        i,
        f"{imp:.3f}",
        va="center",
        ha="left"
    )

plt.tight_layout()
plt.show()