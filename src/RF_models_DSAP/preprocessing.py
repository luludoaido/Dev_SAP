import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .config import (
    EXPRESSION_FILE,
    MIN_NON_ZERO_FRACTION,
    SUBTYPE_FILE,
    VARIANCE_THRESHOLD,
)


def load_data():
    """Load gene expression and subtype data and merge them."""
    expression_df = pd.read_csv(EXPRESSION_FILE)
    print("Data shape Gene Expression:", expression_df.shape)

    subtype_df = pd.read_csv(SUBTYPE_FILE)
    print("Data shape Subtypes:", subtype_df.shape)

    # Keep only samples that have both subtype labels and expression data.
    cancer_df = pd.merge(subtype_df, expression_df, on="submitter_id", how="inner")
    print("Data shape whole DF:", cancer_df.shape)
    print("\nThere are", cancer_df.isna().sum().sum(), "Na's in the Dataframe.")

    return cancer_df


def prepare_multiclass(cancer_df):
    """Prepare features and labels for multiclass subtype classification."""
    df_multi = cancer_df.dropna(subset=["Molecular.Subtype"]).copy()
    print(df_multi["Molecular.Subtype"].value_counts())

    y_multi = df_multi["Molecular.Subtype"]
    X_multi = df_multi.drop(columns=df_multi.select_dtypes(include="object").columns)

    return X_multi, y_multi


def prepare_binary(cancer_df):
    """Prepare features and labels for binary MSI classification."""
    df_binary = cancer_df.dropna(subset=["MSI_phenotype"]).copy()

    # Combine MSI-H and MSI-L into one MSI class for binary classification.
    df_binary["MSI_binary"] = df_binary["MSI_phenotype"].replace(
        {"MSI-H": "MSI", "MSI-L": "MSI"}
    )
    print(df_binary["MSI_binary"].value_counts())

    y_binary = df_binary["MSI_binary"]
    X_binary = df_binary.drop(columns=df_binary.select_dtypes(include="object").columns)

    return X_binary, y_binary


def filter_non_expressed_features(train_X, test_X):
    """Remove features that are zero in too many samples."""
    min_samples = int(MIN_NON_ZERO_FRACTION * train_X.shape[0])
    expressed_features = (train_X > 0).sum(axis=0) >= min_samples

    return train_X.loc[:, expressed_features], test_X.loc[:, expressed_features]


def apply_variance_filter(train_X, test_X):
    """Remove low-variance features using a threshold."""
    variance_filter = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    variance_filter.fit(train_X)
    selected_features = train_X.columns[variance_filter.get_support()]

    return train_X[selected_features], test_X[selected_features]


def select_features(train_X, test_X):
    """Apply all feature selection steps."""
    train_X, test_X = filter_non_expressed_features(train_X, test_X)
    train_X, test_X = apply_variance_filter(train_X, test_X)

    return train_X, test_X
