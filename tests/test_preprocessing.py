"""
Tests for selected functions in Clean_Code.py.

The full script is not imported because it executes data loading,
plotting and model training at import time.
"""

import ast
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


def get_default_source_file() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / "Clean_Code.py",
        Path(__file__).resolve().parent / "Clean_Code.py",
        Path.cwd() / "Clean_Code.py",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return Path("Clean_Code.py")


SOURCE_FILE = Path(os.environ.get("SOURCE_FILE", str(get_default_source_file())))

CONSTANTS_TO_LOAD = {
    "MIN_NON_ZERO_FRACTION",
    "VARIANCE_THRESHOLD",
    "TOP_FEATURE_COUNT",
}

FUNCTIONS_TO_LOAD = {
    "plot_class_distribution",
    "plot_pca",
    "plot_correlation_heatmap",
    "filter_non_expressed_features",
    "apply_variance_filter",
    "select_features",
}


def load_selected_project_code(source_file: Path):
    if not source_file.exists():
        raise FileNotFoundError(
            f"Could not find {source_file}. "
            "Make sure Clean_Code.py exists in the repository root."
        )

    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source)

    selected_nodes = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            assigned_names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }

            if assigned_names & CONSTANTS_TO_LOAD:
                selected_nodes.append(node)

        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS_TO_LOAD:
            selected_nodes.append(node)

    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {
        "pd": pd,
        "plt": plt,
        "sns": sns,
        "PCA": PCA,
        "StandardScaler": StandardScaler,
        "VarianceThreshold": VarianceThreshold,
    }

    exec(compile(module, filename=str(source_file), mode="exec"), namespace)

    return namespace


def test_filter_non_expressed_features_removes_never_expressed_gene():
    functions = load_selected_project_code(SOURCE_FILE)
    filter_non_expressed_features = functions["filter_non_expressed_features"]

    train_X = pd.DataFrame(
        {
            "gene_a": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "gene_b": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "gene_c": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        }
    )

    test_X = pd.DataFrame(
        {
            "gene_a": [1, 2],
            "gene_b": [0, 0],
            "gene_c": [5, 6],
        }
    )

    filtered_train, filtered_test = filter_non_expressed_features(train_X, test_X)

    assert list(filtered_train.columns) == ["gene_a", "gene_c"]
    assert list(filtered_test.columns) == ["gene_a", "gene_c"]


def test_filter_non_expressed_features_keeps_gene_at_minimum_threshold():
    functions = load_selected_project_code(SOURCE_FILE)
    filter_non_expressed_features = functions["filter_non_expressed_features"]

    train_X = pd.DataFrame(
        {
            "one_nonzero_gene": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "zero_nonzero_gene": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )

    test_X = pd.DataFrame(
        {
            "one_nonzero_gene": [3, 4],
            "zero_nonzero_gene": [5, 6],
        }
    )

    filtered_train, filtered_test = filter_non_expressed_features(train_X, test_X)

    assert list(filtered_train.columns) == ["one_nonzero_gene"]
    assert list(filtered_test.columns) == ["one_nonzero_gene"]


def test_filter_non_expressed_features_uses_training_data_for_feature_selection():
    functions = load_selected_project_code(SOURCE_FILE)
    filter_non_expressed_features = functions["filter_non_expressed_features"]

    train_X = pd.DataFrame(
        {
            "expressed_in_train": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "only_expressed_in_test": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )

    test_X = pd.DataFrame(
        {
            "expressed_in_train": [0, 0],
            "only_expressed_in_test": [9, 10],
        }
    )

    filtered_train, filtered_test = filter_non_expressed_features(train_X, test_X)

    assert list(filtered_train.columns) == ["expressed_in_train"]
    assert list(filtered_test.columns) == ["expressed_in_train"]


def test_filter_non_expressed_features_keeps_train_and_test_columns_equal():
    functions = load_selected_project_code(SOURCE_FILE)
    filter_non_expressed_features = functions["filter_non_expressed_features"]

    train_X = pd.DataFrame(
        {
            "gene_a": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "gene_b": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "gene_c": [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
    )

    test_X = pd.DataFrame(
        {
            "gene_a": [2, 3],
            "gene_b": [4, 5],
            "gene_c": [6, 7],
        }
    )

    filtered_train, filtered_test = filter_non_expressed_features(train_X, test_X)

    assert list(filtered_train.columns) == list(filtered_test.columns)


def test_apply_variance_filter_removes_constant_gene():
    functions = load_selected_project_code(SOURCE_FILE)
    apply_variance_filter = functions["apply_variance_filter"]

    train_X = pd.DataFrame(
        {
            "constant_gene": [1, 1, 1, 1],
            "variable_gene": [1, 2, 1, 3],
        }
    )

    test_X = pd.DataFrame(
        {
            "constant_gene": [1, 1],
            "variable_gene": [2, 3],
        }
    )

    filtered_train, filtered_test = apply_variance_filter(train_X, test_X)

    assert list(filtered_train.columns) == ["variable_gene"]
    assert list(filtered_test.columns) == ["variable_gene"]


def test_apply_variance_filter_keeps_column_order():
    functions = load_selected_project_code(SOURCE_FILE)
    apply_variance_filter = functions["apply_variance_filter"]

    train_X = pd.DataFrame(
        {
            "gene_first": [1, 2, 3, 4],
            "constant_gene": [5, 5, 5, 5],
            "gene_second": [4, 3, 2, 1],
        }
    )

    test_X = pd.DataFrame(
        {
            "gene_first": [5, 6],
            "constant_gene": [5, 5],
            "gene_second": [0, 1],
        }
    )

    filtered_train, filtered_test = apply_variance_filter(train_X, test_X)

    assert list(filtered_train.columns) == ["gene_first", "gene_second"]
    assert list(filtered_test.columns) == ["gene_first", "gene_second"]


def test_select_features_applies_existing_filters_in_order():
    functions = load_selected_project_code(SOURCE_FILE)
    select_features = functions["select_features"]

    train_X = pd.DataFrame(
        {
            "not_expressed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "constant_gene": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "useful_gene": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    test_X = pd.DataFrame(
        {
            "not_expressed": [0, 0],
            "constant_gene": [1, 1],
            "useful_gene": [11, 12],
        }
    )

    filtered_train, filtered_test = select_features(train_X, test_X)

    assert list(filtered_train.columns) == ["useful_gene"]
    assert list(filtered_test.columns) == ["useful_gene"]


def test_select_features_preserves_remaining_feature_order():
    functions = load_selected_project_code(SOURCE_FILE)
    select_features = functions["select_features"]

    train_X = pd.DataFrame(
        {
            "gene_first": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "not_expressed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "gene_second": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        }
    )

    test_X = pd.DataFrame(
        {
            "gene_first": [11, 12],
            "not_expressed": [0, 0],
            "gene_second": [13, 14],
        }
    )

    filtered_train, filtered_test = select_features(train_X, test_X)

    assert list(filtered_train.columns) == ["gene_first", "gene_second"]
    assert list(filtered_test.columns) == ["gene_first", "gene_second"]


def test_select_features_does_not_modify_original_dataframes():
    functions = load_selected_project_code(SOURCE_FILE)
    select_features = functions["select_features"]

    train_X = pd.DataFrame(
        {
            "not_expressed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "useful_gene": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    test_X = pd.DataFrame(
        {
            "not_expressed": [0, 0],
            "useful_gene": [11, 12],
        }
    )

    original_train_columns = list(train_X.columns)
    original_test_columns = list(test_X.columns)

    select_features(train_X, test_X)

    assert list(train_X.columns) == original_train_columns
    assert list(test_X.columns) == original_test_columns


def test_plot_class_distribution_runs_and_sets_title_and_axis_labels(monkeypatch):
    functions = load_selected_project_code(SOURCE_FILE)
    plot_class_distribution = functions["plot_class_distribution"]

    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")

    y = pd.Series(["A", "A", "B", "C"])
    plot_class_distribution(y, title="Class Distribution Test", x_label="Class")

    ax = plt.gca()

    assert ax.get_title() == "Class Distribution Test"
    assert ax.get_xlabel() == "Class"
    assert ax.get_ylabel() == "Number of samples"


def test_plot_pca_runs_and_sets_title_and_axis_labels(monkeypatch):
    functions = load_selected_project_code(SOURCE_FILE)
    plot_pca = functions["plot_pca"]

    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")

    X = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0, 3.0, 4.0],
            "gene_b": [2.0, 3.0, 4.0, 5.0],
            "gene_c": [5.0, 4.0, 3.0, 2.0],
        }
    )
    y = pd.Series(["A", "A", "B", "B"])

    plot_pca(X, y, title="PCA Test")

    ax = plt.gca()

    assert ax.get_title() == "PCA Test"
    assert ax.get_xlabel() == "PC1"
    assert ax.get_ylabel() == "PC2"


def test_plot_correlation_heatmap_runs_and_sets_title(monkeypatch):
    functions = load_selected_project_code(SOURCE_FILE)
    plot_correlation_heatmap = functions["plot_correlation_heatmap"]

    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")

    X = pd.DataFrame(
        {
            "gene_a": [1.0, 2.0, 3.0, 4.0],
            "gene_b": [4.0, 3.0, 2.0, 1.0],
            "gene_c": [1.0, 3.0, 2.0, 5.0],
            "gene_d": [2.0, 2.5, 3.0, 3.5],
        }
    )

    plot_correlation_heatmap(X, title="Correlation Heatmap Test")

    ax = plt.gca()

    assert ax.get_title() == "Correlation Heatmap Test"
