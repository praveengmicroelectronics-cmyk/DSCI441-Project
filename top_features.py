"""
top_features.py
===============
Select the top-N most informative features from the full wafer feature matrix.

Feature importance is estimated via a RandomForest trained on an 80/20
stratified split.  Only the training labels are used — test labels are never
seen during selection, preventing data leakage.

The returned DataFrame is passed to all downstream models in Milestone1.py.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def select_top_features(
    features_df: pd.DataFrame,
    y,
    n_top: int       = 10,
    random_state: int = 42,
) -> tuple:
    """
    Rank all features by RF mean decrease in impurity and return the top-N subset.

    Parameters
    ----------
    features_df  : pd.DataFrame
        Full feature matrix (e.g. 34 columns from Feature_Extraction).
        NaN and ±inf values are replaced with 0 before fitting.
    y            : array-like
        Class labels aligned with features_df rows.
    n_top        : int
        Number of top features to retain (default 10).
    random_state : int
        RNG seed for the train/test split and RandomForest (default 42).

    Returns
    -------
    top_df    : pd.DataFrame
        Only the top-N columns, preserving the original row index.
    top_names : list[str]
        Feature column names ordered by importance descending.
    """
    X     = features_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    y_arr = np.array(y).astype(str)

    X_tr, _, y_tr, _ = train_test_split(
        X, y_arr, test_size=0.2,
        stratify=y_arr, random_state=random_state,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_tr, y_tr)

    importance = pd.Series(rf.feature_importances_, index=X.columns)
    top_names  = importance.nlargest(n_top).index.tolist()

    print(f"[top_features] Top {n_top} features (RF importance, train split only):")
    for rank, (feat, imp) in enumerate(importance.nlargest(n_top).items(), start=1):
        print(f"  {rank:2d}. {feat:<35s}  {imp:.4f}")

    return X[top_names].copy(), top_names


def select_top_features_pca_lda(
    features_df: pd.DataFrame,
    y,
    n_top: int        = 10,
    random_state: int = 42,
) -> tuple:
    """
    Rank features by PCA variance-weighted importance and return top-N.

    Importance of feature j is defined as:

        importance_j = sum_k ( explained_variance_ratio_k * loading_{kj}^2 )

    This measures the share of the dataset's captured variance attributable to
    each original feature, aggregated across all principal components.
    StandardScaler is applied first so all features are on the same scale;
    both the scaler and PCA are fit on the training split only to prevent
    data leakage.

    Parameters
    ----------
    features_df  : pd.DataFrame   Full feature matrix. NaN/inf → 0.
    y            : array-like     Class labels (used only for stratified split).
    n_top        : int            Number of top features to return.
    random_state : int            RNG seed.

    Returns
    -------
    top_df            : pd.DataFrame  Only the top-N feature columns.
    top_names         : list[str]     Feature names, descending importance.
    importance_series : pd.Series     PCA importance for every feature.
    """
    X     = features_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    y_arr = np.array(y).astype(str)

    X_tr, _, _, _ = train_test_split(
        X, y_arr, test_size=0.2,
        stratify=y_arr, random_state=random_state,
    )

    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)

    pca = PCA(random_state=random_state)
    pca.fit(X_tr_s)

    # loadings: shape (n_pcs, n_features);  ev_ratio: (n_pcs,)
    ev_ratio   = pca.explained_variance_ratio_
    loadings   = pca.components_
    importance = np.sum(ev_ratio[:, None] * loadings ** 2, axis=0)

    importance_series = pd.Series(importance, index=X.columns)
    top_names         = importance_series.nlargest(n_top).index.tolist()

    print(f"[top_features_pca_lda] Top {n_top} features (PCA importance, train split only):")
    for rank, (feat, imp) in enumerate(importance_series.nlargest(n_top).items(), start=1):
        print(f"  {rank:2d}. {feat:<35s}  {imp:.5f}")

    return X[top_names].copy(), top_names, importance_series
