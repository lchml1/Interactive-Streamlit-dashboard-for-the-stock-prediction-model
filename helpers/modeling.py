# helpers/modeling.py
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _get_models():
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=4,
                        random_state=42,
                        class_weight="balanced",
                    ),
                )
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("model", GradientBoostingClassifier(random_state=42))
            ]
        ),
    }


def _extract_feature_importance(trained_pipeline, feature_cols):
    model = trained_pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        values = np.zeros(len(feature_cols))

    fi_df = pd.DataFrame(
        {"feature": feature_cols, "importance": values}
    ).sort_values("importance", ascending=False)
    return fi_df.reset_index(drop=True)


def train_and_evaluate_models(dataset: pd.DataFrame, feature_cols: list[str]):
    X = dataset[feature_cols].copy()
    y = dataset["target"].copy()

    if len(X) < 120:
        raise ValueError("Not enough data for robust time-series validation.")

    # 僅用 TimeSeriesSplit，避免亂用 random split 造成 data leakage
    n_splits = 5 if len(X) >= 320 else 4 if len(X) >= 220 else 3
    tscv = TimeSeriesSplit(n_splits=n_splits)

    models = _get_models()
    comparison_rows = []
    results = {}

    for model_name, base_model in models.items():
        oof_pred = np.full(len(X), np.nan)
        oof_proba = np.full(len(X), np.nan)
        fold_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = clone(base_model)
            model.fit(X_train, y_train)

            proba_up = model.predict_proba(X_test)[:, 1]
            pred = (proba_up >= 0.5).astype(int)

            oof_pred[test_idx] = pred
            oof_proba[test_idx] = proba_up
            fold_scores.append(accuracy_score(y_test, pred))

        valid = ~np.isnan(oof_pred)
        y_true = y.iloc[valid]
        y_hat = oof_pred[valid].astype(int)

        acc = accuracy_score(y_true, y_hat)
        precision = precision_score(y_true, y_hat, zero_division=0)
        recall = recall_score(y_true, y_hat, zero_division=0)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        cm = confusion_matrix(y_true, y_hat)

        comparison_rows.append(
            {
                "model": model_name,
                "cv_accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        results[model_name] = {
            "oof_pred": oof_pred,
            "oof_proba": oof_proba,
            "metrics": {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": cm,
            },
            "mean_fold_accuracy": float(np.mean(fold_scores)),
        }

    comparison_df = pd.DataFrame(comparison_rows).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["model"]

    final_model = clone(models[best_model_name])
    final_model.fit(X, y)

    feature_importance = _extract_feature_importance(final_model, feature_cols)

    return {
        "model": final_model,
        "best_model_name": best_model_name,
        "best_metrics": results[best_model_name]["metrics"],
        "best_oof_proba": results[best_model_name]["oof_proba"],
        "comparison_df": comparison_df,
        "feature_importance": feature_importance,
    }
