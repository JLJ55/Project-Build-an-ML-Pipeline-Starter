#!/usr/bin/env python
"""
This script trains a Random Forest
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import wandb


def delta_date_feature(dates):
    """
    Given a 2D array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column.
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() - d).dt.days, axis=0).to_numpy()


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def validate_and_correct_types(df):
    """
    Ensures all columns in the dataframe have types compatible with MLflow.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            # Attempt conversion to string
            try:
                df[col] = df[col].astype(str)
            except Exception as e:
                raise ValueError(f"Cannot convert column '{col}' to string. Error: {e}")
        elif df[col].dtype.name not in ["int64", "float64", "bool"]:
            raise ValueError(f"Column '{col}' has unsupported type '{df[col].dtype}' for MLflow.")
    return df


def go(args):
    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # Load Random Forest configuration
    with open(args.rf_config) as fp:
        rf_config = json.load(fp)
    run.config.update(rf_config)

    # Fix the random seed for reproducibility
    rf_config['random_state'] = args.random_seed

    # Load the training dataset artifact
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()
    X = pd.read_csv(trainval_local_path)
    y = X.pop("price")

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by] if args.stratify_by != "none" else None, random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")
    sk_pipe, processed_features = get_inference_pipeline(rf_config, args.max_tfidf_features)

    # Fit the pipeline
    logger.info("Fitting")
    sk_pipe.fit(X_train, y_train)

    # Evaluate the model
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)
    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    # Ensure column types are valid for MLflow
    logger.info("Validating data types for MLflow")
    X_val = validate_and_correct_types(X_val)

    # Save and log the model
    logger.info("Exporting model")
    if os.path.exists("random_forest_dir"):
        shutil.rmtree("random_forest_dir")

    signature = mlflow.models.infer_signature(X_val, y_pred)
    mlflow.sklearn.save_model(
        sk_pipe,
        "random_forest_dir",
        signature=signature,
        input_example=X_train.iloc[:5],
    )

    artifact = wandb.Artifact(
        args.output_artifact,
        type='model_export',
        description='Trained Random Forest model',
        metadata=rf_config
    )
    artifact.add_dir('random_forest_dir')
    run.log_artifact(artifact)

    # Plot feature importance
    fig_feat_imp = plot_feature_importance(sk_pipe, processed_features)

    # Log metrics and feature importance visualization to W&B
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae
    run.log({"feature_importance": wandb.Image(fig_feat_imp)})

    run.finish()


def plot_feature_importance(pipe, feat_names):
    # Collect the feature importance for all non-NLP features
    feat_imp = pipe["random_forest"].feature_importances_[:len(feat_names) - 1]
    # NLP feature importance
    nlp_importance = sum(pipe["random_forest"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)

    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(rf_config, max_tfidf_features):
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]

    ordinal_categorical_preproc = OrdinalEncoder()

    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OrdinalEncoder()
    )

    zero_imputed = [
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    date_imputer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, check_inverse=False, validate=False)
    )

    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(binary=False, max_features=max_tfidf_features, stop_words='english'),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat", non_ordinal_categorical_preproc, non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"]),
        ],
        remainder="drop",
    )

    processed_features = ordinal_categorical + non_ordinal_categorical + zero_imputed + ["last_review", "name"]

    random_forest = RandomForestRegressor(**rf_config)

    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("random_forest", random_forest),
        ]
    )

    return sk_pipe, processed_features


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Random Forest Model")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
             "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words to consider for the TFIDF",
        default=10,
        type=int
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)


