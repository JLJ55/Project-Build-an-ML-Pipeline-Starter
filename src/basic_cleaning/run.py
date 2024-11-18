#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd
import numpy as np

# DO NOT MODIFY
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# DO NOT MODIFY
def go(args):
    logger.info("Starting wandb run.")
    run = wandb.init(
        project="nyc_airbnb",
        group="basic_cleaning",
        job_type="basic_cleaning",
    )
    run.config.update(args)

    # Download input artifact
    logger.info("Fetching raw dataset.")
    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    # Data Cleaning
    logger.info("Cleaning data.")

    # Filter by price range
    logger.info("Filtering data by price range.")
    idx_price = df["price"].between(float(args.min_price), float(args.max_price))
    if np.sum(~idx_price) > 0:
        logger.warning(f"Found {np.sum(~idx_price)} rows outside the price range.")
    df = df[idx_price].copy()

    # Convert `last_review` to datetime
    logger.info("Converting `last_review` to datetime format.")
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Filter by geographical boundaries
    logger.info("Filtering data for valid geographical boundaries.")
    idx_geo = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    invalid_rows = df.loc[~idx_geo, ["id", "longitude", "latitude"]]
    
    if not invalid_rows.empty:
        logger.warning(f"Found {len(invalid_rows)} rows outside geographical boundaries.")
        logger.warning("Logging invalid rows:")
        logger.warning(invalid_rows)
    
    # Drop invalid rows
    df = df[idx_geo].copy()

    # Final Validation
    final_invalid_geo = df[~(df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2))]
    if not final_invalid_geo.empty:
        logger.error("Rows still outside valid geographical boundaries after filtering:")
        logger.error(final_invalid_geo)
        raise ValueError("Filtering failed to remove all invalid geographical entries.")

    # Save the cleaned data
    logger.info("Saving and exporting cleaned data.")
    df.to_csv("clean_sample.csv", index=False)

    # Log the cleaned data artifact to W&B
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    logger.info("Cleaning step completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="The input artifact containing the raw dataset",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The name of the output artifact containing the cleaned dataset",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="The type of the output artifact (e.g., 'clean_sample')",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="A description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="The minimum price for filtering listings",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="The maximum price for filtering listings",
        required=True,
    )

    args = parser.parse_args()

    go(args)

