"""Download the NYC Yellow Taxi trip dataset (Jan 2024 parquet, ~45MB, ~3M rows)."""

import os
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PARQUET_PATH = os.path.join(DATA_DIR, "yellow_tripdata_2024-01.parquet")
URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(PARQUET_PATH):
        print(f"Dataset already exists at {PARQUET_PATH}")
        return PARQUET_PATH
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, PARQUET_PATH)
    print(f"Saved to {PARQUET_PATH}")
    return PARQUET_PATH


if __name__ == "__main__":
    download()
