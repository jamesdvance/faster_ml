"""Three dataframe engines that each implement identical transformations."""

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Raw Pandas (CPU)
# ---------------------------------------------------------------------------
class PandasEngine:
    name = "Pandas (CPU)"

    def __init__(self, parquet_path: str):
        self.df = pd.read_parquet(parquet_path)

    def filter_compute(self):
        """Filter trips > 2 miles, compute tip %, keep rows where tip > 20%."""
        df = self.df[self.df["trip_distance"] > 2].copy()
        df["tip_pct"] = df["tip_amount"] / df["fare_amount"]
        return df[df["tip_pct"] > 0.20]

    def groupby_agg(self):
        """Group by pickup location: mean fare, total tips, count, max distance."""
        return self.df.groupby("PULocationID").agg(
            mean_fare=("fare_amount", "mean"),
            total_tips=("tip_amount", "sum"),
            trip_count=("fare_amount", "count"),
            max_distance=("trip_distance", "max"),
        )

    def window_rank(self):
        """Rank trips by fare within each pickup location, keep top 5 per group."""
        df = self.df.copy()
        df["rank"] = df.groupby("PULocationID")["fare_amount"].rank(
            method="first", ascending=False
        )
        return df[df["rank"] <= 5]


# ---------------------------------------------------------------------------
# 2. cuDF via RAPIDS (GPU)
# ---------------------------------------------------------------------------
class CudfEngine:
    name = "cuDF (GPU)"

    def __init__(self, parquet_path: str):
        import cudf

        self.df = cudf.read_parquet(parquet_path)

    def filter_compute(self):
        import cudf

        df = self.df[self.df["trip_distance"] > 2]
        df["tip_pct"] = df["tip_amount"] / df["fare_amount"]
        return df[df["tip_pct"] > 0.20]

    def groupby_agg(self):
        return self.df.groupby("PULocationID").agg(
            {"fare_amount": ["mean", "count"], "tip_amount": "sum", "trip_distance": "max"}
        )

    def window_rank(self):
        df = self.df.copy()
        df["rank"] = df.groupby("PULocationID")["fare_amount"].rank(
            method="first", ascending=False
        )
        return df[df["rank"] <= 5]


# ---------------------------------------------------------------------------
# 3. Polars with GPU engine (uses cuDF backend)
# ---------------------------------------------------------------------------
class PolarsGPUEngine:
    name = "Polars GPU"

    def __init__(self, parquet_path: str):
        import polars as pl

        self.lf = pl.scan_parquet(parquet_path)

    def filter_compute(self):
        import polars as pl

        return (
            self.lf.filter(pl.col("trip_distance") > 2)
            .with_columns((pl.col("tip_amount") / pl.col("fare_amount")).alias("tip_pct"))
            .filter(pl.col("tip_pct") > 0.20)
            .collect(engine="gpu")
        )

    def groupby_agg(self):
        import polars as pl

        return (
            self.lf.group_by("PULocationID")
            .agg(
                pl.col("fare_amount").mean().alias("mean_fare"),
                pl.col("tip_amount").sum().alias("total_tips"),
                pl.col("fare_amount").count().alias("trip_count"),
                pl.col("trip_distance").max().alias("max_distance"),
            )
            .collect(engine="gpu")
        )

    def window_rank(self):
        import polars as pl

        return (
            self.lf.with_columns(
                pl.col("fare_amount")
                .rank(method="ordinal", descending=True)
                .over("PULocationID")
                .alias("rank")
            )
            .filter(pl.col("rank") <= 5)
            .collect(engine="gpu")
        )
