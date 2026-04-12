"""Benchmark harness: time each engine × transformation, produce comparison table."""

import argparse
import csv
import os
import time
import statistics

from tabulate import tabulate

from download_data import download, PARQUET_PATH
from engines import PandasEngine, CudfEngine, PolarsGPUEngine

TRANSFORMS = ["filter_compute", "groupby_agg", "window_rank"]


def time_call(fn, iterations: int):
    """Run fn `iterations` times, return list of elapsed seconds."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return times


def run_benchmarks(iterations: int = 5, warmup: int = 1):
    download()

    engines = []
    for cls in [PandasEngine, CudfEngine, PolarsGPUEngine]:
        try:
            eng = cls(PARQUET_PATH)
            engines.append(eng)
            print(f"  Loaded {eng.name}")
        except Exception as e:
            print(f"  SKIP {cls.name}: {e}")

    rows = []

    for eng in engines:
        total_median = 0.0
        for tf in TRANSFORMS:
            fn = getattr(eng, tf)

            # warmup (excluded from timing)
            for _ in range(warmup):
                fn()

            times = time_call(fn, iterations)
            med = statistics.median(times)
            total_median += med

            rows.append({
                "Engine": eng.name,
                "Transform": tf,
                "Median (s)": f"{med:.4f}",
                "Mean (s)": f"{statistics.mean(times):.4f}",
                "Std (s)": f"{statistics.stdev(times):.4f}" if len(times) > 1 else "N/A",
                "Min (s)": f"{min(times):.4f}",
                "Max (s)": f"{max(times):.4f}",
            })

        # total pipeline row
        rows.append({
            "Engine": eng.name,
            "Transform": "== TOTAL ==",
            "Median (s)": f"{total_median:.4f}",
            "Mean (s)": "",
            "Std (s)": "",
            "Min (s)": "",
            "Max (s)": "",
        })

    print("\n" + tabulate(rows, headers="keys", tablefmt="github"))

    # save to CSV
    out_path = os.path.join(os.path.dirname(__file__), "results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataframe engine benchmark")
    parser.add_argument("-n", "--iterations", type=int, default=5, help="Timed iterations per transform")
    parser.add_argument("-w", "--warmup", type=int, default=1, help="Warmup iterations (excluded from timing)")
    args = parser.parse_args()
    run_benchmarks(iterations=args.iterations, warmup=args.warmup)
