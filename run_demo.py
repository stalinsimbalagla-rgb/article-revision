from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

from src.pipeline import (
    bootstrap_grouped_r2,
    build_conditioned_intervals,
    grouped_evaluation,
    label_sensitivity,
    load_processed,
    objective_summary,
    plot_model_comparison,
    plot_sensitivity,
    source_quality,
    summarize_sensitivity,
    temporal_evaluation,
    write_json,
)
from src.synthetic_data import DEFAULT_OUTPUT, generate_synthetic_data


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the public workflow on fully synthetic event data."
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Perturbation repetitions per rate and direction (default: 3).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Asset-grouped bootstrap repetitions (default: 200).",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the deterministic synthetic dataset before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repetitions < 1 or args.bootstrap < 10:
        raise ValueError("Use at least 1 perturbation and 10 bootstrap repetitions.")

    if args.regenerate or not DEFAULT_OUTPUT.exists():
        generate_synthetic_data()

    output_dir = ROOT / "demo_outputs"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    processed = load_processed(DEFAULT_OUTPUT)
    intervals = build_conditioned_intervals(processed)
    write_json(output_dir / "01_source_quality.json", source_quality(processed))
    write_json(output_dir / "02_target_summary.json", objective_summary(intervals))

    grouped, predictions = grouped_evaluation(intervals, include_calendar=True)
    operational, _ = grouped_evaluation(intervals, include_calendar=False)
    temporal = temporal_evaluation(intervals)
    grouped.to_csv(output_dir / "03_grouped_models.csv", index=False)
    operational.to_csv(output_dir / "04_models_without_calendar.csv", index=False)
    temporal.to_csv(output_dir / "05_temporal_holdout.csv", index=False)

    confidence_interval = bootstrap_grouped_r2(
        intervals,
        predictions["Gradient boosting"],
        repetitions=args.bootstrap,
    )
    write_json(
        output_dir / "06_bootstrap_r2.json",
        {
            "model": "Gradient boosting",
            "method": "asset-grouped bootstrap of out-of-fold predictions",
            "repetitions": args.bootstrap,
            "r2_95_interval": list(confidence_interval),
        },
    )
    plot_model_comparison(grouped, figures_dir / "grouped_models.png")

    sensitivity_records = label_sensitivity(
        processed,
        repetitions=args.repetitions,
    )
    sensitivity_summary = summarize_sensitivity(sensitivity_records)
    sensitivity_records.to_csv(
        output_dir / "07_label_sensitivity_runs.csv",
        index=False,
    )
    sensitivity_summary.to_csv(
        output_dir / "08_label_sensitivity_summary.csv",
        index=False,
    )
    base_r2 = float(
        grouped.loc[
            grouped["modelo"].eq("Gradient boosting"),
            "r2",
        ].iloc[0]
    )
    plot_sensitivity(
        len(intervals),
        base_r2,
        sensitivity_summary,
        figures_dir / "label_sensitivity.png",
    )
    print(f"Synthetic demonstration completed: {output_dir}")


if __name__ == "__main__":
    main()
