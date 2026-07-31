from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260730
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "synthetic" / "synthetic_asset_events.csv"


def generate_synthetic_data(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    historical_assets: int = 35,
    contemporary_assets: int = 25,
) -> pd.DataFrame:
    """Create a deterministic, non-identifying event log for demonstration."""
    rng = np.random.default_rng(SEED)
    records: list[dict[str, object]] = []
    total_assets = historical_assets + contemporary_assets

    for asset_index in range(total_assets):
        asset_code = f"SYN-{asset_index + 1:04d}"
        group_index = asset_index % 4
        group = f"Synthetic group {chr(65 + group_index)}"
        power = float(rng.choice([7.5, 11.0, 15.0, 22.0, 30.0, 45.0]))
        voltage = float(rng.choice([220.0, 380.0, 440.0]))
        current = float(
            max(1.0, power * 1000 / (np.sqrt(3) * voltage * 0.86))
        )

        if asset_index < historical_assets:
            start = pd.Timestamp("2015-01-01") + pd.Timedelta(
                days=int(rng.integers(0, 500))
            )
            failure_gap_range = (260, 480)
        else:
            start = pd.Timestamp("2022-01-01") + pd.Timedelta(
                days=int(rng.integers(0, 300))
            )
            failure_gap_range = (220, 420)

        event_date = start
        event_labels = [1, 0, 1, 0, 1, 0]
        for event_number, failure_label in enumerate(event_labels):
            if event_number > 0:
                if failure_label == 1:
                    gap_days = int(rng.integers(*failure_gap_range))
                else:
                    gap_days = int(rng.integers(45, 150))
                event_date += pd.Timedelta(days=gap_days)

            repair_hours = float(
                np.round(
                    rng.gamma(2.1, 2.0)
                    + (2.0 if failure_label else 0.5)
                    + power / 40,
                    2,
                )
            )
            records.append(
                {
                    "GRUPO": group,
                    "MOTOR": asset_code,
                    "Nombre": f"Synthetic motor {asset_index + 1:04d}",
                    "Fecha": event_date.date().isoformat(),
                    "Falla": failure_label,
                    "Mantenimiento": 1 - failure_label,
                    "Tiempo de Reparacion (h)": repair_hours,
                    "Potencia": power,
                    "Voltaje (V)": voltage,
                    "Corriente (A)": round(current, 2),
                }
            )

    frame = pd.DataFrame.from_records(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


if __name__ == "__main__":
    generated = generate_synthetic_data()
    print(f"Wrote {len(generated)} synthetic records to {DEFAULT_OUTPUT}")
