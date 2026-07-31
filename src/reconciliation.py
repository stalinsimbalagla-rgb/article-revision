from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pipeline import build_conditioned_intervals, normalize_code


DATE_COLUMNS = list(range(17, 47, 2))
NOTE_COLUMNS = list(range(18, 47, 2))


def load_historical_matrix(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = pd.read_excel(path, sheet_name="Inventory", header=None)
    asset_rows = raw.iloc[4:].copy()
    asset_rows = asset_rows[asset_rows[1].notna()].copy()
    asset_rows = asset_rows[
        ~asset_rows[1].astype(str).str.strip().str.upper().str.startswith("GRUPO")
    ].copy()
    asset_rows["codigo_normalizado"] = asset_rows[1].map(normalize_code)

    date_records: list[dict[str, object]] = []
    nonempty_date_cells = 0
    invalid_date_cells: list[str] = []
    for _, row in asset_rows.iterrows():
        code = row["codigo_normalizado"]
        for column in DATE_COLUMNS:
            value = row[column]
            if pd.isna(value):
                continue
            nonempty_date_cells += 1
            parsed = pd.to_datetime(value, dayfirst=True, errors="coerce")
            if pd.isna(parsed) or not 2010 <= parsed.year <= 2026:
                invalid_date_cells.append(str(value))
                continue
            date_records.append(
                {
                    "MOTOR_NORMALIZADO": code,
                    "Fecha": parsed.normalize(),
                }
            )

    note_values: list[object] = []
    for _, row in asset_rows.iterrows():
        for column in NOTE_COLUMNS:
            if pd.notna(row[column]):
                note_values.append(row[column])
    substantive_notes = [
        value
        for value in note_values
        if isinstance(value, str) and value.strip().upper() not in {"S.N."}
    ]

    dates = pd.DataFrame(date_records)
    summary = {
        "filas_activos": int(len(asset_rows)),
        "codigos_unicos": int(asset_rows["codigo_normalizado"].nunique()),
        "celdas_fecha_no_vacias": int(nonempty_date_cells),
        "fechas_interpretables": int(len(dates)),
        "fechas_no_interpretables": int(
            nonempty_date_cells - len(dates)
        ),
        "detalle_no_interpretable": invalid_date_cells,
        "codigos_con_fecha": int(dates["MOTOR_NORMALIZADO"].nunique()),
        "fecha_min": dates["Fecha"].min().date().isoformat(),
        "fecha_max": dates["Fecha"].max().date().isoformat(),
        "celdas_nota_no_vacias": int(len(note_values)),
        "notas_sustantivas": int(len(substantive_notes)),
        "tipo_intervencion_completo": 0,
    }
    return dates, summary


def reconcile(
    processed: pd.DataFrame,
    historical_dates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    historical_pairs = set(
        zip(
            historical_dates["MOTOR_NORMALIZADO"],
            historical_dates["Fecha"],
        )
    )
    processed = processed.copy()
    processed["coincidencia_exacta"] = [
        (code, date) in historical_pairs
        for code, date in zip(
            processed["MOTOR_NORMALIZADO"],
            processed["Fecha"],
        )
    ]

    processed_codes = set(processed["MOTOR_NORMALIZADO"])
    historical_codes = set(historical_dates["MOTOR_NORMALIZADO"])
    exact = processed[processed["coincidencia_exacta"]].copy()
    exact_intervals = build_conditioned_intervals(exact)

    summary = {
        "codigos_procesados": int(len(processed_codes)),
        "codigos_historicos": int(len(historical_codes)),
        "codigos_coincidentes": int(
            len(processed_codes & historical_codes)
        ),
        "pares_procesados": int(len(processed)),
        "pares_coincidentes": int(processed["coincidencia_exacta"].sum()),
        "pares_sin_coincidencia": int(
            (~processed["coincidencia_exacta"]).sum()
        ),
        "fallas_etiquetadas_en_coincidencias": int(exact["Falla"].sum()),
        "mantenimientos_etiquetados_en_coincidencias": int(
            exact["Mantenimiento"].sum()
        ),
        "etiquetas_falla_corrobables_semanticamente": 0,
        "intervalos_restringidos": int(len(exact_intervals)),
        "motores_en_intervalos_restringidos": int(
            exact_intervals["MOTOR_NORMALIZADO"].nunique()
        ),
        "mediana_restringida_h": float(
            exact_intervals["objetivo_h"].median()
        ),
        "media_restringida_h": float(
            exact_intervals["objetivo_h"].mean()
        ),
    }
    reconciliation_table = processed[
        [
            "MOTOR_NORMALIZADO",
            "Fecha",
            "Falla",
            "Mantenimiento",
            "coincidencia_exacta",
        ]
    ].copy()
    return exact, summary, reconciliation_table
