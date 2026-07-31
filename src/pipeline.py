from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 42
CUTOFF_DATE = pd.Timestamp("2022-01-01")

REQUIRED_COLUMNS = [
    "GRUPO",
    "MOTOR",
    "Nombre",
    "Fecha",
    "Falla",
    "Mantenimiento",
    "Tiempo de Reparacion (h)",
    "Potencia",
    "Voltaje (V)",
    "Corriente (A)",
]

OPERATIONAL_FEATURES = [
    "Potencia",
    "Voltaje (V)",
    "Corriente (A)",
    "Tiempo de Reparacion (h)",
    "n_registros_observados",
    "n_mantenimientos_observados",
    "h_desde_registro_previo",
    "h_desde_falla_previa",
    "reparacion_media_historica",
    "mtbf_historico_condicionado",
    "exposicion_h",
]

CALENDAR_FEATURES = ["anio", "mes_seno", "mes_coseno"]
CATEGORICAL_FEATURES = ["GRUPO"]


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    n: int
    mae_h: float
    rmse_h: float
    r2: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "modelo": self.name,
            "n": self.n,
            "mae_h": self.mae_h,
            "rmse_h": self.rmse_h,
            "r2": self.r2,
        }


def normalize_code(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def load_processed(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Columnas obligatorias ausentes: {missing}")

    frame = frame[REQUIRED_COLUMNS].copy()
    frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="raise").dt.normalize()
    frame["MOTOR_NORMALIZADO"] = frame["MOTOR"].map(normalize_code)

    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("La base procesada contiene valores faltantes.")
    if not frame["Falla"].isin([0, 1]).all():
        raise ValueError("Falla debe ser binaria.")
    if not frame["Mantenimiento"].isin([0, 1]).all():
        raise ValueError("Mantenimiento debe ser binario.")
    if not (frame["Falla"] + frame["Mantenimiento"]).eq(1).all():
        raise ValueError("Cada fila debe tener exactamente una etiqueta activa.")
    return frame


def source_quality(frame: pd.DataFrame) -> dict[str, object]:
    ordered = frame.copy()
    ordered["fecha_siguiente_orden_archivo"] = ordered.groupby("MOTOR")[
        "Fecha"
    ].shift(-1)
    negative_mask = (
        ordered["fecha_siguiente_orden_archivo"].notna()
        & (
            ordered["fecha_siguiente_orden_archivo"]
            < ordered["Fecha"]
        )
    )
    negative_rows = ordered.loc[
        negative_mask,
        ["MOTOR", "Fecha", "fecha_siguiente_orden_archivo"],
    ]
    negative_records = [
        {
            "motor": row["MOTOR"],
            "fecha_actual": row["Fecha"].date().isoformat(),
            "fecha_siguiente": row["fecha_siguiente_orden_archivo"]
            .date()
            .isoformat(),
            "intervalo_h": float(
                (
                    row["fecha_siguiente_orden_archivo"] - row["Fecha"]
                ).total_seconds()
                / 3600
            ),
        }
        for _, row in negative_rows.iterrows()
    ]
    return {
        "registros": int(len(frame)),
        "motores": int(frame["MOTOR_NORMALIZADO"].nunique()),
        "fallas_etiquetadas": int(frame["Falla"].sum()),
        "mantenimientos_etiquetados": int(frame["Mantenimiento"].sum()),
        "fecha_min": frame["Fecha"].min().date().isoformat(),
        "fecha_max": frame["Fecha"].max().date().isoformat(),
        "celdas_faltantes": int(frame[REQUIRED_COLUMNS].isna().sum().sum()),
        "filas_duplicadas": int(frame[REQUIRED_COLUMNS].duplicated().sum()),
        "intervalos_negativos_en_orden_archivo": int(negative_mask.sum()),
        "detalle_intervalos_negativos": negative_records,
    }


def _feature_engineering_one_motor(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("Fecha").copy()
    group["n_registros_observados"] = np.arange(1, len(group) + 1)
    group["n_mantenimientos_observados"] = group["Mantenimiento"].cumsum()
    group["h_desde_registro_previo"] = (
        group["Fecha"].diff().dt.total_seconds() / 3600
    )

    previous_failure = (
        group["Fecha"].where(group["Falla"].eq(1)).ffill().shift(1)
    )
    group["h_desde_falla_previa"] = (
        group["Fecha"] - previous_failure
    ).dt.total_seconds() / 3600
    group["reparacion_media_historica"] = (
        group["Tiempo de Reparacion (h)"].expanding().mean()
    )

    failure_dates: list[pd.Timestamp] = []
    observed_intervals: list[float] = []
    historical_mtbf: list[float] = []
    for _, row in group.iterrows():
        if row["Falla"] == 1:
            if failure_dates:
                observed_intervals.append(
                    (row["Fecha"] - failure_dates[-1]).total_seconds() / 3600
                )
            failure_dates.append(row["Fecha"])
        historical_mtbf.append(
            float(np.mean(observed_intervals))
            if observed_intervals
            else np.nan
        )
    group["mtbf_historico_condicionado"] = historical_mtbf
    group["exposicion_h"] = (
        group["Fecha"] - group["Fecha"].min()
    ).dt.total_seconds() / 3600

    next_failure = (
        group["Fecha"]
        .where(group["Falla"].eq(1))
        .iloc[::-1]
        .shift(1)
        .ffill()
        .iloc[::-1]
    )
    group["fecha_falla_siguiente"] = next_failure
    group["objetivo_h"] = (
        group["fecha_falla_siguiente"] - group["Fecha"]
    ).dt.total_seconds() / 3600
    return group


def build_conditioned_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    engineered = pd.concat(
        [
            _feature_engineering_one_motor(group)
            for _, group in frame.groupby("MOTOR_NORMALIZADO", sort=False)
        ],
        ignore_index=True,
    )
    intervals = engineered[
        engineered["Falla"].eq(1)
        & engineered["objetivo_h"].notna()
        & engineered["objetivo_h"].gt(0)
    ].copy()
    intervals["anio"] = intervals["Fecha"].dt.year
    month = intervals["Fecha"].dt.month
    intervals["mes_seno"] = np.sin(2 * np.pi * month / 12)
    intervals["mes_coseno"] = np.cos(2 * np.pi * month / 12)
    return intervals


def objective_summary(intervals: pd.DataFrame) -> dict[str, float | int]:
    objective = intervals["objetivo_h"]
    return {
        "intervalos": int(len(intervals)),
        "motores": int(intervals["MOTOR_NORMALIZADO"].nunique()),
        "media_h": float(objective.mean()),
        "mediana_h": float(objective.median()),
        "desviacion_estandar_h": float(objective.std()),
        "q1_h": float(objective.quantile(0.25)),
        "q3_h": float(objective.quantile(0.75)),
    }


def _preprocessor(numeric_features: list[str], scaled: bool) -> ColumnTransformer:
    if scaled:
        numeric_pipeline: object = Pipeline(
            [
                ("imputacion", SimpleImputer(strategy="median")),
                ("escala", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = SimpleImputer(strategy="median")
    return ColumnTransformer(
        [
            ("numericas", numeric_pipeline, numeric_features),
            (
                "grupo",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def model_definitions() -> dict[str, object]:
    return {
        "Ridge": Ridge(alpha=10.0),
        "Random forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=SEED,
            n_jobs=-1,
        ),
        "Extra trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=SEED,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=5,
            random_state=SEED,
        ),
        "Hist. gradient boosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }


def _metrics(name: str, y_true: Iterable[float], prediction: Iterable[float]) -> EvaluationResult:
    y_array = np.asarray(list(y_true), dtype=float)
    p_array = np.asarray(list(prediction), dtype=float)
    return EvaluationResult(
        name=name,
        n=len(y_array),
        mae_h=float(mean_absolute_error(y_array, p_array)),
        rmse_h=float(np.sqrt(mean_squared_error(y_array, p_array))),
        r2=float(r2_score(y_array, p_array)),
    )


def grouped_evaluation(
    intervals: pd.DataFrame,
    *,
    include_calendar: bool,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    numeric_features = OPERATIONAL_FEATURES + (
        CALENDAR_FEATURES if include_calendar else []
    )
    feature_columns = numeric_features + CATEGORICAL_FEATURES
    X = intervals[feature_columns]
    y = intervals["objetivo_h"]
    # Se conserva el identificador suministrado para fijar exactamente la
    # asignación determinista de GroupKFold. La versión normalizada se usa
    # para reconciliar fuentes, no para reordenar los pliegues.
    groups = intervals["MOTOR"]
    cross_validation = GroupKFold(n_splits=5)

    baseline_prediction = np.empty(len(intervals))
    for train_index, test_index in cross_validation.split(X, y, groups):
        baseline_prediction[test_index] = float(y.iloc[train_index].median())

    results = [_metrics("Mediana", y, baseline_prediction).as_dict()]
    predictions: dict[str, np.ndarray] = {"Mediana": baseline_prediction}

    for name, estimator in model_definitions().items():
        pipeline = Pipeline(
            [
                (
                    "preparacion",
                    _preprocessor(
                        numeric_features=numeric_features,
                        scaled=name == "Ridge",
                    ),
                ),
                ("modelo", clone(estimator)),
            ]
        )
        prediction = cross_val_predict(
            pipeline,
            X,
            y,
            groups=groups,
            cv=cross_validation,
            n_jobs=-1,
        )
        results.append(_metrics(name, y, prediction).as_dict())
        predictions[name] = prediction
    return pd.DataFrame(results), predictions


def temporal_evaluation(intervals: pd.DataFrame) -> pd.DataFrame:
    numeric_features = OPERATIONAL_FEATURES + CALENDAR_FEATURES
    feature_columns = numeric_features + CATEGORICAL_FEATURES
    train = intervals[intervals["fecha_falla_siguiente"] < CUTOFF_DATE].copy()
    test = intervals[intervals["Fecha"] >= CUTOFF_DATE].copy()

    overlap = set(train["MOTOR"]) & set(test["MOTOR"])
    if overlap:
        raise AssertionError("La separación temporal contiene motores compartidos.")

    y_train = train["objetivo_h"]
    y_test = test["objetivo_h"]
    baseline = np.full(len(test), float(y_train.median()))
    results = [_metrics("Mediana", y_test, baseline).as_dict()]

    for name, estimator in model_definitions().items():
        pipeline = Pipeline(
            [
                (
                    "preparacion",
                    _preprocessor(
                        numeric_features=numeric_features,
                        scaled=name == "Ridge",
                    ),
                ),
                ("modelo", clone(estimator)),
            ]
        )
        pipeline.fit(train[feature_columns], y_train)
        prediction = pipeline.predict(test[feature_columns])
        results.append(_metrics(name, y_test, prediction).as_dict())

    output = pd.DataFrame(results)
    output["n_entrenamiento"] = len(train)
    output["motores_entrenamiento"] = train["MOTOR_NORMALIZADO"].nunique()
    output["n_prueba"] = len(test)
    output["motores_prueba"] = test["MOTOR_NORMALIZADO"].nunique()
    return output


def bootstrap_grouped_r2(
    intervals: pd.DataFrame,
    prediction: np.ndarray,
    *,
    repetitions: int = 2000,
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    groups = intervals["MOTOR"].to_numpy()
    unique_groups = np.unique(groups)
    y = intervals["objetivo_h"].to_numpy()
    values: list[float] = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        sampled_indices = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled_groups]
        )
        if np.var(y[sampled_indices]) == 0:
            continue
        values.append(r2_score(y[sampled_indices], prediction[sampled_indices]))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def _flip_labels(
    frame: pd.DataFrame,
    *,
    direction: str,
    rate: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    perturbed = frame.copy()
    if direction == "falla_a_mantenimiento":
        candidates = perturbed.index[perturbed["Falla"].eq(1)].to_numpy()
        n_flip = max(1, int(round(rate * len(candidates))))
        selected = rng.choice(candidates, size=n_flip, replace=False)
        perturbed.loc[selected, ["Falla", "Mantenimiento"]] = [0, 1]
    elif direction == "mantenimiento_a_falla":
        candidates = perturbed.index[perturbed["Mantenimiento"].eq(1)].to_numpy()
        n_flip = max(1, int(round(rate * len(candidates))))
        selected = rng.choice(candidates, size=n_flip, replace=False)
        perturbed.loc[selected, ["Falla", "Mantenimiento"]] = [1, 0]
    else:
        raise ValueError(f"Dirección no reconocida: {direction}")
    return perturbed


def _gradient_boosting_r2(intervals: pd.DataFrame) -> float:
    numeric_features = OPERATIONAL_FEATURES + CALENDAR_FEATURES
    feature_columns = numeric_features + CATEGORICAL_FEATURES
    pipeline = Pipeline(
        [
            (
                "preparacion",
                _preprocessor(numeric_features=numeric_features, scaled=False),
            ),
            ("modelo", clone(model_definitions()["Gradient boosting"])),
        ]
    )
    prediction = cross_val_predict(
        pipeline,
        intervals[feature_columns],
        intervals["objetivo_h"],
        groups=intervals["MOTOR"],
        cv=GroupKFold(n_splits=5),
        n_jobs=-1,
    )
    return float(r2_score(intervals["objetivo_h"], prediction))


def label_sensitivity(
    frame: pd.DataFrame,
    *,
    repetitions: int = 30,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for rate in (0.05, 0.10, 0.20):
        for direction_index, direction in enumerate(
            ("falla_a_mantenimiento", "mantenimiento_a_falla")
        ):
            for repetition in range(repetitions):
                rng = np.random.default_rng(
                    SEED + repetition + int(rate * 1000) + direction_index * 10000
                )
                perturbed = _flip_labels(
                    frame,
                    direction=direction,
                    rate=rate,
                    rng=rng,
                )
                intervals = build_conditioned_intervals(perturbed)
                records.append(
                    {
                        "tasa": rate,
                        "direccion": direction,
                        "repeticion": repetition + 1,
                        "intervalos": int(len(intervals)),
                        "motores": int(
                            intervals["MOTOR_NORMALIZADO"].nunique()
                        ),
                        "mediana_objetivo_h": float(
                            intervals["objetivo_h"].median()
                        ),
                        "r2_gradient_boosting": _gradient_boosting_r2(intervals),
                    }
                )
    return pd.DataFrame(records)


def summarize_sensitivity(records: pd.DataFrame) -> pd.DataFrame:
    grouped = records.groupby(["tasa", "direccion"])
    return grouped.agg(
        intervalos_mediana=("intervalos", "median"),
        intervalos_p05=("intervalos", lambda values: values.quantile(0.05)),
        intervalos_p95=("intervalos", lambda values: values.quantile(0.95)),
        r2_mediana=("r2_gradient_boosting", "median"),
        r2_p05=("r2_gradient_boosting", lambda values: values.quantile(0.05)),
        r2_p95=("r2_gradient_boosting", lambda values: values.quantile(0.95)),
    ).reset_index()


def plot_model_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    colors = ["#9AA5B1", "#487EAC", "#3D9389", "#DDBF77", "#C96D3D", "#9169DE"]
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.1))
    y_positions = np.arange(len(metrics))

    axes[0].barh(y_positions, metrics["mae_h"], color=colors)
    axes[0].set_yticks(y_positions, metrics["modelo"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE (h)")
    axes[0].set_title("Error absoluto medio")
    axes[0].grid(axis="x", alpha=0.25)
    for y_position, value in enumerate(metrics["mae_h"]):
        axes[0].text(value + 80, y_position, f"{value:,.0f}", va="center")

    axes[1].barh(y_positions, metrics["r2"], color=colors)
    axes[1].set_yticks(y_positions, metrics["modelo"])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel(r"$R^2$")
    axes[1].set_title("Varianza explicada")
    axes[1].grid(axis="x", alpha=0.25)
    for y_position, value in enumerate(metrics["r2"]):
        axes[1].text(
            value + (0.006 if value >= 0 else -0.045),
            y_position,
            f"{value:.3f}",
            va="center",
        )

    figure.suptitle(
        f"Validación cruzada agrupada por motor (n={int(metrics['n'].iloc[0])})",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_sensitivity(
    base_n: int,
    base_r2: float,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    directions = [
        ("falla_a_mantenimiento", "#D95F45", "F→M"),
        ("mantenimiento_a_falla", "#2B9B91", "M→F"),
    ]
    x_base = np.arange(3)
    width = 0.34
    for offset_index, (direction, color, label) in enumerate(directions):
        subset = summary[summary["direccion"].eq(direction)].sort_values("tasa")
        x = x_base + (offset_index - 0.5) * width
        axes[0].errorbar(
            x,
            subset["intervalos_mediana"],
            yerr=[
                subset["intervalos_mediana"] - subset["intervalos_p05"],
                subset["intervalos_p95"] - subset["intervalos_mediana"],
            ],
            fmt="o",
            capsize=3,
            color=color,
            label=label,
        )
        axes[1].errorbar(
            x,
            subset["r2_mediana"],
            yerr=[
                subset["r2_mediana"] - subset["r2_p05"],
                subset["r2_p95"] - subset["r2_mediana"],
            ],
            fmt="o",
            capsize=3,
            color=color,
            label=label,
        )
    for axis in axes:
        axis.set_xticks(x_base, ["5%", "10%", "20%"])
        axis.grid(axis="y", alpha=0.25)
    axes[0].axhline(base_n, color="#617282", linestyle="--", linewidth=1)
    axes[1].axhline(base_r2, color="#617282", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Intervalos reconstruidos")
    axes[0].set_title("Tamaño de muestra")
    axes[1].set_ylabel(r"$R^2$ agrupado")
    axes[1].set_title("Desempeño condicionado")
    axes[1].legend(frameon=False)
    figure.suptitle(
        "Sensibilidad a perturbaciones hipotéticas de etiquetas",
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
