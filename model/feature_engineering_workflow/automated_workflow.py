from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    from template_rules import (
        PROTECTED_ANCHOR_FEATURES,
        REQUIRED_MANUSCRIPT_TEMPLATES,
        build_level1_template_features,
        build_level2_template_features,
        infer_feature_family,
        infer_feature_meta,
        make_lineage_row,
    )
except ModuleNotFoundError:
    from .template_rules import (
        PROTECTED_ANCHOR_FEATURES,
        REQUIRED_MANUSCRIPT_TEMPLATES,
        build_level1_template_features,
        build_level2_template_features,
        infer_feature_family,
        infer_feature_meta,
        make_lineage_row,
    )


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = CURRENT_DIR / "feature_set_add_connection.csv"
DEFAULT_OUTPUT_DIR = CURRENT_DIR / "workflow_outputs"
DEFAULT_MANUSCRIPT_SET = (
    CURRENT_DIR.parent / "feature_selection_combination" / "final_compact_feature_set.csv"
)


@dataclass
class WorkflowConfig:
    source_path: str = str(DEFAULT_SOURCE)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    manuscript_feature_set: str = str(DEFAULT_MANUSCRIPT_SET)
    selection_schedule: List[int] = field(default_factory=lambda: [400, 300, 200, 100, 50])
    first_order_source: int = 20
    first_order_max_candidates: int = 50
    first_order_keep: int = 50
    second_order_keep: int = 50
    second_order_source: int = 10
    second_order_max_candidates: int = 20
    compact_scan_floor: int = 38
    compact_scan_step: int = 2
    final_floor: int = 24
    split_repeats: int =  3
    test_size: float = 0.1
    random_seed: int = 321
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    loss: str = "squared_error"
    n_jobs: int = max(1, min(5, os.cpu_count() or 1))


def parse_args() -> WorkflowConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manuscript-feature-set", default=str(DEFAULT_MANUSCRIPT_SET))
    parser.add_argument("--selection-schedule", default="400,300,200,100,50")
    parser.add_argument("--first-order-source", type=int, default=20)
    parser.add_argument("--first-order-max-candidates", type=int, default=50)
    parser.add_argument("--first-order-keep", type=int, default=50)
    parser.add_argument("--second-order-keep", type=int, default=50)
    parser.add_argument("--second-order-source", type=int, default=10)
    parser.add_argument("--second-order-max-candidates", type=int, default=20)
    parser.add_argument("--compact-scan-floor", type=int, default=38)
    parser.add_argument("--compact-scan-step", type=int, default=2)
    parser.add_argument("--final-floor", type=int, default=24)
    parser.add_argument("--split-repeats", type=int, default=5) 
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=0) 
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=max(1, min(5, os.cpu_count() or 1)))
    args = parser.parse_args()

    schedule = [int(item.strip()) for item in args.selection_schedule.split(",") if item.strip()]

    return WorkflowConfig(
        source_path=args.source_path,
        output_dir=args.output_dir,
        manuscript_feature_set=args.manuscript_feature_set,
        selection_schedule=schedule,
        first_order_source=args.first_order_source,
        first_order_max_candidates=args.first_order_max_candidates,
        first_order_keep=args.first_order_keep,
        second_order_keep=args.second_order_keep,
        second_order_source=args.second_order_source,
        second_order_max_candidates=args.second_order_max_candidates,
        compact_scan_floor=args.compact_scan_floor,
        compact_scan_step=args.compact_scan_step,
        final_floor=args.final_floor,
        split_repeats=args.split_repeats,
        test_size=args.test_size,
        random_seed=args.random_seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_jobs=args.n_jobs,
    )


def build_model(config: WorkflowConfig) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        loss=config.loss,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_seed,
        verbose=0,
    )


def fill_zero_rad(data_frame: pd.DataFrame) -> pd.DataFrame:
    if "B_ionic_radius" not in data_frame.columns or "X_ionic_radius" not in data_frame.columns:
        return data_frame

    b_radius, x_radius = [], []
    for index in range(len(data_frame)):
        if data_frame["B_ionic_radius"].iloc[index] == 0:
            b_radius.append(0.1)
        else:
            b_radius.append(data_frame["B_ionic_radius"].iloc[index])

        if data_frame["X_ionic_radius"].iloc[index] == 0:
            x_radius.append(0.1)
        else:
            x_radius.append(data_frame["X_ionic_radius"].iloc[index])

    data_frame = data_frame.drop(columns=["B_ionic_radius", "X_ionic_radius"])
    data_frame["B_ionic_radius"] = b_radius
    data_frame["X_ionic_radius"] = x_radius
    return data_frame


def data_clean(data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame = data_frame.select_dtypes(exclude="object")
    data_frame = data_frame.dropna(axis="columns")
    data_frame = data_frame.loc[:, ~data_frame.isin([np.inf, -np.inf]).any()]
    data_frame = fill_zero_rad(data_frame)
    return data_frame


def sanitize_feature_frame(data_frame: pd.DataFrame, drop_duplicate_values: bool = False) -> pd.DataFrame:
    data_frame = data_frame.loc[:, ~data_frame.columns.duplicated()].copy()
    data_frame = data_frame.replace([np.inf, -np.inf], np.nan)
    data_frame = data_frame.dropna(axis="columns")
    nunique = data_frame.nunique(dropna=False)
    data_frame = data_frame.loc[:, nunique > 1]
    if drop_duplicate_values and not data_frame.empty:
        data_frame = data_frame.loc[:, ~data_frame.T.duplicated()]
    return data_frame


def make_splits(
    n_samples: int,
    repeats: int,
    test_size: float,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    all_indices = np.arange(n_samples)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for offset in range(repeats):
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size=test_size,
            random_state=seed + offset,
            shuffle=True,
        )
        splits.append((train_idx, test_idx))
    return splits


def evaluate_feature_frame(
    data_frame: pd.DataFrame,
    target: Sequence[float],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: WorkflowConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    features = data_frame.columns.tolist()
    x_values = data_frame.to_numpy(dtype=float)
    y_values = np.asarray(target, dtype=float)

    metric_rows: List[Dict[str, float]] = []
    importances: List[np.ndarray] = []

    def evaluate_split(split_id: int, train_idx: np.ndarray, test_idx: np.ndarray):
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train = scaler.fit_transform(x_values[train_idx])
        x_test = scaler.transform(x_values[test_idx])
        y_train = y_values[train_idx]
        y_test = y_values[test_idx]

        model = build_model(config)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        return (
            {
                "split_id": split_id,
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mse,
                "rmse": np.sqrt(mse),
                "r2": r2_score(y_test, y_pred),
            },
            model.feature_importances_,
        )

    results = Parallel(n_jobs=config.n_jobs)(
        delayed(evaluate_split)(split_id, train_idx, test_idx)
        for split_id, (train_idx, test_idx) in enumerate(splits, start=1)
    )
    metric_rows = [metric_row for metric_row, _ in results]
    importances = [importance for _, importance in results]

    metrics = pd.DataFrame(metric_rows)
    importance = pd.Series(np.mean(importances, axis=0), index=features, name="importance")
    return metrics, importance


def summarize_metrics(metrics: pd.DataFrame) -> Dict[str, float]:
    return {
        "mean_mae": metrics["mae"].mean(),
        "std_mae": metrics["mae"].std(ddof=0),
        "mean_mse": metrics["mse"].mean(),
        "std_mse": metrics["mse"].std(ddof=0),
        "mean_rmse": metrics["rmse"].mean(),
        "std_rmse": metrics["rmse"].std(ddof=0),
        "mean_r2": metrics["r2"].mean(),
        "std_r2": metrics["r2"].std(ddof=0),
    }


def build_ranking_frame(importance: pd.Series) -> pd.DataFrame:
    ranking = (
        importance.sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["family"] = ranking["feature"].map(infer_feature_family)
    return ranking[["rank", "feature", "family", "importance"]]


def save_feature_set(data_frame: pd.DataFrame, target: Sequence[float], path: Path) -> None:
    output = data_frame.copy()
    output.insert(0, "formation_energy", target)
    output.to_csv(path, index=False)


def ordered_unique(features: Iterable[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for feature in features:
        if feature in seen:
            continue
        seen.add(feature)
        unique.append(feature)
    return unique


def build_raw_lineage(data_frame: pd.DataFrame) -> List[Dict[str, object]]:
    protected = set(PROTECTED_ANCHOR_FEATURES)
    return [
        make_lineage_row(
            feature=feature,
            stage="raw",
            complexity=0,
            formula=feature,
            parents=[],
            template_rule="source_feature",
            required_manuscript_template=False,
            protected_anchor=feature in protected,
        )
        for feature in data_frame.columns
    ]


def build_lineage_lookup(lineage_rows: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    for row in lineage_rows:
        lookup[str(row["feature"])] = row
    return lookup


def build_complexity_map(lineage_rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
    complexity: Dict[str, int] = {}
    for row in lineage_rows:
        complexity[str(row["feature"])] = int(row["complexity"])
    return complexity


def build_stage_map(lineage_rows: Iterable[Dict[str, object]]) -> Dict[str, str]:
    stages: Dict[str, str] = {}
    for row in lineage_rows:
        stages[str(row["feature"])] = str(row["stage"])
    return stages


def infer_site_scope(feature: str, feature_stage: str = "raw") -> str:
    if feature_stage.endswith("template"):
        return "motif"
    if feature_stage.endswith("optional"):
        return "optional"

    b_anchor_markers = {
        "MagpieData mean Electronegativity _B",
        "MagpieData mean CovalentRadius _B",
        "B_ionic_radius",
        "B_shannon_rad",
        "MagpieData mean GSvolume_pa _B",
    }
    x_anchor_markers = {
        "MagpieData mean Electronegativity _X",
        "MagpieData mean CovalentRadius _X",
        "X_ionic_radius",
        "X_shannon_rad",
        "MagpieData mean GSvolume_pa _X",
    }
    if feature in b_anchor_markers or " _B" in feature or feature.endswith("_B"):
        return "B"
    if feature in x_anchor_markers or " _X" in feature or feature.endswith("_X"):
        return "X"
    if "local_difference" in feature or "allsites" in feature:
        return "motif"
    if "formula" in feature:
        return "formula"
    return "global"


def family_consistent_operations(
    feature_a: str,
    feature_b: str,
    requested_operations: Iterable[str],
) -> Tuple[str, ...]:
    meta_a = infer_feature_meta(feature_a)
    meta_b = infer_feature_meta(feature_b)
    requested = tuple(requested_operations)
    operations: List[str] = []

    # Addition/subtraction are only meaningful for descriptors from the same
    # physical family and the same coarse unit group.
    same_family_and_units = (
        meta_a.family == meta_b.family and meta_a.unit_group == meta_b.unit_group
    )
    if same_family_and_units:
        if "add" in requested:
            operations.append("add")
        if "sub" in requested:
            operations.append("sub")

    # Ratios are allowed for same-family normalization, or when the denominator
    # is a physically meaningful positive scale term.
    positive_scale_families = {
        "covalent_radius",
        "ionic_radius",
        "shannon_radius",
        "covalent_radius_contrast",
        "ionic_radius_contrast",
        "shannon_radius_contrast",
        "covalent_radius_ratio",
        "ionic_radius_ratio",
        "shannon_radius_ratio",
        "volume",
        "density",
        "packing_fraction",
        "neighbor_distance",
    }
    safe_denominator = meta_b.positive_only and (
        meta_b.family in positive_scale_families
        or (
            meta_b.family.startswith("local_difference:")
            and meta_b.family.split(":", 1)[1] in positive_scale_families
        )
    )
    if "div" in requested and (same_family_and_units or safe_denominator):
        operations.append("div")

    return tuple(operations)


def write_lineage(lineage_rows: Iterable[Dict[str, object]], output_dir: Path) -> None:
    lineage = pd.DataFrame(lineage_rows)
    if lineage.empty:
        return
    lineage = lineage.drop_duplicates(subset=["feature"], keep="last")
    lineage = lineage.sort_values(["complexity", "stage", "feature"])
    lineage.to_csv(output_dir / "feature_lineage.csv", index=False)


def select_top_features(
    data_frame: pd.DataFrame,
    target: Sequence[float],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: WorkflowConfig,
    keep: int,
    stage_name: str,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    candidate_metrics, importance = evaluate_feature_frame(data_frame, target, splits, config)
    ranking = build_ranking_frame(importance)
    keep = min(keep, len(ranking))
    selected_columns = ranking["feature"].iloc[:keep].tolist()
    selected = data_frame[selected_columns].copy()

    selected_metrics, _ = evaluate_feature_frame(selected, target, splits, config)

    candidate_metrics.to_csv(output_dir / f"{stage_name}_candidate_split_metrics.csv", index=False)
    pd.DataFrame([summarize_metrics(candidate_metrics)]).to_csv(
        output_dir / f"{stage_name}_candidate_summary.csv", index=False
    )
    selected_metrics.to_csv(output_dir / f"{stage_name}_split_metrics.csv", index=False)
    ranking.to_csv(output_dir / f"{stage_name}_ranking.csv", index=False)

    summary = summarize_metrics(selected_metrics)
    summary["n_features"] = keep
    pd.DataFrame([summary]).to_csv(output_dir / f"{stage_name}_summary.csv", index=False)

    return selected, ranking, summary


def build_optional_pairwise_candidates(
    data_frame: pd.DataFrame,
    ordered_features: Iterable[str],
    max_candidates: int,
    feature_complexities: Dict[str, int],
    feature_stages: Dict[str, str],
    stage: str,
    max_complexity: int = 2,
    allowed_stage_pairs: Iterable[Tuple[str, str]] | None = None,
    allowed_site_pairs: Iterable[Tuple[str, str]] | None = None,
    operations: Iterable[str] = ("add", "sub", "div"),
) -> Tuple[Dict[str, pd.Series], List[Dict[str, object]]]:
    generated: Dict[str, pd.Series] = {}
    lineage_rows: List[Dict[str, object]] = []
    ordered_features = [feature for feature in ordered_unique(ordered_features) if feature in data_frame.columns]
    allowed_pairs = {tuple(sorted(pair)) for pair in allowed_stage_pairs or []}
    allowed_sites = {tuple(sorted(pair)) for pair in allowed_site_pairs or []}
    operations = tuple(operations)

    for index, feature_a in enumerate(ordered_features):
        for feature_b in ordered_features[index + 1 :]:
            complexity_a = feature_complexities.get(feature_a, 0)
            complexity_b = feature_complexities.get(feature_b, 0)
            result_complexity = max(complexity_a, complexity_b) + 1
            if result_complexity > max_complexity:
                continue

            if allowed_pairs:
                pair = tuple(sorted((feature_stages.get(feature_a, "raw"), feature_stages.get(feature_b, "raw"))))
                if pair not in allowed_pairs:
                    continue

            if allowed_sites:
                site_pair = tuple(
                    sorted(
                        (
                            infer_site_scope(feature_a, feature_stages.get(feature_a, "raw")),
                            infer_site_scope(feature_b, feature_stages.get(feature_b, "raw")),
                        )
                    )
                )
                if site_pair not in allowed_sites:
                    continue

            allowed_operations = family_consistent_operations(feature_a, feature_b, operations)
            if not allowed_operations:
                continue

            meta_a = infer_feature_meta(feature_a)
            meta_b = infer_feature_meta(feature_b)

            for operation in allowed_operations:
                if operation == "add":
                    name = f"({feature_a})+({feature_b})"
                    series = data_frame[feature_a] + data_frame[feature_b]
                    formula = f"({feature_a}) + ({feature_b})"
                elif operation == "sub":
                    name = f"({feature_a})-({feature_b})"
                    series = data_frame[feature_a] - data_frame[feature_b]
                    formula = f"({feature_a}) - ({feature_b})"
                elif operation == "div":
                    name = f"({feature_a})/({feature_b})"
                    series = data_frame[feature_a] / data_frame[feature_b]
                    formula = f"({feature_a}) / ({feature_b})"
                else:
                    continue

                if name in generated or name in data_frame.columns:
                    continue

                generated[name] = series
                lineage_row = make_lineage_row(
                    feature=name,
                    stage=stage,
                    complexity=result_complexity,
                    formula=formula,
                    parents=[feature_a, feature_b],
                    template_rule=f"optional_pairwise_{operation}",
                    required_manuscript_template=False,
                )
                lineage_row["parent_families"] = f"{meta_a.family} | {meta_b.family}"
                lineage_row["parent_unit_groups"] = f"{meta_a.unit_group} | {meta_b.unit_group}"
                lineage_row["parent_scopes"] = f"{meta_a.scope} | {meta_b.scope}"
                lineage_rows.append(lineage_row)
                if len(generated) >= max_candidates:
                    return generated, lineage_rows

    return generated, lineage_rows


def merge_feature_pools(base_frame: pd.DataFrame, additions: Dict[str, pd.Series]) -> pd.DataFrame:
    if not additions:
        return base_frame.copy()
    addition_frame = pd.DataFrame(additions, index=base_frame.index)
    return sanitize_feature_frame(pd.concat([base_frame, addition_frame], axis=1))


def choose_smallest_within_one_std(scan_frame: pd.DataFrame) -> pd.Series:
    best_row = scan_frame.loc[scan_frame["mean_mse"].idxmin()]
    threshold = best_row["mean_mse"] + best_row["std_mse"]
    eligible = scan_frame.loc[scan_frame["mean_mse"] <= threshold].copy()
    return eligible.sort_values("n_features", ascending=True).iloc[0]


def scan_compact_pool(
    data_frame: pd.DataFrame,
    target: Sequence[float],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: WorkflowConfig,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _, importance = evaluate_feature_frame(data_frame, target, splits, config)
    ranking = build_ranking_frame(importance)
    max_size = len(ranking)
    min_size = min(config.compact_scan_floor, max_size)
    candidate_sizes = list(range(max_size, min_size - 1, -config.compact_scan_step))
    if candidate_sizes[-1] != min_size:
        candidate_sizes.append(min_size)

    scan_rows: List[Dict[str, float]] = []
    ordered_columns = ranking["feature"].tolist()

    for size in candidate_sizes:
        subset = data_frame[ordered_columns[:size]].copy()
        metrics, _ = evaluate_feature_frame(subset, target, splits, config)
        summary = summarize_metrics(metrics)
        summary["n_features"] = size
        scan_rows.append(summary)

    scan_frame = pd.DataFrame(scan_rows).sort_values("n_features", ascending=False)
    scan_frame.to_csv(output_dir / "compact_pool_scan.csv", index=False)

    chosen_row = choose_smallest_within_one_std(scan_frame)
    chosen_columns = ordered_columns[: int(chosen_row["n_features"])]
    pre_rfe_pool = data_frame[chosen_columns].copy()
    return pre_rfe_pool, scan_frame


def run_rfe(
    data_frame: pd.DataFrame,
    target: Sequence[float],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: WorkflowConfig,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current = data_frame.copy()
    history_rows: List[Dict[str, object]] = []

    while len(current.columns) >= config.final_floor:
        metrics, importance = evaluate_feature_frame(current, target, splits, config)
        ranking = build_ranking_frame(importance)
        summary = summarize_metrics(metrics)
        summary["n_features"] = len(current.columns)
        summary["features"] = " | ".join(current.columns.tolist())
        history_rows.append(summary)

        if len(current.columns) == config.final_floor:
            break

        feature_to_remove = ranking.iloc[-1]["feature"]
        current = current.drop(columns=[feature_to_remove])

    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(output_dir / "final_rfe_scores.csv", index=False)

    chosen_row = choose_smallest_within_one_std(history_frame)
    final_columns = chosen_row["features"].split(" | ")
    final_frame = data_frame[final_columns].copy()

    _, final_importance = evaluate_feature_frame(final_frame, target, splits, config)
    final_ranking = build_ranking_frame(final_importance)
    final_ranking.to_csv(output_dir / "automated_final_ranking.csv", index=False)

    return final_frame, history_frame, final_ranking


def run_rfe_to_fixed_size(
    data_frame: pd.DataFrame,
    target: Sequence[float],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: WorkflowConfig,
    output_dir: Path,
    stop_size: int = 38,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current = data_frame.copy()
    history_rows: List[Dict[str, object]] = []

    while len(current.columns) >= stop_size:
        metrics, importance = evaluate_feature_frame(current, target, splits, config)
        ranking = build_ranking_frame(importance)
        summary = summarize_metrics(metrics)
        summary["n_features"] = len(current.columns)
        summary["features"] = " | ".join(current.columns.tolist())
        summary["removed_feature"] = "" if len(current.columns) == stop_size else ranking.iloc[-1]["feature"]
        history_rows.append(summary)

        if len(current.columns) == stop_size:
            break

        current = current.drop(columns=[ranking.iloc[-1]["feature"]])

    history_frame = pd.DataFrame(history_rows)
    history_frame.to_csv(output_dir / "automated_rfe_to_38_scores.csv", index=False)

    _, final_importance = evaluate_feature_frame(current, target, splits, config)
    final_ranking = build_ranking_frame(final_importance)
    final_ranking.to_csv(output_dir / "automated_rfe_38_ranking.csv", index=False)
    return current.copy(), history_frame, final_ranking


def write_reference_feature_set_metrics(
    feature_sets: Iterable[Tuple[str, pd.DataFrame, Sequence[float]]],
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    config: WorkflowConfig,
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    summary_rows: List[Dict[str, object]] = []
    split_rows: List[pd.DataFrame] = []
    summary_lookup: Dict[str, Dict[str, float]] = {}

    for name, frame, target in feature_sets:
        metrics, _ = evaluate_feature_frame(frame, target, splits, config)
        metrics.insert(0, "feature_set", name)
        metrics.insert(1, "n_features", frame.shape[1])
        split_rows.append(metrics)

        summary = summarize_metrics(metrics)
        summary["feature_set"] = name
        summary["n_features"] = frame.shape[1]
        summary_rows.append(summary)
        summary_lookup[name] = {
            key: float(value)
            for key, value in summary.items()
            if key not in {"feature_set", "n_features"}
        }

    pd.DataFrame(summary_rows).to_csv(output_dir / "reference_feature_set_metrics.csv", index=False)
    pd.concat(split_rows, ignore_index=True).to_csv(
        output_dir / "reference_feature_set_split_metrics.csv", index=False
    )
    return summary_lookup


def write_alignment_tables(
    final_frame: pd.DataFrame,
    pre_rfe_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    stage1_frame: pd.DataFrame,
    stage2_frame: pd.DataFrame,
    stage3_frame: pd.DataFrame,
    template_candidate_frame: pd.DataFrame,
    config: WorkflowConfig,
    output_dir: Path,
    lineage_lookup: Dict[str, Dict[str, object]],
) -> None:
    manuscript_path = Path(config.manuscript_feature_set)
    if not manuscript_path.exists():
        return

    manuscript_features = pd.read_csv(manuscript_path, nrows=1).columns.tolist()[1:]
    alignment_rows = []
    for feature in manuscript_features:
        lineage = lineage_lookup.get(feature, {})
        alignment_rows.append(
            {
                "feature": feature,
                "family": infer_feature_family(feature),
                "complexity": lineage.get("complexity", 0 if feature in clean_frame.columns else ""),
                "lineage_stage": lineage.get("stage", "raw" if feature in clean_frame.columns else ""),
                "required_manuscript_template": bool(
                    lineage.get("required_manuscript_template", feature in REQUIRED_MANUSCRIPT_TEMPLATES)
                ),
                "in_clean_raw_pool": feature in clean_frame.columns,
                "in_template_candidate_pool": feature in template_candidate_frame.columns,
                "in_stage1_base_pool": feature in stage1_frame.columns,
                "in_stage2_first_order_pool": feature in stage2_frame.columns,
                "in_stage3_second_order_pool": feature in stage3_frame.columns,
                "in_pre_rfe_pool": feature in pre_rfe_frame.columns,
                "in_automated_final": feature in final_frame.columns,
            }
        )

    alignment = pd.DataFrame(alignment_rows)
    alignment.to_csv(output_dir / "manuscript_alignment.csv", index=False)

    summary = {
        "manuscript_feature_count": len(manuscript_features),
        "generated_or_raw_manuscript_features": int(
            (alignment["in_clean_raw_pool"] | alignment["in_template_candidate_pool"]).sum()
        ),
        "exact_overlap_with_pre_rfe_pool": int(alignment["in_pre_rfe_pool"].sum()),
        "exact_overlap_with_automated_final": int(alignment["in_automated_final"].sum()),
    }
    with open(output_dir / "alignment_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def write_manuscript_generation_check(
    clean_frame: pd.DataFrame,
    stage1_frame: pd.DataFrame,
    stage2_frame: pd.DataFrame,
    stage3_frame: pd.DataFrame,
    pre_rfe_frame: pd.DataFrame,
    final_frame: pd.DataFrame,
    template_candidate_frame: pd.DataFrame,
    output_dir: Path,
    lineage_lookup: Dict[str, Dict[str, object]],
) -> None:
    rows = []
    for feature in sorted(REQUIRED_MANUSCRIPT_TEMPLATES):
        lineage = lineage_lookup.get(feature, {})
        rows.append(
            {
                "feature": feature,
                "family": infer_feature_family(feature),
                "complexity": lineage.get("complexity", ""),
                "lineage_stage": lineage.get("stage", ""),
                "generated_successfully": feature in template_candidate_frame.columns
                or feature in clean_frame.columns,
                "in_clean_raw_pool": feature in clean_frame.columns,
                "in_template_candidate_pool": feature in template_candidate_frame.columns,
                "in_stage1_base_pool": feature in stage1_frame.columns,
                "in_stage2_first_order_pool": feature in stage2_frame.columns,
                "in_stage3_second_order_pool": feature in stage3_frame.columns,
                "in_pre_rfe_pool": feature in pre_rfe_frame.columns,
                "in_automated_final": feature in final_frame.columns,
                "formula": lineage.get("formula", ""),
                "parents": lineage.get("parents", ""),
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "manuscript_generation_check.csv", index=False)


def run_workflow(config: WorkflowConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "workflow_config.json", "w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    raw_data = pd.read_csv(config.source_path)
    raw_data = raw_data.drop(columns=["file_name", "formula"], errors="ignore")
    target = raw_data.pop("formation_energy")
    clean_data = data_clean(raw_data)
    clean_data = sanitize_feature_frame(clean_data)

    feature_summary = {
        "raw_numeric_features_after_cleaning": int(clean_data.shape[1]),
        "n_samples": int(clean_data.shape[0]),
    }
    with open(output_dir / "initial_feature_summary.json", "w", encoding="utf-8") as file:
        json.dump(feature_summary, file, indent=2)

    save_feature_set(clean_data, target, output_dir / "clean_feature_pool.csv")
    splits = make_splits(len(clean_data), config.split_repeats, config.test_size, config.random_seed)

    lineage_rows = build_raw_lineage(clean_data)
    stage_rows: List[Dict[str, float]] = []

    # Stage 1: raw feature reduction from the expanded descriptor pool.
    current = clean_data.copy()
    stage1_ranking = pd.DataFrame()
    raw_global_ranking = pd.DataFrame()
    stage_prefix = "stage1_base"
    for step_id, keep in enumerate(config.selection_schedule, start=1):
        current, stage1_ranking, summary = select_top_features(
            current,
            target,
            splits,
            config,
            keep=keep,
            stage_name=f"{stage_prefix}_{step_id}",
            output_dir=output_dir,
        )
        if step_id == 1:
            raw_global_ranking = stage1_ranking.copy()
        summary["stage"] = f"{stage_prefix}_{step_id}"
        stage_rows.append(summary)

    base_pool = current.copy()
    save_feature_set(base_pool, target, output_dir / "stage1_base_pool.csv")

    # Stage 2: level-1 motif-prior templates plus capped optional pairwise terms.
    level1_template_generated, _, level1_template_lineage, missing_level1 = (
        build_level1_template_features(clean_data)
    )
    lineage_rows.extend(level1_template_lineage)

    protected_sources = [feature for feature in PROTECTED_ANCHOR_FEATURES if feature in clean_data.columns]
    raw_source_ranking = raw_global_ranking if not raw_global_ranking.empty else stage1_ranking
    stage1_sources = [
        feature
        for feature in raw_source_ranking["feature"].tolist()
        if feature in clean_data.columns
        and infer_site_scope(feature, "raw") in {"B", "X"}
    ][: config.first_order_source]
    first_order_inputs = ordered_unique(stage1_sources + protected_sources)

    level1_optional_generated, level1_optional_lineage = build_optional_pairwise_candidates(
        clean_data,
        first_order_inputs,
        max_candidates=config.first_order_max_candidates,
        feature_complexities=build_complexity_map(lineage_rows),
        feature_stages=build_stage_map(lineage_rows),
        stage="level1_optional",
        max_complexity=1,
        allowed_stage_pairs=[("raw", "raw")],
        allowed_site_pairs=[
            ("B", "B"),
            ("B", "X"),
            ("X", "X"),
        ],
        operations=("add", "sub", "div"),
    )
    lineage_rows.extend(level1_optional_lineage)

    level1_generated: Dict[str, pd.Series] = {}
    level1_generated.update(level1_template_generated)
    for feature, series in level1_optional_generated.items():
        if feature not in level1_generated:
            level1_generated[feature] = series

    if missing_level1:
        pd.DataFrame({"missing_template": missing_level1}).to_csv(
            output_dir / "missing_level1_templates.csv", index=False
        )

    first_order_pool = merge_feature_pools(base_pool, level1_generated)
    first_order_selected, first_order_ranking, summary = select_top_features(
        first_order_pool,
        target,
        splits,
        config,
        keep=config.first_order_keep,
        stage_name="stage2_first_order",
        output_dir=output_dir,
    )
    summary["stage"] = "stage2_first_order"
    summary["level1_template_candidates"] = len(level1_template_generated)
    summary["level1_optional_candidates"] = len(level1_optional_generated)
    stage_rows.append(summary)
    save_feature_set(first_order_selected, target, output_dir / "stage2_first_order_pool.csv")

    # Stage 3: required level-2 templates are generated independently of Stage 2
    # selection; the cap only limits optional pairwise terms.
    level1_candidate_frame = pd.DataFrame(level1_generated, index=clean_data.index)
    level2_source_frame = sanitize_feature_frame(
        pd.concat([clean_data, level1_candidate_frame], axis=1)
    )
    level2_template_generated, _, level2_template_lineage, missing_level2 = (
        build_level2_template_features(level2_source_frame)
    )
    lineage_rows.extend(level2_template_lineage)

    if missing_level2:
        pd.DataFrame({"missing_template": missing_level2}).to_csv(
            output_dir / "missing_level2_templates.csv", index=False
        )

    complexity_map = build_complexity_map(lineage_rows)
    stage_map = build_stage_map(lineage_rows)
    second_order_inputs = [
        feature
        for feature in first_order_ranking["feature"].tolist()
        if feature in first_order_selected.columns
        and (
            (
                stage_map.get(feature, "raw") == "raw"
                and infer_site_scope(feature, "raw") in {"B", "X"}
            )
            or stage_map.get(feature, "raw") == "level1_template"
        )
        and complexity_map.get(feature, 0) <= 1
    ][: config.second_order_source]
    second_order_optional_generated, second_order_optional_lineage = build_optional_pairwise_candidates(
        first_order_selected,
        second_order_inputs,
        max_candidates=config.second_order_max_candidates,
        feature_complexities=complexity_map,
        feature_stages=stage_map,
        stage="level2_optional",
        max_complexity=2,
        allowed_stage_pairs=[
            ("raw", "level1_template"),
            ("level1_template", "level1_template"),
        ],
        allowed_site_pairs=[
            ("B", "motif"),
            ("X", "motif"),
            ("motif", "motif"),
        ],
        operations=("add", "sub", "div"),
    )
    lineage_rows.extend(second_order_optional_lineage)

    level2_generated: Dict[str, pd.Series] = {}
    level2_generated.update(level2_template_generated)
    for feature, series in second_order_optional_generated.items():
        if feature not in level2_generated:
            level2_generated[feature] = series

    all_template_generated: Dict[str, pd.Series] = {}
    all_template_generated.update(level1_generated)
    for feature, series in level2_generated.items():
        if feature not in all_template_generated:
            all_template_generated[feature] = series

    template_candidate_pool = sanitize_feature_frame(
        pd.DataFrame(all_template_generated, index=clean_data.index)
    )
    save_feature_set(template_candidate_pool, target, output_dir / "template_candidate_pool.csv")

    too_complex = [
        row for row in lineage_rows if row["stage"] != "raw" and int(row["complexity"]) > 2
    ]
    if too_complex:
        raise ValueError("Generated features with complexity > 2 were found.")

    second_order_pool = merge_feature_pools(first_order_selected, level2_generated)
    second_order_selected, _, summary = select_top_features(
        second_order_pool,
        target,
        splits,
        config,
        keep=config.second_order_keep,
        stage_name="stage3_second_order",
        output_dir=output_dir,
    )
    summary["stage"] = "stage3_second_order"
    summary["level2_template_candidates"] = len(level2_template_generated)
    summary["level2_optional_candidates"] = len(second_order_optional_generated)
    stage_rows.append(summary)
    save_feature_set(second_order_selected, target, output_dir / "stage3_second_order_pool.csv")

    automated_rfe_38_frame, rfe_history, _ = run_rfe_to_fixed_size(
        second_order_selected,
        target,
        splits,
        config,
        output_dir,
        stop_size=38,
    )
    save_feature_set(automated_rfe_38_frame, target, output_dir / "automated_rfe_38_feature_set.csv")
    save_feature_set(automated_rfe_38_frame, target, output_dir / "pre_rfe_feature_pool.csv")
    save_feature_set(automated_rfe_38_frame, target, output_dir / "automated_final_feature_set.csv")

    lineage_lookup = build_lineage_lookup(lineage_rows)
    write_lineage(lineage_rows, output_dir)
    write_alignment_tables(
        automated_rfe_38_frame,
        automated_rfe_38_frame,
        clean_data,
        base_pool,
        first_order_selected,
        second_order_selected,
        template_candidate_pool,
        config,
        output_dir,
        lineage_lookup,
    )
    write_manuscript_generation_check(
        clean_data,
        base_pool,
        first_order_selected,
        second_order_selected,
        automated_rfe_38_frame,
        automated_rfe_38_frame,
        template_candidate_pool,
        output_dir,
        lineage_lookup,
    )

    reference_feature_sets: List[Tuple[str, pd.DataFrame, Sequence[float]]] = [
        ("stage3_50", second_order_selected, target),
        ("automated_rfe_38", automated_rfe_38_frame, target),
    ]
    manuscript_path = Path(config.manuscript_feature_set)
    if manuscript_path.exists():
        manuscript_frame = pd.read_csv(manuscript_path)
        manuscript_target = manuscript_frame.pop("formation_energy")
        if len(manuscript_frame) == len(clean_data):
            reference_feature_sets.append(
                ("manuscript_final_34", manuscript_frame, manuscript_target)
            )
    reference_metrics = write_reference_feature_set_metrics(
        reference_feature_sets,
        splits,
        config,
        output_dir,
    )

    pd.DataFrame(stage_rows).to_csv(output_dir / "workflow_stage_summary.csv", index=False)

    generation_check_path = output_dir / "manuscript_generation_check.csv"
    if generation_check_path.exists():
        generation_check = pd.read_csv(generation_check_path)
        required_generated = int(generation_check["generated_successfully"].sum())
    else:
        required_generated = 0

    rfe_38_row = rfe_history.loc[rfe_history["n_features"] == 38].iloc[0]

    workflow_summary = {
        "initial_clean_feature_pool_size": int(clean_data.shape[1]),
        "stage1_base_pool_size": int(base_pool.shape[1]),
        "stage2_first_order_pool_size": int(first_order_selected.shape[1]),
        "stage3_second_order_pool_size": int(second_order_selected.shape[1]),
        "stage3_pool_size": int(second_order_selected.shape[1]),
        "template_candidate_pool_size": int(template_candidate_pool.shape[1]),
        "level1_template_candidates": int(len(level1_template_generated)),
        "level1_optional_candidates": int(len(level1_optional_generated)),
        "level2_template_candidates": int(len(level2_template_generated)),
        "level2_optional_candidates": int(len(second_order_optional_generated)),
        "required_manuscript_templates": int(len(REQUIRED_MANUSCRIPT_TEMPLATES)),
        "required_manuscript_templates_generated": required_generated,
        "automated_rfe_38_size": int(automated_rfe_38_frame.shape[1]),
        "pre_rfe_pool_size": int(automated_rfe_38_frame.shape[1]),
        "automated_rfe_38_mean_mse": float(rfe_38_row["mean_mse"]),
        "automated_rfe_38_mean_mae": float(rfe_38_row["mean_mae"]),
        "automated_rfe_to_38_best_mse": float(rfe_history["mean_mse"].min()),
        "reference_manuscript_final_34_mean_mse": float(
            reference_metrics.get("manuscript_final_34", {}).get("mean_mse", np.nan)
        ),
        "max_generated_complexity": int(
            max(row["complexity"] for row in lineage_rows if row["stage"] != "raw")
        ),
    }
    with open(output_dir / "workflow_summary.json", "w", encoding="utf-8") as file:
        json.dump(workflow_summary, file, indent=2)


if __name__ == "__main__":
    run_workflow(parse_args())
