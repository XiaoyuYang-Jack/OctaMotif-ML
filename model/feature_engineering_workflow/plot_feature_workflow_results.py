from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKFLOW_DIR = CURRENT_DIR / "workflow_outputs"
DEFAULT_PICTURE_DIR = CURRENT_DIR / "workflow_results_pictures"

LINEWIDTH = 2.5
FONTSIZE = 20
TICKSIZE = 17

COLORS = {
    "mse": "mediumslateblue",
    "raw": "#4C78A8",
    "level1": "#F58518",
    "level2": "#54A24B",
    "electronegativity": "#7B61A8",
    "motif_en_radius": "#D65F5F",
    "radius": "#4C78A8",
    "local_difference": "#54A24B",
    "valence": "#F58518",
    "unfilled": "#B279A2",
    "bandgap": "#72B7B2",
    "ewald_density": "#9D755D",
    "structure_other": "#BAB0AC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-dir", default=str(DEFAULT_WORKFLOW_DIR))
    parser.add_argument("--picture-dir", default=str(DEFAULT_PICTURE_DIR))
    parser.add_argument("--file-format", default="png", choices=["png", "pdf", "jpg", "svg"])
    parser.add_argument("--error-bar", default="std", choices=["std", "sem", "both"])
    return parser.parse_args()


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(width=LINEWIDTH, labelsize=TICKSIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(LINEWIDTH)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def split_repeats(workflow_dir: Path) -> int:
    config_path = workflow_dir / "workflow_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as file:
            return int(json.load(file).get("split_repeats", 1))

    split_metrics_path = workflow_dir / "stage1_base_1_split_metrics.csv"
    if split_metrics_path.exists():
        return len(pd.read_csv(split_metrics_path))
    return 1


def mse_error(data: pd.DataFrame, workflow_dir: Path, error_bar: str) -> pd.Series:
    if error_bar == "sem":
        return data["std_mse"] / np.sqrt(split_repeats(workflow_dir))
    return data["std_mse"]


def error_suffix(error_bar: str) -> str:
    return "_SEM" if error_bar == "sem" else ""


def read_feature_columns(path: Path) -> List[str]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    return [column for column in columns if column != "formation_energy"]


def make_lineage_lookup(workflow_dir: Path) -> Dict[str, Dict[str, object]]:
    lineage = pd.read_csv(workflow_dir / "feature_lineage.csv")
    lineage = lineage.drop_duplicates(subset=["feature"], keep="last")
    return lineage.set_index("feature").to_dict(orient="index")


def stage_label(stage: str, n_features: int) -> str:
    # if stage.startswith("stage1_base"):
    #     return f"FS\n{n_features}"
    if stage == "stage1_base_1":
        return f"FS-1\n{n_features}"
    if stage == "stage1_base_2":
        return f"FS-2\n{n_features}"
    if stage == "stage1_base_3":
        return f"FS-3\n{n_features}"
    if stage == "stage1_base_4":
        return f"FS-4\n{n_features}"
    if stage == "stage1_base_5":
        return f"FS-5\n{n_features}"
    if stage == "stage2_first_order":
        return "FC-FS-1\n50"
    if stage == "stage3_second_order":
        return "FC-FS-2\n50"
    return stage.replace("_", "\n")


# Figure 1, stage 1, stage 2, stage 3
def plot_stage_performance(
    workflow_dir: Path,
    picture_dir: Path,
    file_format: str,
    error_bar: str = "std",
    ylim: tuple[float, float] | None = None,
) -> None:
    data = pd.read_csv(workflow_dir / "workflow_stage_summary.csv")
    x = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        x,
        data["mean_mse"],
        yerr=mse_error(data, workflow_dir, error_bar),
        marker="o",
        markersize=8,
        color= 'blueviolet',# COLORS["mse"],
        markeredgecolor="w",
        linewidth=2,
        capsize=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [stage_label(row.stage, int(row.n_features)) for row in data.itertuples()],
        rotation=0,
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Feature engineering stage", fontsize=FONTSIZE)
    ax.set_ylabel("MSE", fontsize=FONTSIZE)
    ax.yaxis.set_major_locator(MultipleLocator(0.004))
    style_axis(ax)
    save_figure(fig, picture_dir / f"1_stage_wise_mse{error_suffix(error_bar)}.{file_format}")

# Figure 3, stage 4: RFE 50-to-38
def plot_rfe_plateau(
    workflow_dir: Path,
    picture_dir: Path,
    file_format: str,
    ylim: tuple[float, float] | None = None,
    error_bar: str = "std",
) -> None:
    data = pd.read_csv(workflow_dir / "automated_rfe_to_38_scores.csv")
    x = np.arange(len(data))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        x,
        data["mean_mse"],
        yerr=mse_error(data, workflow_dir, error_bar),
        marker="o",
        markersize=8,
        color=COLORS["mse"],
        markeredgecolor="w",
        linewidth=2,
        capsize=5,
    )
    # ax.scatter(
    #     x.iloc[-1] if hasattr(x, "iloc") else x[-1],
    #     data["mean_mse"].iloc[-1],
    #     s=140,
    #     color="#E45756",
    #     edgecolor="k",
    #     linewidth=1.5,
    #     zorder=5,
    #     label="38-feature pool",
    # )
    ax.set_xticks(x)
    ax.set_xticklabels(data["n_features"].astype(str).tolist())
    ax.set_xlabel("Number of retained features", fontsize=FONTSIZE)
    ax.set_ylabel("MSE", fontsize=FONTSIZE)
    if ylim is not None:
        ax.set_ylim(*ylim)
    # ax.set_xlim(data["mean_mse"].min() - 0.004, data["mean_mse"].max() + 0.004)
    ax.yaxis.set_major_locator(MultipleLocator(0.004))
    # ax.legend(prop={"size": 16}, framealpha=0, loc="upper left")
    style_axis(ax)
    save_figure(fig, picture_dir / f"3_rfe_50_to_38_mse{error_suffix(error_bar)}.{file_format}")


def complexity_label(feature: str, lineage_lookup: Dict[str, Dict[str, object]]) -> str:
    complexity = int(lineage_lookup.get(feature, {}).get("complexity", 0))
    if complexity == 1:
        return "Level-1"
    if complexity >= 2:
        return "Level-2"
    return "Raw"


def feature_family(feature: str, lineage_lookup: Dict[str, Dict[str, object]]) -> str:
    return str(lineage_lookup.get(feature, {}).get("family", "other"))


def family_group(family: str) -> str:
    if family.startswith("local_difference:"):
        return "Local difference"
    if family.startswith("electronegativity_") and "radius" in family:
        return "Motif En/r"
    if "electronegativity" in family:
        return "Electronegativity"
    if "radius" in family:
        return "Radius"
    if "valence" in family:
        return "Valence"
    if "unfilled" in family:
        return "Unfilled"
    if family == "bandgap":
        return "GSgap"
    if family in {"ewald", "density"}:
        return "Ewald/density"
    return "Structure/other"


def count_categories(
    feature_sets: Dict[str, Iterable[str]],
    category_func,
    categories: List[str],
) -> pd.DataFrame:
    rows = []
    for set_name, features in feature_sets.items():
        counts = {category: 0 for category in categories}
        for feature in features:
            category = category_func(feature)
            counts[category] = counts.get(category, 0) + 1
        counts["feature_set"] = set_name
        rows.append(counts)
    return pd.DataFrame(rows).set_index("feature_set")

# Figure 2
def plot_stacked_bar(
    counts: pd.DataFrame,
    colors: Dict[str, str],
    ylabel: str,
    path: Path,
    percent: bool = False,
) -> None:
    if percent:
        counts = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))

    for category in counts.columns:
        values = counts[category].to_numpy()
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.65,
            color=colors[category],
            label=category,
            edgecolor="white",
            linewidth=0.8,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.tolist(), rotation=20) # , ha="right"
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.legend(prop={"size": 13}, framealpha=0, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    if percent:
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MultipleLocator(20))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(5))
    style_axis(ax)
    save_figure(fig, path)


def load_feature_sets(workflow_dir: Path) -> Dict[str, List[str]]:
    manuscript_alignment = pd.read_csv(workflow_dir / "manuscript_alignment.csv")
    return {
        # "Template\ncandidate": read_feature_columns(workflow_dir / "template_candidate_pool.csv"),
        "Stage3\n50": read_feature_columns(workflow_dir / "stage3_second_order_pool.csv"),
        "Auto RFE\n38": read_feature_columns(workflow_dir / "automated_rfe_38_feature_set.csv"),
        "Final\n34": manuscript_alignment["feature"].tolist(),
    }



def load_workflow_stage_feature_sets(workflow_dir: Path) -> Dict[str, List[str]]:
    return {
        "FS-1\n400": pd.read_csv(workflow_dir / "stage1_base_1_ranking.csv")["feature"].head(400).tolist(),
        "FS-2\n300": pd.read_csv(workflow_dir / "stage1_base_2_ranking.csv")["feature"].head(300).tolist(),
        "FS-3\n200": pd.read_csv(workflow_dir / "stage1_base_3_ranking.csv")["feature"].head(200).tolist(),
        "FS-4\n100": pd.read_csv(workflow_dir / "stage1_base_4_ranking.csv")["feature"].head(100).tolist(),
        "FS-5\n50": read_feature_columns(workflow_dir / "stage1_base_pool.csv"),
        "FC-FS-1\n50": read_feature_columns(workflow_dir / "stage2_first_order_pool.csv"),
        "FC-FS-2\n50": read_feature_columns(workflow_dir / "stage3_second_order_pool.csv"),
        # "RFE\n38": read_feature_columns(workflow_dir / "automated_rfe_38_feature_set.csv"),
    }


# Figure 2
def plot_workflow_stage_family(
    workflow_dir: Path,
    picture_dir: Path,
    file_format: str,
) -> None:
    lineage_lookup = make_lineage_lookup(workflow_dir)
    feature_sets = load_workflow_stage_feature_sets(workflow_dir)
    categories = [
        "Electronegativity",
        "Motif En/r",
        "Radius",
        "Local difference",
        "Valence",
        "Unfilled",
        "Bandgap",
        "Ewald/density",
        "Structure/other",
    ]
    counts = count_categories(
        feature_sets,
        lambda feature: family_group(feature_family(feature, lineage_lookup)),
        categories,
    )
    plot_stacked_bar(
        counts[categories],
        {
            "Electronegativity": COLORS["electronegativity"],
            "Motif En/r": COLORS["motif_en_radius"],
            "Radius": COLORS["radius"],
            "Local difference": COLORS["local_difference"],
            "Valence": COLORS["valence"],
            "Unfilled": COLORS["unfilled"],
            "Bandgap": COLORS["bandgap"],
            "Ewald/density": COLORS["ewald_density"],
            "Structure/other": COLORS["structure_other"],
        },
        "Feature proportion (%)",
        picture_dir / f"2_workflow_stage_feature_family.{file_format}",
        percent=True,
    )


def main() -> None:
    args = parse_args()
    workflow_dir = Path(args.workflow_dir)
    picture_dir = Path(args.picture_dir)
    picture_dir.mkdir(parents=True, exist_ok=True)

    error_bars = ("std", "sem") if args.error_bar == "both" else (args.error_bar,)
    for error_bar in error_bars:
        plot_stage_performance(workflow_dir, picture_dir, args.file_format, error_bar=error_bar,
                               ylim=(0.015, 0.030),
                               ) # Figure 1
        plot_rfe_plateau(
            workflow_dir,
            picture_dir,
            args.file_format,
            ylim=(0.014, 0.027),
            error_bar=error_bar,
        ) # Figure 3

    plot_workflow_stage_family(workflow_dir, picture_dir, args.file_format) # Figure 2

if __name__ == "__main__":
    main()
