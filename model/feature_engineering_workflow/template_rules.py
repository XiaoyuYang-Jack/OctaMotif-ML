from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd


OCTAHEDRAL_COORDINATION = 6

ANCHOR_FEATURES = {
    "electronegativity_b": "MagpieData mean Electronegativity _B",
    "electronegativity_x": "MagpieData mean Electronegativity _X",
    "covalent_radius_b": "MagpieData mean CovalentRadius _B",
    "covalent_radius_x": "MagpieData mean CovalentRadius _X",
    "ionic_radius_b": "B_ionic_radius",
    "ionic_radius_x": "X_ionic_radius",
    "shannon_radius_b": "B_shannon_rad",
    "shannon_radius_x": "X_shannon_rad",
    "ground_state_volume_b": "MagpieData mean GSvolume_pa _B",
    "ground_state_volume_x": "MagpieData mean GSvolume_pa _X",
    "packing_fraction": "packing fraction",
}

PROTECTED_ANCHOR_FEATURES = tuple(ANCHOR_FEATURES.values())

# This set covers the hand-built motif descriptors used in the manuscript
# feature-engineering script. Required templates are generated before any cap
# on optional candidates is applied.
REQUIRED_MANUSCRIPT_TEMPLATES = {
    "coval_r_6X-B",
    "EnB/rB",
    "EnX/rX",
    "EnB/rB-6*EnX/rX",
    "Electronegativity_B-6X",
    "(EnB-6*EnX)/coval_r_B-6X",
    "Electronegativity_B-X",
    "(vB+6*vX)/pf",
    "EnB/IrB",
    "EnX/IrX",
    "EnB/IrB-6*EnX/IrX",
    "I_r_6X-B",
    "I_r_X-B",
    "(EnB-6*EnX)/Ir_B-6X",
    "cr_B_cr_X",
    "Ir_B_Ir_X",
    "SIr_B_SIr_X",
    "EnB/SIrB",
    "EnX/SIrX",
    "EnB/SIrB-6*EnX/SIrX",
    "SI_r_6X-B",
    "(EnB-6*EnX)/SIr_B-6X",
}


LEVEL1_TEMPLATES = [
    {
        "name": "coval_r_6X-B",
        "kind": "weighted_rhs_minus_lhs",
        "lhs": ANCHOR_FEATURES["covalent_radius_b"],
        "rhs": ANCHOR_FEATURES["covalent_radius_x"],
        "weight": OCTAHEDRAL_COORDINATION,
        "formula": "-covalent_radius_B + 6*covalent_radius_X",
    },
    {
        "name": "EnB/rB",
        "kind": "ratio",
        "num": ANCHOR_FEATURES["electronegativity_b"],
        "den": ANCHOR_FEATURES["covalent_radius_b"],
        "formula": "electronegativity_B / covalent_radius_B",
    },
    {
        "name": "EnX/rX",
        "kind": "ratio",
        "num": ANCHOR_FEATURES["electronegativity_x"],
        "den": ANCHOR_FEATURES["covalent_radius_x"],
        "formula": "electronegativity_X / covalent_radius_X",
    },
    {
        "name": "Electronegativity_B-6X",
        "kind": "weighted_lhs_minus_rhs",
        "lhs": ANCHOR_FEATURES["electronegativity_b"],
        "rhs": ANCHOR_FEATURES["electronegativity_x"],
        "weight": OCTAHEDRAL_COORDINATION,
        "formula": "electronegativity_B - 6*electronegativity_X",
    },
    {
        "name": "Electronegativity_B-X",
        "kind": "sub",
        "lhs": ANCHOR_FEATURES["electronegativity_b"],
        "rhs": ANCHOR_FEATURES["electronegativity_x"],
        "formula": "electronegativity_B - electronegativity_X",
    },
    {
        "name": "(vB+6*vX)/pf",
        "kind": "weighted_sum_ratio",
        "lhs": ANCHOR_FEATURES["ground_state_volume_b"],
        "rhs": ANCHOR_FEATURES["ground_state_volume_x"],
        "den": ANCHOR_FEATURES["packing_fraction"],
        "weight": OCTAHEDRAL_COORDINATION,
        "formula": "(GSvolume_B + 6*GSvolume_X) / packing_fraction",
    },
    {
        "name": "EnB/IrB",
        "kind": "ratio",
        "num": ANCHOR_FEATURES["electronegativity_b"],
        "den": ANCHOR_FEATURES["ionic_radius_b"],
        "formula": "electronegativity_B / ionic_radius_B",
    },
    {
        "name": "EnX/IrX",
        "kind": "ratio",
        "num": ANCHOR_FEATURES["electronegativity_x"],
        "den": ANCHOR_FEATURES["ionic_radius_x"],
        "formula": "electronegativity_X / ionic_radius_X",
    },
    {
        "name": "I_r_6X-B",
        "kind": "weighted_rhs_minus_lhs",
        "lhs": ANCHOR_FEATURES["ionic_radius_b"],
        "rhs": ANCHOR_FEATURES["ionic_radius_x"],
        "weight": OCTAHEDRAL_COORDINATION,
        "formula": "-ionic_radius_B + 6*ionic_radius_X",
    },
    {
        "name": "I_r_X-B",
        "kind": "weighted_rhs_minus_lhs",
        "lhs": ANCHOR_FEATURES["ionic_radius_b"],
        "rhs": ANCHOR_FEATURES["ionic_radius_x"],
        "weight": 1,
        "formula": "-ionic_radius_B + ionic_radius_X",
    },
    {
        "name": "cr_B_cr_X",
        "kind": "ratio",
        "num": ANCHOR_FEATURES["covalent_radius_b"],
        "den": ANCHOR_FEATURES["covalent_radius_x"],
        "formula": "covalent_radius_B / covalent_radius_X",
    },
    {
        "name": "Ir_B_Ir_X",
        "kind": "ratio",
        "num": ANCHOR_FEATURES["ionic_radius_b"],
        "den": ANCHOR_FEATURES["ionic_radius_x"],
        "formula": "ionic_radius_B / ionic_radius_X",
    },
    {
        "name": "SIr_B_SIr_X",
        "kind": "ratio",
        "num": ANCHOR_FEATURES["shannon_radius_b"],
        "den": ANCHOR_FEATURES["shannon_radius_x"],
        "formula": "shannon_radius_B / shannon_radius_X",
    },
    {
        "name": "EnB/SIrB",
        "kind": "ratio",
        "num": ANCHOR_FEATURES["electronegativity_b"],
        "den": ANCHOR_FEATURES["shannon_radius_b"],
        "formula": "electronegativity_B / shannon_radius_B",
    },
    {
        "name": "EnX/SIrX",
        "kind": "ratio",
        "num": ANCHOR_FEATURES["electronegativity_x"],
        "den": ANCHOR_FEATURES["shannon_radius_x"],
        "formula": "electronegativity_X / shannon_radius_X",
    },
    {
        "name": "SI_r_6X-B",
        "kind": "weighted_rhs_minus_lhs",
        "lhs": ANCHOR_FEATURES["shannon_radius_b"],
        "rhs": ANCHOR_FEATURES["shannon_radius_x"],
        "weight": OCTAHEDRAL_COORDINATION,
        "formula": "-shannon_radius_B + 6*shannon_radius_X",
    },
]


LEVEL2_TEMPLATES = [
    {
        "name": "EnB/rB-6*EnX/rX",
        "kind": "weighted_lhs_minus_rhs",
        "lhs": "EnB/rB",
        "rhs": "EnX/rX",
        "weight": OCTAHEDRAL_COORDINATION,
        "formula": "(EnB/rB) - 6*(EnX/rX)",
    },
    {
        "name": "(EnB-6*EnX)/coval_r_B-6X",
        "kind": "ratio",
        "num": "Electronegativity_B-6X",
        "den": "coval_r_6X-B",
        "formula": "(electronegativity_B - 6*electronegativity_X) / (-covalent_radius_B + 6*covalent_radius_X)",
    },
    {
        "name": "EnB/IrB-6*EnX/IrX",
        "kind": "weighted_lhs_minus_rhs",
        "lhs": "EnB/IrB",
        "rhs": "EnX/IrX",
        "weight": OCTAHEDRAL_COORDINATION,
        "formula": "(EnB/IrB) - 6*(EnX/IrX)",
    },
    {
        "name": "(EnB-6*EnX)/Ir_B-6X",
        "kind": "ratio",
        "num": "Electronegativity_B-6X",
        "den": "I_r_6X-B",
        "formula": "(electronegativity_B - 6*electronegativity_X) / (-ionic_radius_B + 6*ionic_radius_X)",
    },
    {
        "name": "EnB/SIrB-6*EnX/SIrX",
        "kind": "weighted_lhs_minus_rhs",
        "lhs": "EnB/SIrB",
        "rhs": "EnX/SIrX",
        "weight": OCTAHEDRAL_COORDINATION,
        "formula": "(EnB/SIrB) - 6*(EnX/SIrX)",
    },
    {
        "name": "(EnB-6*EnX)/SIr_B-6X",
        "kind": "ratio",
        "num": "Electronegativity_B-6X",
        "den": "SI_r_6X-B",
        "formula": "(electronegativity_B - 6*electronegativity_X) / (-shannon_radius_B + 6*shannon_radius_X)",
    },
]


@dataclass(frozen=True)
class FeatureMeta:
    family: str
    unit_group: str
    scope: str
    positive_only: bool


def infer_feature_family(name: str) -> str:
    if name in {
        "EnB/rB",
        "EnX/rX",
        "EnB/rB-6*EnX/rX",
    }:
        return "electronegativity_covalent_radius_ratio"
    if name in {
        "EnB/IrB",
        "EnX/IrX",
        "EnB/IrB-6*EnX/IrX",
    }:
        return "electronegativity_ionic_radius_ratio"
    if name in {
        "EnB/SIrB",
        "EnX/SIrX",
        "EnB/SIrB-6*EnX/SIrX",
    }:
        return "electronegativity_shannon_radius_ratio"
    if name in {
        "Electronegativity_B-6X",
        "Electronegativity_B-X",
    }:
        return "electronegativity_contrast"
    if name in {
        "coval_r_6X-B",
    }:
        return "covalent_radius_contrast"
    if name in {
        "I_r_6X-B",
        "I_r_X-B",
    }:
        return "ionic_radius_contrast"
    if name in {
        "SI_r_6X-B",
    }:
        return "shannon_radius_contrast"
    if name in {
        "cr_B_cr_X",
    }:
        return "covalent_radius_ratio"
    if name in {
        "Ir_B_Ir_X",
    }:
        return "ionic_radius_ratio"
    if name in {
        "SIr_B_SIr_X",
    }:
        return "shannon_radius_ratio"
    if name in {
        "(EnB-6*EnX)/coval_r_B-6X",
    }:
        return "electronegativity_covalent_radius_contrast_ratio"
    if name in {
        "(EnB-6*EnX)/Ir_B-6X",
    }:
        return "electronegativity_ionic_radius_contrast_ratio"
    if name in {
        "(EnB-6*EnX)/SIr_B-6X",
    }:
        return "electronegativity_shannon_radius_contrast_ratio"
    if "local_difference_in_" in name:
        suffix = name.replace("local_difference_in_", "")
        return f"local_difference:{infer_feature_family(suffix)}"
    if "Electronegativity" in name or name.startswith("En"):
        return "electronegativity"
    if "CovalentRadius" in name or "coval_r" in name or name.startswith("cr_"):
        return "covalent_radius"
    if "shannon_rad" in name or "SIr" in name or "SI_r_" in name:
        return "shannon_radius"
    if "ionic_radius" in name or "I_r_" in name or "Ir_" in name:
        return "ionic_radius"
    if "GSbandgap" in name or name.endswith("gap") or "gap_" in name:
        return "bandgap"
    if "NdValence" in name:
        return "d_valence"
    if "NpValence" in name:
        return "p_valence"
    if "NsValence" in name:
        return "s_valence"
    if "NValence" in name:
        return "valence"
    if "NdUnfilled" in name:
        return "d_unfilled"
    if "NpUnfilled" in name:
        return "p_unfilled"
    if "NsUnfilled" in name:
        return "s_unfilled"
    if "NUnfilled" in name:
        return "unfilled"
    if "GSvolume" in name or "(vB+6*vX)/pf" in name:
        return "volume"
    if "ewald" in name.lower():
        return "ewald"
    if "density" in name:
        return "density"
    if "packing fraction" in name or name.endswith("/pf"):
        return "packing_fraction"
    if "neighbor distance" in name:
        return "neighbor_distance"
    return "other"


def infer_unit_group(family: str) -> str:
    if family in {
        "electronegativity_covalent_radius_ratio",
        "electronegativity_covalent_radius_contrast_ratio",
    }:
        return "electronegativity_per_covalent_radius"
    if family in {
        "electronegativity_ionic_radius_ratio",
        "electronegativity_ionic_radius_contrast_ratio",
    }:
        return "electronegativity_per_ionic_radius"
    if family in {
        "electronegativity_shannon_radius_ratio",
        "electronegativity_shannon_radius_contrast_ratio",
    }:
        return "electronegativity_per_shannon_radius"
    if family.endswith("_contrast"):
        if family.startswith("electronegativity"):
            return "electronegativity"
        if "radius" in family:
            return "length_like"
    if family.endswith("_ratio"):
        return "dimensionless"
    if "radius" in family or family in {"volume", "neighbor_distance"}:
        return "length_like"
    if family in {"bandgap", "ewald"}:
        return "energy_like"
    if "valence" in family or family == "unfilled":
        return "electron_count"
    if family == "electronegativity":
        return "electronegativity"
    if family == "density":
        return "density"
    if family == "packing_fraction":
        return "dimensionless"
    if family.startswith("local_difference:"):
        return infer_unit_group(family.split(":", 1)[1])
    return "dimensionless"


def infer_scope(name: str) -> str:
    if "_B" in name and "_X" not in name:
        return "B"
    if "_X" in name and "_B" not in name:
        return "X"
    if "local_difference" in name or "allsites" in name:
        return "motif"
    if "formula" in name:
        return "formula"
    if "density" in name or "ewald" in name or "packing fraction" in name:
        return "crystal"
    if "B-" in name or "B/" in name or "6*EnX" in name or "6X" in name:
        return "motif"
    return "global"


def infer_feature_meta(name: str) -> FeatureMeta:
    family = infer_feature_family(name)
    unit_group = infer_unit_group(family)
    positive_only = (
        unit_group
        in {
            "length_like",
            "energy_like",
            "density",
            "electronegativity_per_covalent_radius",
            "electronegativity_per_ionic_radius",
            "electronegativity_per_shannon_radius",
        }
        or family
        in {
            "covalent_radius_ratio",
            "ionic_radius_ratio",
            "shannon_radius_ratio",
            "packing_fraction",
        }
        or name in {
            "packing fraction",
        }
    )
    return FeatureMeta(
        family=family,
        unit_group=unit_group,
        scope=infer_scope(name),
        positive_only=positive_only,
    )


def available_anchor_features(columns: Iterable[str]) -> Dict[str, str]:
    columns = set(columns)
    return {key: value for key, value in ANCHOR_FEATURES.items() if value in columns}


def template_parents(template: Dict[str, object]) -> Tuple[str, ...]:
    parent_keys = ("lhs", "rhs", "num", "den")
    parents: List[str] = []
    for key in parent_keys:
        value = template.get(key)
        if isinstance(value, str):
            parents.append(value)
    return tuple(dict.fromkeys(parents))


def compute_template_series(template: Dict[str, object], data_frame: pd.DataFrame) -> pd.Series:
    kind = template["kind"]

    if kind == "sub":
        return data_frame[template["lhs"]] - data_frame[template["rhs"]]
    if kind == "weighted_lhs_minus_rhs":
        return data_frame[template["lhs"]] - template["weight"] * data_frame[template["rhs"]]
    if kind == "weighted_rhs_minus_lhs":
        return -data_frame[template["lhs"]] + template["weight"] * data_frame[template["rhs"]]
    if kind == "ratio":
        return data_frame[template["num"]] / data_frame[template["den"]]
    if kind == "weighted_sum_ratio":
        numerator = data_frame[template["lhs"]] + template["weight"] * data_frame[template["rhs"]]
        return numerator / data_frame[template["den"]]

    raise ValueError(f"Unsupported template kind: {kind}")


def make_lineage_row(
    feature: str,
    stage: str,
    complexity: int,
    formula: str,
    parents: Iterable[str],
    template_rule: str,
    required_manuscript_template: bool,
    protected_anchor: bool = False,
) -> Dict[str, object]:
    metadata = infer_feature_meta(feature)
    return {
        "feature": feature,
        "stage": stage,
        "complexity": complexity,
        "formula": formula,
        "parents": " | ".join(parents),
        "template_rule": template_rule,
        "required_manuscript_template": bool(required_manuscript_template),
        "protected_anchor": bool(protected_anchor),
        "family": metadata.family,
        "unit_group": metadata.unit_group,
    }


def build_template_features(
    data_frame: pd.DataFrame,
    templates: Iterable[Dict[str, object]],
    stage: str,
    complexity: int,
) -> Tuple[Dict[str, pd.Series], Dict[str, FeatureMeta], List[Dict[str, object]], List[str]]:
    generated: Dict[str, pd.Series] = {}
    metadata: Dict[str, FeatureMeta] = {}
    lineage_rows: List[Dict[str, object]] = []
    missing_templates: List[str] = []

    for template in templates:
        name = str(template["name"])
        try:
            series = compute_template_series(template, data_frame)
        except (KeyError, ValueError):
            missing_templates.append(name)
            continue

        generated[name] = series
        metadata[name] = infer_feature_meta(name)
        lineage_rows.append(
            make_lineage_row(
                feature=name,
                stage=stage,
                complexity=complexity,
                formula=str(template["formula"]),
                parents=template_parents(template),
                template_rule=str(template["kind"]),
                required_manuscript_template=name in REQUIRED_MANUSCRIPT_TEMPLATES,
            )
        )

    return generated, metadata, lineage_rows, missing_templates


def build_level1_template_features(
    data_frame: pd.DataFrame,
) -> Tuple[Dict[str, pd.Series], Dict[str, FeatureMeta], List[Dict[str, object]], List[str]]:
    return build_template_features(data_frame, LEVEL1_TEMPLATES, "level1_template", 1)


def build_level2_template_features(
    data_frame: pd.DataFrame,
) -> Tuple[Dict[str, pd.Series], Dict[str, FeatureMeta], List[Dict[str, object]], List[str]]:
    return build_template_features(data_frame, LEVEL2_TEMPLATES, "level2_template", 2)
