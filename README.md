# OctaMotif-ML (Octahedral Motif–Guided Interpretable ML)

Code and data accompanying the manuscript:

**Octahedral Motif-Guided Design of Optoelectronic Semiconductors via Interpretable Machine Learning**  
Xiaoyu Yang\*#, Wenyue Zhao\*#, Xin He, Weidong Fei, Yuanhui Sun\*, Yu Zhao\*, Lijun Zhang\*  
\# Equal contribution  
\* Corresponding authors: sunyh@szlab.ac.cn; yuzhao@hit.edu.cn; lijun_zhang@jlu.edu.cn

---

## Repository Structure

```text
.
├── data
│   ├── dataset.csv
│   ├── feature_construction
│   └── structure_files
├── material_explore_results
├── model
│   ├── feature_engineering_workflow
│   ├── feature_selection_combination
│   ├── cross_validation
│   ├── plot_results
│   └── symbolic_regression
└── README.md
```

- `data/dataset.csv` stores the structure IDs (`file_name`) and targets.
- `data/structure_files/` contains the corresponding structure files. Please unzip the `structure_files.zip` file before using.
- `data/feature_construction/` contains scripts for octahedral motif identification and descriptor construction.
- `model/feature_engineering_workflow/` contains the staged automated workflow, template rules, workflow outputs, and plotting scripts for the feature-engineering analysis.
- `model/` contains feature selection & combination, model training & cross-validation, plotting, and symbolic regression (SISSO) workflows.
- `material_explore_results/` contains the final stable semiconductor candidates and solar optoelectronic material candidates, together with their CIF files.

---

## Requirements

### Python packages

Install via `pip`:

- numpy
- pandas
- matplotlib
- pymatgen
- matminer
- scikit-learn
- mendeleev

Example:

```bash
pip install numpy pandas matplotlib pymatgen matminer scikit-learn mendeleev
```

### Symbolic regression (optional)

This work uses **SISSO** for symbolic regression:

- SISSO repository: https://github.com/rouyang2017/SISSO
- Please follow SISSO’s official installation and usage instructions.

---

## Usage

### 1) Octahedral motif identification

The dataset is stored in `data/`. `dataset.csv` provides `file_name` for each crystal, and the corresponding structure file is in `data/structure_files/`.

Run:

```bash
cd data/feature_construction
python identify_octa.py
```

This script reads `dataset.csv`, identifies octahedral sites and associated chemistry, and writes the results to:

- `dataset_octa.csv`

**Note (required patch):**
To ensure the script runs correctly, modify the return value of `local_env.site_is_of_motif_type` to:

```python
return neighs_cent, motif_type
```

---

### 2) Descriptor construction (feature construction)

Run:

```bash
cd data/feature_construction
python get_feature.py
```

This script reads `dataset_octa.csv` and constructs the full feature set (definitions are provided in the main text and Supplementary Information). Output:

- `feature_set.csv`

**Note (required patch for matminer `ElementProperty`):**
To use `ElementProperty` from `matminer`, apply the following modifications:

1) Add an argument `name` in `__init__` and store it:

```python
self.name = name
```

2) Append `name` to feature labels in `feature_labels()`:

```python
def feature_labels(self):
    labels = []
    name = self.name
    for attr in self.features:
        src = self.data_source.__class__.__name__
        for stat in self.stats:
            labels.append(f"{src} {stat} {attr} {name}")
    return labels
```

---

### 3) Automated feature-engineering workflow

The folder `model/feature_engineering_workflow/` provides a transparent implementation of the staged feature-engineering procedure used in this work. It contains three main scripts:

- `template_rules.py`
- `automated_workflow.py`
- `plot_feature_workflow_results.py`

These scripts document how the descriptor space evolves from the initial raw pool to the final compact motif-informed pool and allow readers to reproduce the staged feature-engineering analysis directly.

The script `template_rules.py` is not intended to be run directly. It defines the protected motif anchors, level-1 and level-2 template rules, feature-family inference, and lineage metadata used by the workflow. Its key inputs are the predefined primitive motif anchors and template expressions hard-coded in the script, and its outputs are the generated template-feature definitions and metadata returned internally to `automated_workflow.py`.

Run the workflow:

```bash
cd model/feature_engineering_workflow
python automated_workflow.py --output-dir workflow_outputs
```

This script reads the manuscript final feature set for alignment/reference analysis:

- `model/feature_selection_combination/final_compact_feature_set.csv`

The script performs repeated raw-feature selection, motif-guided level-1 and level-2 feature generation, stage-wise GBRT evaluation, and RFE from the 50-feature motif-informed pool to the compact 38-feature pool. Key outputs are written to:

- `model/feature_engineering_workflow/workflow_outputs/clean_feature_pool.csv`
- `model/feature_engineering_workflow/workflow_outputs/stage1_base_pool.csv`
- `model/feature_engineering_workflow/workflow_outputs/stage2_first_order_pool.csv`
- `model/feature_engineering_workflow/workflow_outputs/stage3_second_order_pool.csv`
- `model/feature_engineering_workflow/workflow_outputs/automated_rfe_38_feature_set.csv`
- `model/feature_engineering_workflow/workflow_outputs/automated_rfe_38_ranking.csv`
- `model/feature_engineering_workflow/workflow_outputs/feature_lineage.csv`
- `model/feature_engineering_workflow/workflow_outputs/workflow_stage_summary.csv`
- `model/feature_engineering_workflow/workflow_outputs/workflow_summary.json`

Plot the workflow results:

```bash
python plot_feature_workflow_results.py
```

This script reads the workflow tables in:

- `model/feature_engineering_workflow/workflow_outputs/`

and writes the summary figures to:

- `model/feature_engineering_workflow/workflow_results_pictures/1_stage_wise_mse.png` or `1_stage_wise_mse_SEM.png`
- `model/feature_engineering_workflow/workflow_results_pictures/2_workflow_stage_feature_family.png`
- `model/feature_engineering_workflow/workflow_results_pictures/3_rfe_50_to_38_mse.png` or `3_rfe_50_to_38_mse_SEM.png`


### 4) Iterative feature selection and algebraic combination

Run:

```bash
cd model/feature_selection_combination
python feature_selection_combination.py
```

This script reads `feature_set.csv`, performs iterative recursive feature elimination (RFE) and algebraic feature combination, and outputs:

- `final_compact_feature_set.csv`
- RFE logs/results saved to files in the same folder

---

### 5) Model training and cross-validation (GBRT)

Run:

```bash
cd model/cross_validation
python plot_train_results.py
```

This script reads `final_compact_feature_set.csv` and performs:

- one random train/test split training
- 10-fold cross-validation

Cross-validation outputs are saved in:

- `model/cross_validation/cross_validation_results/`

---

### 6) Visualization of results

Run:

```bash
cd model/plot_results
python plot_train_results.py
```

This script visualizes outputs from feature engineering, model training, and cross-validation.

---

### 7) Symbolic regression with SISSO

First, install SISSO.

Then run:

```bash
cd model/symbolic_regression
mpirun -np <number_of_cores> SISSO > run.log
```

SISSO reads `train.dat` and outputs files such as:

- `SISSO.out` (and other SISSO output files)

To visualize SISSO results, run:

```bash
python plot-sisso.py
```

**Note:**
In this repo, SISSO feature names follow an earlier naming convention:

- `A` denotes the octahedral central atom (`B_Octa` in the manuscript)
- `B` denotes the octahedral coordinating atom (`X_Octa` in the manuscript)

---

### 8) Material exploration results

The folder `material_explore_results/` contains the final screening results reported in the Supplementary Tables.

- `stable_semiconductors_129.csv` lists 129 stable semiconductor candidates with `Ehull < 0 eV` and `bandgap > 0 eV`, corresponding to Supplementary Table 2.
- `stable_semiconductors_129_cif_files/` contains the CIF files for these 129 stable semiconductor candidates.
- `SOMs_19.csv` lists the 19 candidates selected from the 129 stable semiconductors with reduced effective mass `mu < 1` and optical absorption `I > 1000`, corresponding to Supplementary Table 3.
- `SOMs_19_cif_files/` contains the CIF files for these 19 optoelectronic material candidates.

---

## Citation

If you use this repository, please cite the associated manuscript:

*Octahedral Motif-Guided Design of Optoelectronic Semiconductors via Interpretable Machine Learning* (details to be updated upon publication).