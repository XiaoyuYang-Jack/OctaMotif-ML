# OctaMotif-ML (Octahedral Motif–Guided Interpretable ML)

Code and data accompanying the manuscript:

**Octahedral Motif-Guided Design of Optoelectronic Semiconductors via Interpretable Machine Learning**  
Xiaoyu Yang\#, Wenyue Zhao\#, Xin He, Weidong Fei, Yuanhui Sun\*, Yu Zhao\*, Lijun Zhang\*  
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
├── model
│   ├── feature_selection_combination
│   ├── cross_validation
│   ├── plot_results
│   └── symbolic_regression
└── README.md
```

- `data/dataset.csv` stores the structure IDs (`file_name`) and targets.
- `data/structure_files/` contains the corresponding structure files. Please unzip the `structure_files.zip` file before using.
- `data/feature_construction/` contains scripts for octahedral motif identification and descriptor construction.
- `model/` contains feature selection & combination, model training & cross-validation, plotting, and symbolic regression (SISSO) workflows.

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

### 3) Iterative feature selection and algebraic combination

Run:

```bash
cd model/feature_selection_combination
python feature_selection_combination.py
```

This script reads `feature_set.csv`, performs iterative recursive feature elimination (RFE) and algebraic feature combination, and outputs:

- `best_feature_set.csv`
- RFE logs/results saved to files in the same folder

---

### 4) Model training and cross-validation (GBRT)

Run:

```bash
cd model/cross_validation
python plot_train_results.py
```

This script reads `best_feature_set.csv` and performs:

- one random train/test split training
- 10-fold cross-validation

Cross-validation outputs are saved in:

- `model/cross_validation/cross_validation_results/`

---

### 5) Visualization of results

Run:

```bash
cd model/plot_results
python plot_train_results.py
```

This script visualizes outputs from feature engineering, model training, and cross-validation.

---

### 6) Symbolic regression with SISSO

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

## Citation

If you use this repository, please cite the associated manuscript:

*Octahedral Motif-Guided Design of Optoelectronic Semiconductors via Interpretable Machine Learning* (details to be updated upon publication).
