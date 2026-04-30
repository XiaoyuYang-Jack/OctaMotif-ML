import argparse
from cProfile import label
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import GradientBoostingRegressor


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = CURRENT_DIR.parent.parent / "data" / "feature_construction" / "feature_set.csv"


class RegressionMetrics():
    """
    Regression metrics

    Args:
        True_value: list, numpy, pandas.DataFrame
            Actual target properties of the material.
        Predicted_value: list, numpy, pandas.DataFrame
            Target properties of the material predicted by the model.
    """
    def __init__(self, True_value, Predicted_value):
        self.True_value = True_value
        self.Predicted_value = Predicted_value

    @property
    def mae(self):
        """
        Mean absoluate error.

        Returns: float
        """
        self._mae = np.sum(np.absolute(np.array(self.True_value) - np.array(self.Predicted_value)))/len(self.True_value)
        return self._mae

    @property
    def mse(self):
        """
        Mean square error.

        Returns: float
        """
        self._mse = np.sum((np.array(self.True_value) - np.array(self.Predicted_value))**2)/len(self.True_value)
        return self._mse

    @property
    def rmse(self):
        """
        Root mean square error.

        Returns: float
        """
        self._rmse = np.sqrt(self.mse)
        return self._rmse

    @property
    def r2(self):
        """
        Coefficient of determination.

        Returns： float
        """
        self._r2 = 1 - (np.sum((np.array(self.True_value) - np.array(self.Predicted_value))**2)/len(self.True_value))/np.var(self.True_value)
        return self._r2

    @property
    def msle(self):
        """
        Mean squared log error.

        Returns: float
        """
        self._msle = np.sum(np.log( 1 + np.array(self.True_value)) - np.log(1 + np.array(self.Predicted_value)))/len(self.True_value)
        return self._msle
    
    @property
    def rmsle(self):
        """
        Root mean squared log error.

        Returns: float
        """
        if self.msle < 0:
            raise ValueError('Mean Squared Logarithmic Error cannot be used when targets contain negative values!')
        else:
            self._rmsle = np.sqrt(self._msle)
        return self._rmsle


def fill_zero_rad(data):
    B_ionic_radius,X_ionic_radius = [], []
    for i in range(len(data)):
        if data['B_ionic_radius'][i] == 0 :
            B_ionic_radius.append(0.1)
        else : 
            B_ionic_radius.append(data['B_ionic_radius'][i])
        
        if data['X_ionic_radius'][i] == 0 :
            X_ionic_radius.append(0.1)
        else : 
            X_ionic_radius.append(data['X_ionic_radius'][i])
    data = data.drop(columns=['B_ionic_radius','X_ionic_radius'])
    data['B_ionic_radius'] = B_ionic_radius
    data['X_ionic_radius'] = X_ionic_radius
    
    return data


def data_clean(data):
    columns_with_strings = data.select_dtypes(include='object').columns
    data = data.drop(columns=columns_with_strings)
    data = data.dropna(axis='columns')
    data = data.loc[:, ~data.isin([np.inf, -np.inf]).any()]
    # fill the ionic radius 0 values with 0.1
    data = fill_zero_rad(data)

    return data

def define_model():
    hyperparameter = { 
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "loss": "squared_error",
        "verbose" : 1,
        "min_samples_split" : 2,
        "min_samples_leaf" : 1,
        }

    model = GradientBoostingRegressor(
        **hyperparameter
        )
    return model,hyperparameter


def summarize_fold_metrics(fold_metrics):
    summary = {}
    for metric in ["mae", "mse", "rmse", "r2"]:
        values = fold_metrics[metric]
        summary[f"mean_{metric}"] = values.mean()
        summary[f"std_{metric}"] = values.std(ddof=0)
        summary[f"sem_{metric}"] = values.std(ddof=0) / np.sqrt(len(values))
    return summary


def figure_name_for_error_bar(error_bar):
    if error_bar == "sem":
        return "feature_MSE_SEM.png"
    return "feature_MSE.png"


def plot_rfe_scores(scores, output_path, error_bar="std"):
    if isinstance(scores, (str, Path)):
        scores = pd.read_csv(scores)

    if "number_removed" in scores.columns:
        x_values = scores["number_removed"]
    else:
        x_values = range(len(scores))

    if "mean_mse" in scores.columns:
        y_values = scores["mean_mse"]
        error_column = f"{error_bar}_mse"
        y_error = scores[error_column] if error_column in scores.columns else None
    else:
        y_values = scores["MSE"]
        y_error = None

    plt.figure(figsize=(9, 6))
    ax = plt.subplot(111)
    plt.xlabel("Number of features removed", fontsize=14, )
    plt.ylabel("Cross validation MSE", fontsize=14, )
    plt.tick_params(labelsize=14)
    ax.errorbar(
        x_values,
        y_values,
        yerr=y_error,
        marker="o",
        markersize=6,
        c="mediumslateblue",
        mec="w",
        linewidth=2,
        capsize=4,
    )
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()


def feature_combination(data):

    data['coval_r_6X-B'] = -data['MagpieData mean CovalentRadius _B'] + 6*data['MagpieData mean CovalentRadius _X']
    data['EnB/rB'] = data['MagpieData mean Electronegativity _B']/data['MagpieData mean CovalentRadius _B']
    data['EnX/rX'] = data['MagpieData mean Electronegativity _X']/data['MagpieData mean CovalentRadius _X']
    data['EnB/rB-6*EnX/rX'] = data['MagpieData mean Electronegativity _B']/data['MagpieData mean CovalentRadius _B'] - 6*data['MagpieData mean Electronegativity _X']/data['MagpieData mean CovalentRadius _X']
    data['Electronegativity_B-6X'] = data['MagpieData mean Electronegativity _B'] - 6*data['MagpieData mean Electronegativity _X']
    data['(EnB-6*EnX)/coval_r_B-6X'] = data['Electronegativity_B-6X']/data['coval_r_6X-B']
    data['Electronegativity_B-X'] = data['MagpieData mean Electronegativity _B'] - data['MagpieData mean Electronegativity _X']
    data['(vB+6*vX)/pf'] = (data['MagpieData mean GSvolume_pa _B'] +6* data['MagpieData mean GSvolume_pa _X'])/data['packing fraction']
    data['EnB/IrB-6*EnX/IrX'] = data['MagpieData mean Electronegativity _B']/data['B_ionic_radius'] - 6*data['MagpieData mean Electronegativity _X']/data['X_ionic_radius']
    data['I_r_6X-B'] = -data['B_ionic_radius'] + 6*data['X_ionic_radius']
    data['I_r_X-B'] = -data['B_ionic_radius'] + data['X_ionic_radius']
    data['(EnB-6*EnX)/Ir_B-6X'] = data['Electronegativity_B-6X']/data['I_r_6X-B']
    # data['EnB/IrB-n*EnX/IrX'] = data['MagpieData mean Electronegativity _B']/data['B_ionic_radius'] - data['Bnum/Anum']*data['MagpieData mean Electronegativity _X']/data['X_ionic_radius']
    data['cr_B_cr_X'] = data['MagpieData mean CovalentRadius _B']/data['MagpieData mean CovalentRadius _X']
    data['Ir_B_Ir_X'] = data['B_ionic_radius']/data['X_ionic_radius']
    data['SIr_B_SIr_X'] = data['B_shannon_rad']/data['X_shannon_rad']
    data['EnB/SIrB-6*EnX/SIrX'] = data['MagpieData mean Electronegativity _B']/data['B_shannon_rad'] - 6*data['MagpieData mean Electronegativity _X']/data['X_shannon_rad']
    data['EnB/SIrB'] = data['MagpieData mean Electronegativity _B']/data['B_shannon_rad']
    data['EnX/SIrX'] = data['MagpieData mean Electronegativity _X']/data['X_shannon_rad']
    data['SI_r_6X-B'] = -data['B_shannon_rad'] + 6*data['X_shannon_rad']
    data['(EnB-6*EnX)/SIr_B-6X'] = data['Electronegativity_B-6X']/data['SI_r_6X-B']

    return data



def feature_selection(data):
    data = data[[ # num=38
        'EnB/rB-6*EnX/rX', # complexity=2
        'MagpieData mean NdValence formula',
        'MagpieData minimum Electronegativity formula',
        'MagpieData maximum GSbandgap formula',
        'MagpieData range CovalentRadius formula',
        'Electronegativity_B-6X',  # complexity=1
        'MagpieData mode Electronegativity formula',
        'MagpieData range NdValence formula',
        'local_difference_in_Electronegativity',
        'MagpieData maximum NdValence formula',
        'MagpieData range NUnfilled formula',
        'MagpieData avg_dev GSbandgap formula',
        'MagpieData avg_dev NUnfilled formula',
        'mean neighbor distance variation',
        'MagpieData mean NsUnfilled formula',
        'local_difference_in_CovalentRadius',
        'MagpieData mean NValence formula',
        'EnB/SIrB',  # complexity=1
        'MagpieData mode GSvolume_pa _B',
        'MagpieData mean NpValence formula',
        'MagpieData mean NUnfilled formula',
        'MagpieData maximum NdUnfilled formula',
        'density',
        'std_dev ewald_site_energy',
        'MagpieData mean Electronegativity formula',
        'ewald_energy_per_atom',
        '(EnB-6*EnX)/coval_r_B-6X',  # complexity=2
        'local_difference_in_Electronegativity_allsites',
        'local_difference_in_CovalentRadius_allsites',
        'local_difference_in_NdValence_allsites',
        'local_difference_in_GSbandgap_allsites',
        'EnB/SIrB-6*EnX/SIrX',  # complexity=2
        'EnX/SIrX',  # complexity=1
        'EnB/rB', # complexity=1
        'EnX/rX', # complexity=1
        'SIr_B_SIr_X', # complexity=1
        'cr_B_cr_X', # complexity=1
        '(EnB-6*EnX)/SIr_B-6X', # complexity=2
        ]]

    return data


def RFE(data,data_target,model,hyperparameter,output_dir=CURRENT_DIR,error_bar="std"):
    '''Recursive Feature Elimination (RFE) to select informative descriptors'''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def cross_val_kfold(X,y,model,hyperparameter):

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        k = 0
        FI_list = []
        fold_rows = []
        for train_index, test_index in kf.split(X):
            k = k+1
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)

            test_score = np.zeros((hyperparameter["n_estimators"],), dtype=np.float64)
            for i, y_pred in enumerate(model.staged_predict(X_test)):
                test_score[i] = mean_squared_error(y_test, y_pred)
            
            data_loss= pd.DataFrame()
            data_loss['train_score_'] = model.train_score_
            data_loss['test_score'] = test_score

            rms=RegressionMetrics(y_test,y_pred)
            fold_rows.append(
                {
                    "fold": k,
                    "mae": rms.mae,
                    "mse": rms.mse,
                    "rmse": rms.rmse,
                    "r2": rms.r2,
                }
            )

            FI_list.append(model.feature_importances_)

        FI = np.mean(FI_list,axis=0)
        fold_metrics = pd.DataFrame(fold_rows)
        summary = summarize_fold_metrics(fold_metrics)

        return summary,FI,fold_metrics


    select_feature = data.columns.values

    score_rows = []
    split_score_rows = []

    with open(output_dir / 'features_select.txt', 'w') as file:

        for i in range(len(data.columns.values)):
            if i < len(data.columns.values)-1 : 

                select_data = data[select_feature].copy()
                
                scaled_data = MinMaxScaler(feature_range=(0,1)).fit_transform(select_data)

                summary,feature_importance,fold_metrics = cross_val_kfold(X=scaled_data,y=data_target,model=model,hyperparameter=hyperparameter)

                number_removed = i
                n_features = len(select_feature)
                print('Feature number: ',n_features,',MSE : ',round(summary["mean_mse"], 4))
                file.write('\n'+'Feature number: '+str(n_features)+','+',MSE : '+str(summary["mean_mse"])+','+'\n')

                sorted_idx_ = np.argsort(-feature_importance)
                eliminated_feature = np.array(select_feature)[sorted_idx_][-1]

                score_row = {
                    "number_removed": number_removed,
                    "n_features": n_features,
                    "removed_feature": eliminated_feature,
                    "features": " | ".join(select_feature),
                    **summary,
                    "MAE": summary["mean_mae"],
                    "MSE": summary["mean_mse"],
                    "RMSE": summary["mean_rmse"],
                    "R2": summary["mean_r2"],
                }
                score_rows.append(score_row)

                fold_metrics.insert(0, "number_removed", number_removed)
                fold_metrics.insert(1, "n_features", n_features)
                fold_metrics.insert(2, "removed_feature", eliminated_feature)
                fold_metrics["features"] = " | ".join(select_feature)
                split_score_rows.append(fold_metrics)

                for f in select_feature[sorted_idx_] : 
                    print(f)
                    file.write(f+','+'\n')
                print('Eliminate: ',eliminated_feature)
                select_feature = np.array(select_feature)[sorted_idx_][0:-1] 
                print('select_feature : ',len(select_feature))
                print(' ')
                
                
    data_scores = pd.DataFrame(score_rows)
    data_scores.to_csv(output_dir / 'feature_select_scores.csv', index=False)
    if split_score_rows:
        pd.concat(split_score_rows, ignore_index=True).to_csv(
            output_dir / 'feature_select_split_scores.csv',
            index=False,
        )

    print(data_scores[["n_features", "mean_mae", "mean_mse", "mean_rmse", "mean_r2"]])
    plot_rfe_scores(
        data_scores,
        output_dir / figure_name_for_error_bar(error_bar),
        error_bar=error_bar,
    )


def final_compact_featureset(data):
    # Feature number: 34,MSE : 0.0237
    data = data[[
    'EnB/rB-6*EnX/rX',
    'MagpieData mean NdValence formula',
    'local_difference_in_NdValence_allsites',
    'local_difference_in_Electronegativity_allsites',
    'MagpieData maximum GSbandgap formula',
    'MagpieData minimum Electronegativity formula',
    'Electronegativity_B-6X',
    'MagpieData mode Electronegativity formula',
    'MagpieData range NdValence formula',
    'MagpieData range NUnfilled formula',
    'MagpieData maximum NdValence formula',
    'MagpieData range CovalentRadius formula',
    'MagpieData avg_dev NUnfilled formula',
    'MagpieData mean NsUnfilled formula',
    'MagpieData maximum NdUnfilled formula',
    'local_difference_in_Electronegativity',
    'MagpieData mode GSvolume_pa _B',
    'MagpieData avg_dev GSbandgap formula',
    'mean neighbor distance variation',
    'MagpieData mean NValence formula',
    'MagpieData mean NpValence formula',
    'local_difference_in_GSbandgap_allsites',
    'local_difference_in_CovalentRadius',
    'EnB/SIrB-6*EnX/SIrX',
    'std_dev ewald_site_energy',
    'EnB/SIrB',
    'density',
    '(EnB-6*EnX)/coval_r_B-6X',
    'ewald_energy_per_atom',
    'MagpieData mean Electronegativity formula',
    'local_difference_in_CovalentRadius_allsites',
    'MagpieData mean NUnfilled formula',
    'SIr_B_SIr_X',
    'EnB/rB',
    ]]
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output-dir", default=str(CURRENT_DIR))
    parser.add_argument("--error-bar", default="std", choices=["std", "sem"])
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


if __name__ == '__main__' :
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        plot_rfe_scores(
            output_dir / 'feature_select_scores.csv',
            output_dir / figure_name_for_error_bar(args.error_bar),
            error_bar=args.error_bar,
        )
        raise SystemExit

    # 0. load data
    data = pd.read_csv(args.data_path).drop(columns=['file_name','formula'])
    data_target = data.pop('formation_energy') 

    # 1. data clean
    data = data_clean(data)

    # 2. define model
    model,hyperparameter = define_model()
    
    # 3. feature combination
    data = feature_combination(data)

    # 4. initial feature selection
    data = feature_selection(data)
    print(data)

    # 5. Recursive Feature Elimination
    RFE(
        data=data,
        data_target=data_target,
        model=model,
        hyperparameter=hyperparameter,
        output_dir=output_dir,
        error_bar=args.error_bar,
    )

    # 6. output selected feature set
    data = final_compact_featureset(data)
    data.insert(0,'formation_energy',data_target)
    data.to_csv(output_dir / 'final_compact_feature_set.csv',index=False)
