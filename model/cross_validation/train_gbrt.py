import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler  
from matplotlib.ticker import MultipleLocator
import seaborn as sns

def fill_zero_rad(data):
    B_ionic_radius,X_ionic_radius = [], []
    for i in range(len(data)):
        if data['B_ionic_radius'][i] == 0 :
            B_ionic_radius.append(0.1)
        else : 
            B_ionic_radius.append(data['B_ionic_radius'][i])
        
        if data['B_ionic_radius'][i] == 0 :
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
        "n_estimators": 1000,
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

def plot_ytest_ypred(y_test,y_pred,model_name,s=70,alpha=0.7,color = 'royalblue',
                        xlabel='Actual Value',ylabel='Predictive Value',text_x=6, text_y=0.25,
                        x_max=None,x_min=None,y_max=None,y_min=None,x_locator=None,y_locator=None,
                        fontsize=25,):
    rms=RegressionMetrics(y_test,y_pred)
    mse = round(rms.mse,4)
    mae = round(rms.mae,4)
    r2 = round(rms.r2,4)
    rmse = round(rms.rmse,4)

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    plt.scatter(
        y_test, 
        y_pred, 
        color = color,
        s = s,
        marker = 'o',
        alpha=alpha,
        )
    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)

    if x_max != None:
        ax.set_xlim([x_min,x_max])
    if y_max != None:
        ax.set_ylim([y_min,y_max])
    
    if x_locator!=None:
        ax.xaxis.set_major_locator(MultipleLocator(x_locator))
    if y_locator!=None:
        ax.yaxis.set_major_locator(MultipleLocator(y_locator))


    # print(mae)
    MIN = min(y_test)
    MAX = max(y_test)
    x = np.arange(MIN, MAX, 0.01)
    ax.plot(x,x, lw = 1, zorder = 0, color = 'black')

    plt.tick_params(width=2.5,labelsize=fontsize)
    
    linewidth = 2.5
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    fig.tight_layout()
    plt.savefig(model_name, dpi = 600,bbox_inches='tight',)
    # plt.show()
    return mae,mse,rmse,r2

def plot_trainset_testset_predic(y_test_all,y_pred_all,y_test,y_predic,xlabel,ylabel,
                                     alpha=0.7,s=50,x_max=None,x_min=None,y_max=None,y_min=None,
                                     x_locator=None,y_locator=None,
                                    fontsize=20,
                                     ):
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)

    rms=RegressionMetrics(y_test,y_predic)
    mse = round(rms.mse,3)
    mae = round(rms.mae,3)
    r2 = round(rms.r2,3)
    rmse = round(rms.rmse,3)
    print('mae,mse,rmse,r2 : ',mae,mse,rmse,r2)
    print(' ')

    plt.scatter( 
        y_test_all, 
        y_pred_all, 
        label = 'Training set',
        color = 'orange',
        marker = 'o',
        edgecolors = 'k',
        s = s,
        alpha = alpha,
        )

    plt.scatter(
            y_test, 
            y_predic, 
            label = 'Test set',
            color = 'blueviolet',
            marker = 'o',
            s = s,
            edgecolors = 'k',
            alpha = alpha,
            )

    plt.legend(fontsize=fontsize)

    ax.set_xlabel(xlabel, fontsize = fontsize, fontweight = 'semibold')
    ax.set_ylabel(ylabel, fontsize = fontsize, fontweight = 'semibold')

    # ax.grid(linestyle='--')
    MIN = min(y_test_all)
    MAX = max(y_pred_all)
    x = np.arange(MIN, MAX, 0.01)
    ax.plot(x,x, lw = 1, zorder = 0, color = 'black')

    if x_max != None:
        ax.set_xlim([x_min,x_max])
    if y_max != None:
        ax.set_ylim([y_min,y_max])
    
    if x_locator!=None:
        ax.xaxis.set_major_locator(MultipleLocator(x_locator))
    if y_locator!=None:
        ax.yaxis.set_major_locator(MultipleLocator(y_locator))


    plt.tick_params(width=2.5,labelsize=fontsize)
    
    linewidth = 2.5
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)
    fig.tight_layout()
    plt.savefig('Validation', dpi = 600,bbox_inches='tight',)

    plt.clf()

    return mae,mse,rmse,r2

def plot_loss(train_score,test_score,n_estimators=1000):
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(1, 1, 1)
    plt.plot(
        np.arange(n_estimators) + 1,
        train_score,
        "b-",
        label="Train set loss",
        linewidth = 2.5,
    )
    plt.plot(
        np.arange(n_estimators) + 1, 
        test_score,
        "r-", 
        label="Test set loss",
        linewidth = 2.5
    )
    plt.legend(loc="upper right",prop = {'size':16})
    plt.xlabel("Boosting Iterations",fontsize = 20, fontweight = 'semibold')
    plt.ylabel("MSE loss",fontsize = 20, fontweight = 'semibold')
    
    linewidth = 2.5
    ax1.spines['top'].set_linewidth(linewidth)
    ax1.spines['right'].set_linewidth(linewidth)
    ax1.spines['left'].set_linewidth(linewidth)
    ax1.spines['bottom'].set_linewidth(linewidth)
    plt.tick_params(width=2.5,labelsize=20)

    fig.tight_layout()
    plt.savefig('GBRT_loss.png', dpi = 600)

def train_gbrt():
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(scaled_data, 
                                                        data_target, 
                                                        test_size = 0.1,
                                                        random_state=321,
                                                        )
    print('train set number : ',len(X_train))
    print('test set number : ',len(X_test))
    print('Begin training: ')

    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    print('Training finish. ')

    test_score = np.zeros((hyperparameter["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(model.staged_predict(X_test)):
        test_score[i] = mean_squared_error(y_test, y_pred)

    plot_loss(model.train_score_,test_score)


    mse = mean_squared_error(data_target, model.predict(scaled_data))
    print("The mean squared error (MSE) on DataSet: {:.4f}".format(mse))
    for i, y_pred_all in enumerate(model.staged_predict(scaled_data)):
        test_score[i] = mean_squared_error(data_target, y_pred_all)


    plot_trainset_testset_predic(
        data_target, 
        y_pred_all,
        y_test, 
        y_pred,
        xlabel='Actual formation energy',ylabel='Predictive formation energy',
        # text_x=-1, text_y=-3.5
        )

def cross_val_kfold(X,y,model,hyperparameter):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    mae_list,mse_list,rmse_list,r2_list = [],[],[],[]
    k = 0
    FI_list = []
    for train_index, test_index in kf.split(X):
        k = k+1
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        sklearn_model = 'GBRT_'+str(k)
        
        print('Begin training: ')
        model.fit(X_train, y_train)

        test_score = np.zeros((hyperparameter["n_estimators"],), dtype=np.float64)
        for i, y_pred in enumerate(model.staged_predict(X_test)):
            test_score[i] = mean_squared_error(y_test, y_pred)
        
        test_score_all = np.zeros((hyperparameter["n_estimators"],), dtype=np.float64)
        for j, y_pred_all in enumerate(model.staged_predict(X_train)):
            test_score_all[i] = mean_squared_error(y_train, y_pred_all)

        data_loss= pd.DataFrame()
        data_loss['train_score_'] = model.train_score_
        data_loss['test_score'] = test_score
        data_loss.to_csv(str(k)+'data_loss.csv',index=False)


        mae,mse,rmse,r2 = plot_ytest_ypred(y_test,y_pred,sklearn_model,
                            xlabel='Actual formation energy',ylabel='Predictive formation energy',text_x=-1, text_y=-3.5)

        y_pred_train = model.predict(X_train)
        
        # output prediction results
        # testset
        data_testset_pred = pd.DataFrame()
        data_testset_pred['test_target'] = y_test
        data_testset_pred['y_pred'] = y_pred
        # allset
        data_allset_pred = pd.DataFrame()
        data_allset_pred['train_target'] = y_train
        data_allset_pred['y_pred_train'] = y_pred_train

        data_testset_pred.to_csv('GBRT_'+str(k)+'data_testset_pred.csv',index=False)
        data_allset_pred.to_csv('GBRT_'+str(k)+'data_trainset_pred.csv',index=False)
        
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        r2_list.append(r2)

        FI_list.append(model.feature_importances_)

    FI = np.mean(FI_list,axis=0) 

    print('MAE',np.mean(mae_list),'\n',mae_list,'\n')
    print('MSE',np.mean(mse_list),'\n',mse_list,'\n')
    print('RMSE',np.mean(rmse_list),'\n',rmse_list,'\n')
    print('R2',np.mean(r2_list),'\n',r2_list,'\n')

    return mae_list,mse_list,rmse_list,r2_list,FI



if __name__ == "__main__":

    # 0. load best feature set
    data = pd.read_csv('../feature_selection_combination/best_feature_set.csv')
    data_target = data.pop('formation_energy') 
    features = data.columns.tolist()

    # 1. scale data
    maxmin_scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = maxmin_scaler.fit_transform(data)  


    # 2. define model
    sklearn_model = 'GBRT_Ef'
    model,hyperparameter = define_model()

    # 3. train model
    train_gbrt()

    # 4. 10-fold train and validation
    MAE_list,MSE_list,RMSE_list,R2_list,FI = cross_val_kfold(X=scaled_data,y=data_target,model=model,hyperparameter=hyperparameter)

    # save results
    data_cv = pd.DataFrame()
    data_cv['MAE'] = MAE_list
    data_cv['MSE'] = MSE_list
    data_cv['RMSE'] = RMSE_list
    data_cv['R2'] = R2_list
    data_cv.to_csv('cross_validation_results.csv')

    featre_list,importance_list = [],[]
    sorted_idx_ = np.argsort(-FI)
    feature_sort = dict(zip(FI,features))
    for key in sorted(feature_sort.keys(),reverse=True):
        importance_list.append(key)
        featre_list.append(feature_sort[key])

    data_FI = pd.DataFrame()
    data_FI['feature'] = featre_list
    data_FI['importance'] = importance_list
    data_FI.to_csv('feature_importance.csv')