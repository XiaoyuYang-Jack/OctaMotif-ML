import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


'''
RMSE and MaxAE:   0.368691  2.742077
37:[((MagpieDatameanElectronegativityformula*sqrt(MagpieDatameanNpValenceformula))*sqrt((local_difference_in_Electronegativity_allsites/MagpieDatameanNVal]
156:[(((MagpieDataavg_devNUnfilledformula/MagpieDatameanNUnfilledformula))^2*(MagpieDatameanNpValenceformula*(ewald_energy_per_atom*MagpieDatarangeNUnfille]
coefficients_001:    -0.8789549192E+00    0.3396252716E-02
Intercept_001:     0.1626686146E+00
RMSE,MaxAE_001:     0.3686909472E+00    0.2742077235E+01
'''



def plot_ytest_ypred(y_pred,y_test,model_name,s=70,alpha=0.7,color = 'royalblue',
                        xlabel='PredictiveValue',ylabel='Actual Value',
                        x_max=None,x_min=None,y_max=None,y_min=None,x_locator=None,y_locator=None,
                        fontsize=25,):

    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    plt.scatter(
        y_pred, 
        y_test, 
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



data = pd.read_csv('train.dat')
print(data)
data['sisso1'] = data['MagpieDatameanElectronegativityformula']*np.sqrt(data['MagpieDatameanNpValenceformula'])*np.sqrt(data['local_difference_in_Electronegativity_allsites']/data['MagpieDatameanNValenceformula'])
data['sisso2'] = (data['MagpieDataavg_devNUnfilledformula']/data['MagpieDatameanNUnfilledformula'])**2*(data['MagpieDatameanNpValenceformula']*data['ewald_energy_per_atom']*data['MagpieDatarangeNUnfilledformula'])
f_sisso_Ef = -0.8789549192*data['sisso1']+0.3396252716E-02*data['sisso2']+0.1626686146



print(np.corrcoef(data['formation_energy'], f_sisso_Ef))  # 0.9028

plot_ytest_ypred(f_sisso_Ef, data['formation_energy'], 
                 s=30,xlabel='SR formation energy (eV/atom)',ylabel='DFT formation energy (eV/atom)',
                 model_name='SISSO_results.png',
                 x_locator=1,y_locator=1,)