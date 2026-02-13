import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

def plot_fold_line(y,xnumber,file_name,figsizex=8,figsizey=8):

    plt.figure(figsize=(figsizex,figsizey))
    ax = plt.subplot(111)
    plt.xlabel("Number of feature removed", fontsize = 20, fontweight = 'semibold')
    plt.ylabel("Cross validation MSE", fontsize = 20, fontweight = 'semibold')
    plt.plot(
        range(0, xnumber),
        y,
        marker = 'o', 
        markersize = 7, 
        c = 'mediumslateblue',
        mec = 'w',  
        linewidth = 2,
    )
    xmajorLocator = MultipleLocator(4) 
    ax.xaxis.set_major_locator(xmajorLocator)
    ymajorLocator = MultipleLocator(0.025) 
    ax.yaxis.set_major_locator(ymajorLocator)
    
    plt.tick_params(width=2.5,labelsize=20)

    ax = plt.gca()
    linewidth = 2.5
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)

    plt.tight_layout()
    plt.savefig(file_name, dpi = 600)


def plot_four_bar(x,height1,height2,height3,height4,filename,color1,color2,color3,color4,xlabel,ylabel='Counts',linewidth = 2.5,):
        
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(111)

    x = range(len(x))

    ax1.bar(
        [i - 0.3 for i in x],
        height=height1,
        bottom=0,
        width=0.2,
        label='MAE',
        color = color1,
    )
    ax1.bar(
        [i - 0.1 for i in x],
        height=height2,
        bottom=0,
        width=0.2,
        label='MSE',
        color = color2,
    )

    ax1.bar(
        [i + 0.1 for i in x],
        height=height3,
        bottom=0,
        width=0.2,
        label='RMSE',
        color = color3,
    )

    plt.tick_params(width=2.5,labelsize=20)
    plt.xticks(fontsize=16,)
    plt.xlabel(xlabel,fontsize = 20,fontweight = 'semibold')
    plt.ylabel(ylabel,fontsize = 20,fontweight = 'semibold')
    ax1.legend(prop = {'size':16},framealpha=0, loc="upper left",)

    ax1.spines['top'].set_linewidth(linewidth)
    ax1.spines['right'].set_linewidth(linewidth)
    ax1.spines['left'].set_linewidth(linewidth)
    ax1.spines['bottom'].set_linewidth(linewidth)
    

    ax2 = ax1.twinx()
    ax2.bar(
        [i + 0.3 for i in x],
        height=height4,
        bottom=0,
        width=0.2,
        label='R2',
        color = color4,
    )
    ax2.set_xlabel("R2",fontsize = 20,fontweight = 'semibold')

    ax1.xaxis.set_major_locator(MultipleLocator(1))
    plt.tick_params(width=2.5,labelsize=20)
    plt.xticks(fontsize=16,)

    plt.xlabel(xlabel,fontsize = 20,fontweight = 'semibold')
    plt.ylabel('R2',fontsize = 20,fontweight = 'semibold')
    ax2.legend(prop = {'size':16},framealpha=0, loc="upper right",)
    ax2.set_ylim([0.5,1.1])
    plt.tight_layout()
    plt.savefig(filename,dpi = 600)


def plot_trainset_testset_predic(y_test_all,y_pred_all,y_test,y_predic,xlabel,ylabel,
                                     alpha=0.7,s=50,x_max=None,x_min=None,y_max=None,y_min=None,
                                     x_locator=None,y_locator=None,
                                    fontsize=20,
                                     ):
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)

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



def plot_dist(df, feature, target, pic_name='dist_plot.png',ylabel='Formation Energy',
                fcols=6):
    import math

    frows = math.ceil(len(feature)/fcols)
    print(fcols, frows)
    plt.figure(figsize=(8*fcols, 8*frows))

    i = 0
    for col in feature: 
        
        i += 1
        ax1 = plt.subplot(frows, fcols, i)
        plt.scatter(df[col], target,alpha = 0.5,)

        plt.xlabel(col,fontsize = 20, fontweight = 'semibold')
        plt.ylabel(ylabel,fontsize = 20, fontweight = 'semibold')
        plt.tick_params(width=2.5,labelsize=20)


        linewidth = 2.5
        ax1.spines['top'].set_linewidth(linewidth)
        ax1.spines['right'].set_linewidth(linewidth)
        ax1.spines['left'].set_linewidth(linewidth)
        ax1.spines['bottom'].set_linewidth(linewidth)

    plt.tight_layout()
    plt.savefig(pic_name,dpi = 300)



def plot_correlation_coefficient(df,features,
                                file_name,
                                file_format = 'jpg',
                                title_fontsize = 15,
                                ticks_fontsize = 6,
                                fontweight = 'semibold',
                                linewidths = 0.2,
                                colors = 'coolwarm',
                                show_num = True,
                                num_size = 5,
                                num_weight = 'normal',
                                num_color =  'blue'
                                ):
    corr_df = df[features].corr()

    # mask = np.zeros_like(corr_df, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True

    fig,ax = plt.subplots()
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)
    ax = sns.heatmap(corr_df,  
                    #  mask=mask,
                    xticklabels=True, 
                    yticklabels=True, 
                    cmap = sns.color_palette(colors, 10),
                    cbar = True,
                    cbar_kws={"ticks":[-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.2,0.4,0.6,0.8,1.0]},
                    vmin =-1,
                    vmax = 1 ,
                    linewidths = linewidths,
                    )
    plt.xticks(fontsize = ticks_fontsize , rotation = 90)
    plt.yticks(fontsize = ticks_fontsize , rotation = 0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(file_name, dpi = 600)


def plot_feature_importance(features,feature_importance,sorted_idx_,model_name,
                                alpha=1,color='GnBu',fontsize=20,
                                show_xlabel=True,
                                x_max=None,x_min=None,y_max=None,y_min=None,x_locator=None,y_locator=None,
                                ):
    from matplotlib import cm
    plt.figure(figsize=(9,8))
    ax1 = plt.subplot(1, 1, 1)

    id = list(range(len(feature_importance)))
    id.sort(reverse=True)
    id = np.array(id)

    norm_values = (id - id[-1])/id[0]

    map_vir = cm.get_cmap(name=color)
    colors = map_vir(norm_values)
    # colors = map_vir(id)

    plt.bar(np.arange(len(features)),
            feature_importance[sorted_idx_],
            tick_label=np.array(features)[sorted_idx_],
            alpha=alpha,
            color = colors)

    plt.tick_params(width=2.5,labelsize=fontsize,labelbottom=False) 

    if show_xlabel==True:
        plt.xlabel('Features', fontsize = fontsize, fontweight = 'semibold')

    plt.ylabel('Importance ', fontsize = fontsize, fontweight = 'semibold')
    
    if x_max != None:
        ax1.set_xlim([x_min,x_max])
    if y_max != None:
        ax1.set_ylim([y_min,y_max])
    
    if x_locator!=None:
        ax1.xaxis.set_major_locator(MultipleLocator(x_locator))
    if y_locator!=None:
        ax1.yaxis.set_major_locator(MultipleLocator(y_locator))

    # plt.xticks(rotation=315)
    linewidth = 2.5
    ax1.spines['top'].set_linewidth(linewidth)
    ax1.spines['right'].set_linewidth(linewidth)
    ax1.spines['left'].set_linewidth(linewidth)
    ax1.spines['bottom'].set_linewidth(linewidth)
    
    plt.tight_layout()
    # plt.subplots_adjust(bottom=1)
    plt.savefig(model_name, dpi = 600)



if __name__ == '__main__':

    # 0. load results
    data = pd.read_csv('../feature_selection_combination/best_feature_set.csv')
    data_target = data['formation_energy']

    data_features = pd.read_csv('../cross_validation/cross_validation_results/feature_importance.csv')
    features = data_features['feature']
    feature_importance = data_features['importance']
    sorted_idx_ = np.argsort(-feature_importance)

    data_train = pd.read_csv('../cross_validation/cross_validation_results/GBRT_1data_trainset_pred.csv')
    data_test = pd.read_csv('../cross_validation/cross_validation_results/GBRT_1data_testset_pred.csv')
    test_target = data_test['test_target']
    y_pred = data_test['y_pred']

    featre_select_scores = pd.read_csv('../feature_selection_combination/feature_select_scores.csv')


    # 1. plot feature selection results
    plot_fold_line(y=featre_select_scores['MSE'],xnumber=37,file_name = 'REF_feature_select')


    # 2. plot cross validation results
    x = []
    for i in range(10):
        x.append(i+1)
    cross_validation = pd.read_csv('../cross_validation/cross_validation_results/cross_validation_results.csv'
                                    )[['MAE','MSE','RMSE','R2']]
    plot_four_bar(x,cross_validation['MAE'],cross_validation['MSE'],cross_validation['RMSE'],cross_validation['R2'],
                    filename='cross_validation',color1='royalblue',color2='orange',color3='blueviolet',color4='limegreen',
                    xlabel='Cross validation folds',ylabel='Metrics',linewidth = 2.5,)


    # 3. lot real VS. predicted
    plot_trainset_testset_predic(
        y_test_all=data_train['train_target'],
        y_pred_all=data_train['y_pred_train'],
        y_test = test_target,y_predic = y_pred,
        xlabel='DFT formation energy (eV/atom)',
        ylabel='GBRT formation energy (eV/atom)',
        )


    # 4. plot feature distribution
    plot_dist(data, features[sorted_idx_], data_target, pic_name='Feature_distribution.png',ylabel='DFT formation energy (eV/atom)')


    # 5. plot feature correlation coefficient heatmap
    plot_correlation_coefficient(
        data,
        features,
        'Feature_correlation_coefficient',
        linewidths = 0.5,
        num_size = 4,
    )


    # 6. plot feature importance
    plot_feature_importance(features[:10],
                            feature_importance[:10],
                            sorted_idx_[:10],
                            model_name = 'feature_importance_top10',
                            alpha=0.75,
                            # color = 'cool'
                            )
