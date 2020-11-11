import os
os.chdir('C:/Users/Jacob/Documents/Classes/MSE723_Yingling/Project/')
from Regression_Class import Regressor
from Classify_Class import Classifier
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import pandas as pd


################################################################################
# Regression Only
################################################################################

# nlist = [5, 10, 15, 20, 30, 35, 60, 100]
nlist = [30]
obj = Regressor('Prior_Classify.csv', nlist, estimators=500, nsteps=10)#, ftest=False, mutual=False)

obj.acc
obj.rfe_features

################################################################################
# Regression and Classification
################################################################################

obj = Classifier('Classify_dataset.csv', nlist, estimators=500, nsteps=10, loss_='deviance')#, ftest=False, mutual=False)
obj.acc

data = pd.read_csv('Prior_Classify.csv')
data.insert(loc=len(list(data))-1, column='metallic', value=obj.selector.predict(obj.x))
data.to_csv('Post_classify.csv', index=False)
obj = Regressor('Post_classify.csv', nlist, estimators=100, nsteps=10)#, ftest=False, mutual=False)
obj.acc
obj.rfe_features

################################################################################
# Plotting
################################################################################

matplotlib.rcParams.update({'font.size': 14})
plot = True
if plot:
    plt.figure(figsize=(12,8))
    plt.plot(nlist, obj.acc['f_regression'], label='f_regression')
    plt.plot(nlist, obj.acc['mutual'], label='mutual')
    plt.plot(nlist, obj.acc['rfe'], label='rfe')
    plt.xlabel('Number of Features')
    # plt.ylabel('$R^2$')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('plots/Feature_Selection_R2_for_various_number_of_features.png')
    # plt.savefig('plots/Feature_Selection_Accuracy_for_various_number_of_features.png')


res = np.array(obj.y_test - obj.y_pred_r)
max_r = np.max(abs(res))
min_r = np.min(abs(res))
cmap = matplotlib.cm.get_cmap('viridis')
if plot:
    plt.figure(figsize=(12,8))
    plt.scatter(obj.y_test, obj.y_pred_f, label='F-test: $R^2= $'+str(obj.acc['f_regression'][-1])[:4], color='magenta', alpha=0.8, s=50)
    plt.scatter(obj.y_test, obj.y_pred_m, label='Mutual: $R^2= $'+str(obj.acc['mutual'][-1])[:4], color='cyan', alpha=0.8, s=50)
    plt.scatter(obj.y_test, obj.y_pred_r, label='RFE: $R^2= $'+str(obj.acc['rfe'][-1])[:4], color='darkorange', alpha=0.8, s=50)
    plt.plot([0,6], [0,6], color='black')
    plt.ylabel('GB Bandgap [ev]')
    plt.xlabel('DFT Bandgap [eV]')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('plots/R^2_accuracy_for_various_methods.png')


def corrplot(features, title_, save=False):
    ticks = ['$'+i+'$' for i in features]
    corr = obj.x_train[features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    plt.title(title_)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=ticks, yticklabels=ticks)
    plt.tight_layout()
    if save:
        plt.savefig('plots/corr-'+title_+'.png')

s = True
corrplot(obj.Ftest_features, 'F-test', save=s)
corrplot(obj.mutual_features, 'Mutual', save=s)
corrplot(obj.rfe_features, 'RFE', save=s)
