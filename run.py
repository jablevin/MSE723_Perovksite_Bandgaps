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

nlist = [5, 10, 15, 20, 30, 35, 60, 100]
nlist = [35]
obj = Regressor('no_zero_BG.csv', nlist, estimators=500, nsteps=10)
obj = Regressor('Prior_Classify.csv', nlist, estimators=500, nsteps=10)

obj.acc
obj.rfe_features

################################################################################
# Regression and Classification
################################################################################

obj = Classifier('Classify_dataset.csv', nlist, estimators=500, nsteps=10, loss_='deviance')
obj.acc

data = pd.read_csv('Prior_Classify.csv')
data.insert(loc=len(list(data))-1, column='metallic', value=obj.selector.predict(obj.x))
data.to_csv('Post_classify.csv', index=False)
obj = Regressor('Post_classify.csv', nlist, estimators=100, nsteps=10)
obj.acc
obj.rfe_features

################################################################################
# Plotting
################################################################################

matplotlib.rcParams.update({'font.size': 14})
plot = True
if plot:
    plt.figure(figsize=(12,8))
    plt.plot(nlist, obj.acc['f_regression'], label='F-test', color='magenta')
    plt.plot(nlist, obj.acc['mutual'], label='Mutual Info', color='cyan')
    plt.plot(nlist, obj.acc['rfe'], label='RFE', color='darkorange')
    plt.xlabel('Number of Features')
    plt.ylabel('$R^2$')
    # plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('plots/R^2 for PC4.png')
    # plt.savefig('plots/Feature_Selection_Accuracy_for_various_number_of_features.png')



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
    # plt.savefig('plots/est_BG_vs_actual.png')
    plt.savefig('plots/est_BG_vs_actual_no_0_BG.png')


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


def singlecorrplot(features, title_, save=False):
    ax = [0,1,2]
    fig, (ax[0], ax[1],  ax[2]) = plt.subplots(3, 1, figsize=(10,25))

    for i in range(len(ax)):
        ticks = ['$'+i+'$' for i in features[i]]
        corr = obj.x_train[features[i]].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        ax[i].set_title(title_[i])
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    xticklabels=ticks, yticklabels=ticks, ax=ax[i])
    plt.tight_layout()
    if save:
        plt.savefig('plots/all_corr.png', dpi=200)


s = True
singlecorrplot([obj.Ftest_features, obj.mutual_features, obj.rfe_features],
               ['F-test', 'Mutual', 'RFE'], save=s)


s = True
corrplot(obj.Ftest_features, 'F-test', save=s)
corrplot(obj.mutual_features, 'Mutual', save=s)
corrplot(obj.rfe_features, 'RFE', save=s)

################################################################################
# Periodic Table Plot
################################################################################

from bokeh.io import output_file, show, export_png
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap
from collections import Counter

periods = ["I", "II", "III", "IV", "V", "VI", "VII"]
groups = [str(x) for x in range(1, 19)]

df = elements.copy()
df["atomic mass"] = df["atomic mass"].astype(str)
df["group"] = df["group"].astype(str)
df["period"] = [periods[x-1] for x in df.period]
df = df[df.group != "-"]
df = df[df.symbol != "Lr"]
df = df[df.symbol != "Lu"]

data = pd.read_csv('Prior_Classify4.csv')
a = np.unique(data['Atomic #a'])
b = np.unique(data['Atomic #b'])
both = np.intersect1d(a, b)
ca = Counter(a)
cb = Counter(b)
r_a = np.array(sorted((ca - cb).elements()))
r_b = np.array(sorted((cb - ca).elements()))

site = []
for i in df['atomic number']:
    if i in r_a:
        site.append('A site')
    elif i in r_b:
        site.append('B site')
    elif i in both:
        site.append('Either site')
    else:
        site.append('None')

df['Site'] = site
cmap = {"A site"     : "#d93b43",
        "B site"     : "#599d7A",
        'Either site': '#CCCC00',
        'None'       : '#D3D3D3'}

plot = True
if plot:
    p = figure(title="Periodic Table", plot_width=1000, plot_height=450,
               x_range=groups, y_range=list(reversed(periods)),
               tools="hover", toolbar_location=None)
    r = p.rect("group", "period", 0.95, 0.95, source=df, fill_alpha=0.6, legend_field="Site",
              color=factor_cmap('Site', palette=list(cmap.values()), factors=list(cmap.keys())))
    text_props = {"source": df, "text_align": "left", "text_baseline": "middle"}
    x = dodge("group", -0.4, range=p.x_range)
    p.text(x=x, y="period", text="symbol", text_font_style="bold", **text_props)
    p.text(x=x, y=dodge("period", 0.3, range=p.y_range), text="atomic number",
           text_font_size="11px", **text_props)
    p.text(x=x, y=dodge("period", -0.35, range=p.y_range), text="name",
           text_font_size="7px", **text_props)
    p.text(x=x, y=dodge("period", -0.2, range=p.y_range), text="atomic mass",
           text_font_size="7px", **text_props)
    p.text(x=["3", "3"], y=["VI", "VII"], text=["LA", "AC"], text_align="center", text_baseline="middle")
    p.outline_line_color = None
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.legend.orientation = "horizontal"
    p.legend.location ="top_center"
    export_png(p, filename="plots/periodic_table.png")
