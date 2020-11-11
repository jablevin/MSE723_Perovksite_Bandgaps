import pandas as pd
import os
from pymatgen import MPRester
import NuclearTools.Tools as nt
from mendeleev import element
import numpy as np

os.chdir('C:/Users/Jacob/Documents/Classes/MSE723_Yingling/Project/')
m = MPRester("xRMmkiwmwMSaqTZQ")

df = pd.read_csv('starter_set.csv')
scale = pd.read_csv('chem_scales.csv')

elems = ['Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be',
       'Bh', 'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm',
       'Cn', 'Co', 'Cr', 'Cs', 'Cu', 'D', 'Db', 'Ds', 'Dy', 'Er', 'Es',
       'Eu', 'F', 'Fe', 'Fm', 'Fr', 'Ga', 'Gd', 'Ge', 'H', 'He', 'Hf',
       'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K', 'Kr', 'La', 'Li', 'Lr',
       'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na', 'Nb', 'Nd', 'Ne',
       'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd', 'Pm', 'Po',
       'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rg', 'Rh', 'Rn', 'Ru',
       'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'T', 'Ta',
       'Tb', 'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'Uuh', 'Uuo', 'Uup',
       'Uuq', 'Uus', 'Uut', 'V', 'W', 'Xe', 'Y', 'Yb', 'Zn', 'Zr']

data = {}
holding = {'BG':[], 'E_a':[], 'E_b':[], 'E_c':[],
           'I_a':[], 'I_b':[], 'I_c':[], 'Nv_a':[], 'Nv_b':[], 'Nv_c':[],
           'Ea_a':[], 'Ea_b':[], 'Ea_c': [], 'S_a':[], 'S_b':[], 'S_c':[],
           'Atomic #a':[], 'Atomic #b':[]}


c = element('O')
c2 = c.electron_affinity
c3 = c.electronegativity('pauling')
c4 = c.ionenergies[1]
c5 = c.nvalence()
c6 = scale['scale'][scale.index[scale['element']=='O'][0]]
for item in list(df['Compound']):
    try:
        b = m.get_data(item)[0]
    except:
        b = []
    if b:
        holding['BG'].append(b['band_gap'])
    else:
        holding['BG'].append(np.NaN)

    elem1 = nt.indv_elements(item)[0][0]
    elem2 = nt.indv_elements(item)[0][1]

    a = element(elem1)
    a2 = a.electron_affinity
    a3 = a.electronegativity('pauling')
    a4 = a.ionenergies[1]
    a5 = a.nvalence()
    a6 = scale['scale'][scale.index[scale['element']==elem1][0]]

    b = element(elem2)
    b2 = b.electron_affinity
    b3 = b.electronegativity('pauling')
    b4 = b.ionenergies[1]
    b5 = b.nvalence()
    b6 = scale['scale'][scale.index[scale['element']==elem2][0]]

    holding['Ea_a'].append(a2)
    holding['E_a'].append(a3)
    holding['I_a'].append(a4)
    holding['Nv_a'].append(a5)
    holding['S_a'].append(a6)
    holding['Atomic #a'].append(a.atomic_number)

    holding['Ea_b'].append(b2)
    holding['E_b'].append(b3)
    holding['I_b'].append(b4)
    holding['Nv_b'].append(b5)
    holding['S_b'].append(b6)
    holding['Atomic #b'].append(b.atomic_number)

    holding['Ea_c'].append(c2)
    holding['E_c'].append(c3)
    holding['I_c'].append(c4)
    holding['Nv_c'].append(c5)
    holding['S_c'].append(c6)


for i in list(holding.keys()):
    df[i] = holding[i]

df.dropna(axis=0, subset=['BG'], inplace=True)
df.fillna(0, axis=1, inplace=True)

holding2 = {'BG ao':[], 'BG bo':[]}
for item in list(df['Compound']):
    elem1 = nt.indv_elements(item)[0][0]
    elem2 = nt.indv_elements(item)[0][1]
    a = element(elem1)
    b = element(elem2)

    try:
        holding2['BG ao'].append(m.get_data(a.symbol+'O')[0]['band_gap'])
    except:
        try:
            holding2['BG ao'].append(m.get_data(a.symbol+'O3')[0]['band_gap'])
        except:
            holding2['BG ao'].append(0)

    try:
        holding2['BG bo'].append(m.get_data(b.symbol+'O')[0]['band_gap'])
    except:
        try:
            holding2['BG bo'].append(m.get_data(b.symbol+'O3')[0]['band_gap'])
        except:
            holding2['BG bo'].append(0)


for i in list(holding2.keys()):
    df[i] = holding2[i]

df.to_csv('working_on_it.csv', index=False)

k = list(df).index('E_a')
coeff1 = [1, 2, 3]
coeff2 = [1, 2, 3]
methods = ['+', '-', '*', '/']
headers = list(df)
df2 = df.copy()
# del df
for c1 in coeff1:
    for c2 in coeff2:
        for method in methods:
            for i in range(14):
                counter = 1
                for j in range(i+1, 14):
                    if counter % 3:
                        if method == '+':
                            df2[str(c1)+headers[k+i]+method+str(c2)+headers[k+j]] = c1*df[headers[k+i]] + c2*df[headers[k+j]]
                        elif method == '-':
                            df2[str(c1)+headers[k+i]+method+str(c2)+headers[k+j]] = c1*df[headers[k+i]] - c2*df[headers[k+j]]
                        elif method == '*':
                            df2[str(c1)+headers[k+i]+method+str(c2)+headers[k+j]] = c1*df[headers[k+i]] * c2*df[headers[k+j]]
                        elif method == '/':
                            df2[str(c1)+headers[k+i]+method+str(c2)+headers[k+j]] = c1*df[headers[k+i]] / c2*df[headers[k+j]]
                    counter += 1

            for i in range(14):
                counter = 1
                for j in range(i+1, 14):
                    if counter % 3:
                        if method == '+':
                            df2[str(c1)+headers[k+i]+'^2'+method+str(c2)+headers[k+j]] = c1*df[headers[k+i]]**2 + c2*df[headers[k+j]]
                        elif method == '-':
                            df2[str(c1)+headers[k+i]+'^2'+method+str(c2)+headers[k+j]] = c1*df[headers[k+i]]**2 - c2*df[headers[k+j]]
                        elif method == '*':
                            df2[str(c1)+headers[k+i]+'^2'+method+str(c2)+headers[k+j]] = c1*df[headers[k+i]]**2 * c2*df[headers[k+j]]
                        elif method == '/':
                            df2[str(c1)+headers[k+i]+'^2'+method+str(c2)+headers[k+j]] = c1*df[headers[k+i]]**2 / c2*df[headers[k+j]]
                    counter += 1

            for i in range(14):
                counter = 1
                for j in range(i+1, 14):
                    if counter % 3:
                        if method == '+':
                            df2[str(c1)+headers[k+i]+method+str(c2)+headers[k+j]+'^2'] = c1*df[headers[k+i]] + c2*df[headers[k+j]]**2
                        elif method == '-':
                            df2[str(c1)+headers[k+i]+method+str(c2)+headers[k+j]+'^2'] = c1*df[headers[k+i]] - c2*df[headers[k+j]]**2
                        elif method == '*':
                            df2[str(c1)+headers[k+i]+method+str(c2)+headers[k+j]+'^2'] = c1*df[headers[k+i]] * c2*df[headers[k+j]]**2
                        elif method == '/':
                            df2[str(c1)+headers[k+i]+method+str(c2)+headers[k+j]+'^2'] = c1*df[headers[k+i]] / c2*df[headers[k+j]]**2
                    counter += 1

            for i in range(14):
                counter = 1
                for j in range(i+1, 14, 3):
                    if counter % 3:
                        if method == '+':
                            df2[str(c1)+headers[k+i]+'^2'+method+str(c2)+headers[k+j]+'^2'] = c1*df[headers[k+i]]**2 + c2*df[headers[k+j]]**2
                        elif method == '-':
                            df2[str(c1)+headers[k+i]+'^2'+method+str(c2)+headers[k+j]+'^2'] = c1*df[headers[k+i]]**2 - c2*df[headers[k+j]]**2
                        elif method == '*':
                            df2[str(c1)+headers[k+i]+'^2'+method+str(c2)+headers[k+j]+'^2'] = c1*df[headers[k+i]]**2 * c2*df[headers[k+j]]**2
                        elif method == '/':
                            df2[str(c1)+headers[k+i]+'^2'+method+str(c2)+headers[k+j]+'^2'] = c1*df[headers[k+i]]**2 / c2*df[headers[k+j]]**2
                    counter += 1


df2.drop('E_c', axis=1, inplace=True)
df2.drop('Ea_c', axis=1, inplace=True)
df2.drop('I_c', axis=1, inplace=True)
df2.drop('Nv_c', axis=1, inplace=True)
df2.drop('S_c', axis=1, inplace=True)
df2.drop('Compound', axis=1, inplace=True)
df2.to_csv('almost_there.csv', index=False)

# df2 = pd.read_csv('almost_there.csv')

################################################################################
# Perform Correlation work
################################################################################

headers = list(df2)
nump = df2.to_numpy().transpose().copy()
len(headers)

corr_matrix = pd.DataFrame(data=np.corrcoef(nump), columns=headers)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
8052 - len(to_drop)
df2.drop(to_drop, axis=1, inplace=True)


heads = list(df2)
p = heads.index('BG')
new_dataset = df2[[heads[1:p]+heads[p+1:]+[heads[p]]][0]].copy()
new_dataset.to_csv('Prior_Classify.csv', index=False)

################################################################################
# Perform Variance work
################################################################################

# heads = list(new_dataset)
# hist = np.array(new_dataset.var()) / np.array(new_dataset.mean())
# indices = np.array([int(i) for i in range(len(hist)) if hist[i] >= -1 and hist[i] <= 1])
# heads2 = []
# for i in indices:
#     heads2.append(heads[i])
#
# if 'BG' not in heads2:
#     heads2.append('BG')
#
#
# new_dataset = new_dataset[heads2].copy()
# new_dataset.to_csv('Prior_Classify4.csv', index=False)
#
#
# import matplotlib.pyplot as plt
# plot = True
# if plot:
#     plt.figure(figsize=(12,8))
#     plt.hist(hist, bins=5000)
#     plt.xlim(-50, 50)

################################################################################
# Make Classification Dataset
################################################################################

hold = []
for i in range(len(new_dataset)):
    hold.append(1 if new_dataset['BG'][i] > 0 else 0)

new_dataset['BG or not'] = hold
new_dataset.drop('BG', axis=1, inplace=True)

new_dataset.to_csv('Classify_dataset.csv', index=False)
