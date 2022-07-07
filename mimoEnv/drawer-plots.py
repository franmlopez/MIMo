import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pacmap

df_proprio_3 = pd.read_csv('results/proprio_testmu01_mu1_sigma0_iter3.csv', index_col='Unnamed: 0')
df_proprio_4 = pd.read_csv('results/proprio_testmu01_mu1_sigma0_iter4.csv', index_col='Unnamed: 0')
df_proprio_5 = pd.read_csv('results/proprio_testmu01_mu1_sigma0_iter5.csv', index_col='Unnamed: 0')
df_proprio_mu1_tmu1 = df_proprio_3.append(df_proprio_4, ignore_index=True).append(df_proprio_5, ignore_index=True)

df_proprio_3 = pd.read_csv('results/proprio_testmu012_mu1_sigma0_iter3.csv', index_col='Unnamed: 0')
df_proprio_4 = pd.read_csv('results/proprio_testmu012_mu1_sigma0_iter4.csv', index_col='Unnamed: 0')
df_proprio_5 = pd.read_csv('results/proprio_testmu012_mu1_sigma0_iter5.csv', index_col='Unnamed: 0')
df_proprio_mu1_tmu12 = df_proprio_3.append(df_proprio_4, ignore_index=True).append(df_proprio_5, ignore_index=True)

df_proprio_3 = pd.read_csv('results/proprio_testmu01_mu12_sigma0_iter3.csv', index_col='Unnamed: 0')
df_proprio_4 = pd.read_csv('results/proprio_testmu01_mu12_sigma0_iter4.csv', index_col='Unnamed: 0')
df_proprio_5 = pd.read_csv('results/proprio_testmu01_mu12_sigma0_iter5.csv', index_col='Unnamed: 0')
df_proprio_mu12_tmu1 = df_proprio_3.append(df_proprio_4, ignore_index=True).append(df_proprio_5, ignore_index=True)

df_proprio_3 = pd.read_csv('results/proprio_testmu012_mu12_sigma0_iter3.csv', index_col='Unnamed: 0')
df_proprio_4 = pd.read_csv('results/proprio_testmu012_mu12_sigma0_iter4.csv', index_col='Unnamed: 0')
df_proprio_5 = pd.read_csv('results/proprio_testmu012_mu12_sigma0_iter5.csv', index_col='Unnamed: 0')
df_proprio_mu12_tmu12 = df_proprio_3.append(df_proprio_4, ignore_index=True).append(df_proprio_5, ignore_index=True)

df_proprio_mu1_tmu1 = df_proprio_mu1_tmu1.groupby('0').mean()
df_proprio_mu1_tmu1['forces'] = 'mu1_tmu1'
df_proprio_mu1_tmu12 = df_proprio_mu1_tmu12.groupby('0').mean()
df_proprio_mu1_tmu12['forces'] = 'mu1_tmu12'
df_proprio_mu12_tmu1 = df_proprio_mu12_tmu1.groupby('0').mean()
df_proprio_mu12_tmu1['forces'] = 'mu12_tmu1'
df_proprio_mu12_tmu12 = df_proprio_mu12_tmu12.groupby('0').mean()
df_proprio_mu12_tmu12['forces'] = 'mu12_tmu12'
df_proprio_all = df_proprio_mu1_tmu1.append(df_proprio_mu1_tmu12).append(df_proprio_mu12_tmu1).append(df_proprio_mu12_tmu12)

embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)

train_data = pd.DataFrame.to_numpy(df_proprio_all.loc[:, df_proprio_all.columns != 'forces'])
transformed = embedding.fit_transform(train_data, init='pca')

forces_to_colors = {
    'mu1_tmu1': 'tab:blue',
    'mu1_tmu12': 'tab:pink',
    'mu12_tmu1': 'tab:green',
    'mu12_tmu12': 'tab:orange',
    }
forces_labels = df_proprio_all['forces'].map(forces_to_colors)

plt.figure(figsize=(10,10))
scatter=plt.scatter(transformed[:,0], transformed[:,1], cmap="Spectral", c=forces_labels)

plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

legend_elements = [Line2D([0], [0], marker='o', color='w', label='mu=1, test_mu=1',
                          markerfacecolor='tab:blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='mu=1, test_mu=12',
                          markerfacecolor='tab:pink', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='mu=12, test_mu=1',
                          markerfacecolor='tab:green', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='mu=12, test_mu=12',
                          markerfacecolor='tab:orange', markersize=10)
                   ]

plt.legend(handles=legend_elements, loc='lower left', fontsize = 12)

plt.show()