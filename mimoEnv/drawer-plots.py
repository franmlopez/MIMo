import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pacmap

"""
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

"""

df_proprio_3 = pd.read_csv('results/proprio_testmu00_mu6_sigma6_iter3.csv', index_col='Unnamed: 0')
df_proprio_4 = pd.read_csv('results/proprio_testmu00_mu6_sigma6_iter4.csv', index_col='Unnamed: 0')
df_proprio_5 = pd.read_csv('results/proprio_testmu00_mu6_sigma6_iter5.csv', index_col='Unnamed: 0')
df_proprio_6 = pd.read_csv('results/proprio_testmu00_mu6_sigma6_iter6.csv', index_col='Unnamed: 0')
df_proprio_7 = pd.read_csv('results/proprio_testmu00_mu6_sigma6_iter7.csv', index_col='Unnamed: 0')
df_proprio_8 = pd.read_csv('results/proprio_testmu00_mu6_sigma6_iter8.csv', index_col='Unnamed: 0')
df_proprio_9 = pd.read_csv('results/proprio_testmu00_mu6_sigma6_iter9.csv', index_col='Unnamed: 0')
df_proprio_tmu0 = df_proprio_3.append(df_proprio_4, ignore_index=True).append(df_proprio_5, ignore_index=True).append(
                    df_proprio_6, ignore_index=True).append(df_proprio_7, ignore_index=True).append(
                    df_proprio_8, ignore_index=True).append(df_proprio_9, ignore_index=True)

df_proprio_3 = pd.read_csv('results/proprio_testmu03_mu6_sigma6_iter3.csv', index_col='Unnamed: 0')
df_proprio_4 = pd.read_csv('results/proprio_testmu03_mu6_sigma6_iter4.csv', index_col='Unnamed: 0')
df_proprio_5 = pd.read_csv('results/proprio_testmu03_mu6_sigma6_iter5.csv', index_col='Unnamed: 0')
df_proprio_6 = pd.read_csv('results/proprio_testmu03_mu6_sigma6_iter6.csv', index_col='Unnamed: 0')
df_proprio_7 = pd.read_csv('results/proprio_testmu03_mu6_sigma6_iter7.csv', index_col='Unnamed: 0')
df_proprio_8 = pd.read_csv('results/proprio_testmu03_mu6_sigma6_iter8.csv', index_col='Unnamed: 0')
df_proprio_9 = pd.read_csv('results/proprio_testmu03_mu6_sigma6_iter9.csv', index_col='Unnamed: 0')
df_proprio_tmu3 = df_proprio_3.append(df_proprio_4, ignore_index=True).append(df_proprio_5, ignore_index=True).append(
                    df_proprio_6, ignore_index=True).append(df_proprio_7, ignore_index=True).append(
                    df_proprio_8, ignore_index=True).append(df_proprio_9, ignore_index=True)

df_proprio_3 = pd.read_csv('results/proprio_testmu06_mu6_sigma6_iter3.csv', index_col='Unnamed: 0')
df_proprio_4 = pd.read_csv('results/proprio_testmu06_mu6_sigma6_iter4.csv', index_col='Unnamed: 0')
df_proprio_5 = pd.read_csv('results/proprio_testmu06_mu6_sigma6_iter5.csv', index_col='Unnamed: 0')
df_proprio_6 = pd.read_csv('results/proprio_testmu06_mu6_sigma6_iter6.csv', index_col='Unnamed: 0')
df_proprio_7 = pd.read_csv('results/proprio_testmu06_mu6_sigma6_iter7.csv', index_col='Unnamed: 0')
df_proprio_8 = pd.read_csv('results/proprio_testmu06_mu6_sigma6_iter8.csv', index_col='Unnamed: 0')
df_proprio_9 = pd.read_csv('results/proprio_testmu06_mu6_sigma6_iter9.csv', index_col='Unnamed: 0')
df_proprio_tmu6 = df_proprio_3.append(df_proprio_4, ignore_index=True).append(df_proprio_5, ignore_index=True).append(
                    df_proprio_6, ignore_index=True).append(df_proprio_7, ignore_index=True).append(
                    df_proprio_8, ignore_index=True).append(df_proprio_9, ignore_index=True)

df_proprio_3 = pd.read_csv('results/proprio_testmu09_mu6_sigma6_iter3.csv', index_col='Unnamed: 0')
df_proprio_4 = pd.read_csv('results/proprio_testmu09_mu6_sigma6_iter4.csv', index_col='Unnamed: 0')
df_proprio_5 = pd.read_csv('results/proprio_testmu09_mu6_sigma6_iter5.csv', index_col='Unnamed: 0')
df_proprio_6 = pd.read_csv('results/proprio_testmu09_mu6_sigma6_iter6.csv', index_col='Unnamed: 0')
df_proprio_7 = pd.read_csv('results/proprio_testmu09_mu6_sigma6_iter7.csv', index_col='Unnamed: 0')
df_proprio_8 = pd.read_csv('results/proprio_testmu09_mu6_sigma6_iter8.csv', index_col='Unnamed: 0')
df_proprio_9 = pd.read_csv('results/proprio_testmu09_mu6_sigma6_iter9.csv', index_col='Unnamed: 0')
df_proprio_tmu9 = df_proprio_3.append(df_proprio_4, ignore_index=True).append(df_proprio_5, ignore_index=True).append(
                    df_proprio_6, ignore_index=True).append(df_proprio_7, ignore_index=True).append(
                    df_proprio_8, ignore_index=True).append(df_proprio_9, ignore_index=True)

df_proprio_3 = pd.read_csv('results/proprio_testmu012_mu6_sigma6_iter3.csv', index_col='Unnamed: 0')
df_proprio_4 = pd.read_csv('results/proprio_testmu012_mu6_sigma6_iter4.csv', index_col='Unnamed: 0')
df_proprio_5 = pd.read_csv('results/proprio_testmu012_mu6_sigma6_iter5.csv', index_col='Unnamed: 0')
df_proprio_6 = pd.read_csv('results/proprio_testmu012_mu6_sigma6_iter6.csv', index_col='Unnamed: 0')
df_proprio_7 = pd.read_csv('results/proprio_testmu012_mu6_sigma6_iter7.csv', index_col='Unnamed: 0')
df_proprio_8 = pd.read_csv('results/proprio_testmu012_mu6_sigma6_iter8.csv', index_col='Unnamed: 0')
df_proprio_9 = pd.read_csv('results/proprio_testmu012_mu6_sigma6_iter9.csv', index_col='Unnamed: 0')
df_proprio_tmu12 = df_proprio_3.append(df_proprio_4, ignore_index=True).append(df_proprio_5, ignore_index=True).append(
                    df_proprio_6, ignore_index=True).append(df_proprio_7, ignore_index=True).append(
                    df_proprio_8, ignore_index=True).append(df_proprio_9, ignore_index=True)

df_proprio_tmu0 = df_proprio_tmu0.groupby('0').mean()
df_proprio_tmu0['forces'] = 'test_mu0'
df_proprio_tmu3 = df_proprio_tmu3.groupby('0').mean()
df_proprio_tmu3['forces'] = 'test_mu3'
df_proprio_tmu6 = df_proprio_tmu6.groupby('0').mean()
df_proprio_tmu6['forces'] = 'test_mu6'
df_proprio_tmu9 = df_proprio_tmu9.groupby('0').mean()
df_proprio_tmu9['forces'] = 'test_mu9'
df_proprio_tmu12 = df_proprio_tmu12.groupby('0').mean()
df_proprio_tmu12['forces'] = 'test_mu12'
df_proprio_all = df_proprio_tmu0.append(df_proprio_tmu3).append(df_proprio_tmu6).append(df_proprio_tmu9).append(df_proprio_tmu12)

embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=200)

train_data = pd.DataFrame.to_numpy(df_proprio_all.drop(['1', 'forces'], axis=1))
transformed = embedding.fit_transform(train_data, init='pca')

forces_to_colors = {
    'test_mu0': 'tab:blue',
    'test_mu3': 'tab:pink',
    'test_mu6': 'tab:green',
    'test_mu9': 'tab:orange',
    'test_mu12': 'tab:olive',
    }
forces_labels = df_proprio_all['forces'].map(forces_to_colors)

plt.figure(figsize=(10,10))
scatter=plt.scatter(transformed[:,0], transformed[:,1], cmap="Spectral", c=forces_labels)

plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

legend_elements = [Line2D([0], [0], marker='o', color='w', label='test_mu=0',
                          markerfacecolor='tab:blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='test_mu=3',
                          markerfacecolor='tab:pink', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='test_mu=6',
                          markerfacecolor='tab:green', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='test_mu=9',
                          markerfacecolor='tab:orange', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='test_mu=12',
                          markerfacecolor='tab:olive', markersize=10)
                   ]

plt.legend(handles=legend_elements, loc='lower left', fontsize = 12)

plt.show()