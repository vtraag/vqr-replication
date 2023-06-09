#%% Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from common import gev_names_df, figure_dir, data_dir
import os

#%% Read data
institution_df = pd.read_csv(data_dir / 'institutional.csv', index_col='INSTITUTION_ID')

#%% Aggregate at GEV level
GEV_df = institution_df.groupby('GEV_id').sum()
GEV_df = pd.merge(GEV_df, gev_names_df,
                  left_index=True, right_on='GEV_id')

GEV_df = GEV_df.set_index('GEV_id').loc[gev_names_df['GEV_id'],:]

#%% Plot missing publications

sns.set_style('white')
sns.set_palette('Set1')
plt.figure(figsize=(6, 4))

xlabels = GEV_df['GEV']
x = np.arange(len(xlabels))
gap = 0.01
group_gap = 0.3
width = 0.8
bars = []

# Create stacked bar charts
h = 0
y = GEV_df['no_reviewers_2'];
b = plt.bar(x=x, height=y, bottom=h, width=width, label='2 reviewers')
h += y
y = GEV_df['no_reviewers_1']
b = plt.bar(x=x, height=y, bottom=h, width=width, label='1 reviewer')
h += y
y = GEV_df['no_reviewers_0']
b = plt.bar(x=x, height=y, bottom=h, width=width, label='No reviewer')

plt.xticks(np.arange(len(xlabels)), xlabels, rotation='vertical')
plt.ylabel('Number of publications')
plt.grid(axis='y', ls=':')
plt.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center',
           frameon=False,
           ncol=3)
sns.despine()
plt.savefig(figure_dir / 'GEV_missing_pubs.pdf', bbox_inches='tight')
