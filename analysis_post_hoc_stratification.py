#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

colors = sns.palettes.mpl_palette('Set1', 8)

figure_dir = '../figures/'
if not os.path.exists(figure_dir):
  os.makedirs(figure_dir)

#%% Read institutional data
stratification_df = pd.read_csv('../data/public/institutional.csv', index_col='INSTITUTION_ID')

stratification_df['prop_pubs_in_study'] = stratification_df['in_sample'].fillna(0)/stratification_df['VQR_submissions']

#%%
all_pubs = np.repeat(stratification_df.index, stratification_df['VQR_submissions'])

n_samples = 1000
sample_size = (int)(stratification_df['in_sample'].sum())
n_pubs_samples = []
for i in range(n_samples):
  sample = pd.Series(np.random.choice(all_pubs, sample_size, replace=False))
  n_pubs_sample = sample.groupby(sample).size()
  n_pubs_samples.append(n_pubs_sample)

sample_df = (pd.concat(n_pubs_samples, axis=1)
             .fillna(0)
             .sort_index()
             .div(stratification_df['VQR_submissions'], axis='index')
            )

#%% Plot histogram

plt.figure(figsize=(4, 3))
bins = np.linspace(0, 15/100, 16)
sns.histplot(stratification_df, x='prop_pubs_in_study',
             bins=bins, element='step', alpha=0.4,
             label='Empirical sample')
sns.kdeplot(data=sample_df.melt(), x='value', label='Theoretical expectation')
plt.grid(axis='y', ls=':')
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
plt.ylabel('Frequency')
plt.xlabel('Sample %')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), 
           ncol=2,
           frameon=False)
plt.xlim(0, 0.15)
sns.despine()
plt.savefig(os.path.join(figure_dir, 'stratification.pdf'), bbox_inches='tight')
