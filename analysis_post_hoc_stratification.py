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

inst_df = stratification_df.groupby('INSTITUTION_ID').sum()
inst_df['prop_pubs_in_study'] = inst_df['in_sample'].fillna(0)/inst_df['VQR_submissions']

#%%

n_samples = 1000
n_pubs_samples = []

for GEV, GEV_df in stratification_df.groupby('GEV_id'):
  all_pubs = np.repeat(GEV_df.index, GEV_df['VQR_submissions'])

  sample_size = (int)(GEV_df['in_sample'].sum())
  for i in range(n_samples):
    sample = pd.Series(np.random.choice(all_pubs, sample_size, replace=False))
    n_pubs_sample = sample.groupby(sample).size()
    n_pubs_sample.index.name = 'INSTITUTION_ID'
    n_pubs_sample = (pd.DataFrame(n_pubs_sample)
                     .assign(GEV_id=GEV)
                     .set_index('GEV_id', append=True)
                    )
    n_pubs_sample.columns = [i]
    n_pubs_samples.append(n_pubs_sample)

#%%
sample_df = (pd.concat(n_pubs_samples, axis=1)
             .melt(var_name='sample',ignore_index=False)
             .groupby(['INSTITUTION_ID', 'sample'])
             .sum()
             .unstack('sample')
             .fillna(0)
             .div(inst_df['VQR_submissions'], axis='index')
            )

#%% Plot histogram

plt.figure(figsize=(4, 3))
bins = np.linspace(0, 15/100, 16)
sns.histplot(inst_df, x='prop_pubs_in_study',
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
