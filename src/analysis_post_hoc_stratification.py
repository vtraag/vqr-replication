#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os
from common import data_dir, figure_dir

#%% Read institutional data
stratification_df = pd.read_csv(data_dir / 'institutional.csv', index_col='INSTITUTION_ID')

# Aggregate to the institutional level
inst_df = stratification_df.groupby('INSTITUTION_ID').sum()
inst_df['prop_pubs_in_study'] = inst_df['in_sample'].fillna(0)/inst_df['VQR_submissions']

#%% Take samples

n_samples = 1000
n_pubs_samples = []

for GEV, GEV_df in stratification_df.groupby('GEV_id'):

  # We simply replicate the institution ID as frequently as there are submissions
  # This way we can take a simple sample from all these publications.
  all_pubs = np.repeat(GEV_df.index, GEV_df['VQR_submissions'])

  # Consider the sample size that was observed in the empirical sample.
  # Due to minor differences of some classifications, this is not exactly 10%,
  # otherwise we could have simply sampled exactly 10% in each GEV.
  sample_size = (int)(GEV_df['in_sample'].sum())
  for i in range(n_samples):

    # Take sample
    sample = pd.Series(np.random.choice(all_pubs, sample_size, replace=False))

    # Group by institution
    n_pubs_sample = sample.groupby(sample).size()
    n_pubs_sample.index.name = 'INSTITUTION_ID'

    # Add other relevant information
    n_pubs_sample = (pd.DataFrame(n_pubs_sample)
                     .assign(GEV_id=GEV)
                     .set_index('GEV_id', append=True)
                    )
    n_pubs_sample.columns = [i]

    # Add it to our list of samples
    n_pubs_samples.append(n_pubs_sample)

#%% Create dataframe of all samples at the overall institutional level.
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
plt.savefig(figure_dir / 'stratification.pdf', bbox_inches='tight')
