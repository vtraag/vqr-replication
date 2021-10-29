import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

colors = sns.palettes.mpl_palette('Set1', 8)

results_dir = '../results/'
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

#%% Read institutional data
stratification_df = pd.read_csv('../data/public/institutional.csv', index_col='INSTITUTION_ID')

stratification_df['prop_pubs_in_study'] = stratification_df['in_analysis']/stratification_df['VQR_submissions']
#%% Plot histogram

plt.figure(figsize=(4, 3))
bins = np.linspace(0, 15, 16)
plt.hist(100*stratification_df['prop_pubs_in_study'],
         bins=bins, color=colors[1],histtype='stepfilled')
plt.grid(axis='y', ls=':')
plt.ylabel('Frequency')
plt.xlabel('Sample %')
sns.despine()
plt.savefig(os.path.join(results_dir, 'stratification.pdf'), bbox_inches='tight')