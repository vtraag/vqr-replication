#%% Import libraries
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from common import gev_names_df, extract_variable

#%% Set the directory we want to transform the fit results for

results_dir = Path('../results/20230213201810')

#%% Load the original data

inst_df = pd.read_csv('../data/public/institutional.csv')
metric_df = pd.read_csv('../data/public/metrics.csv')

# We call the original index the paper_id
metric_df = (metric_df
             .reset_index()
             .rename(columns={'index': 'paper_id'})
            )

# Now we merge to get also the GEV names
metric_df = pd.merge(metric_df, gev_names_df, 
                     on='GEV_id', how='left')

# And we use the paper_id again as the index
metric_df = (metric_df
             .set_index('paper_id')
             .sort_index()
            )

#%% Aggregate metrics per institution

g = metric_df.groupby(['INSTITUTION_ID', 'GEV'])
inst_metric_df = g.mean()
inst_metric_df['n_pubs'] = g.size()
 
#%%

citation_scores = ['ncs', 
                   'njs',
                   'PERCENTILE_INDICATOR_VALUE', 
                   'PERCENTILE_CITATIONS']

MAD_dfs = []
inst_MAD_dfs = []
inst_MAPD_dfs = []

for citation_score in citation_scores:

  draws_df = pd.read_csv(results_dir / citation_score / 'citation_prediction' / 'draws.csv')     
  
  #%%
  # Change format
  pred_df = extract_variable(draws_df, 'review_score_ppc', axis='columns', index_dtypes=[int])
  pred_df = pred_df.rename(columns={'review_score_ppc': f'{citation_score}_review_pred'})

  # Reshape the dataframe such that it has the paper identifiers
  # as the index, with the prediction types and the individuals draws
  # as the columns
  pred_df = (pred_df
             .T
             .unstack(level=0)
             .swaplevel(0, 1, axis='columns')
             )

  abs_diff_df = np.abs(pred_df.subtract(metric_df['REV_2_SCORE'], axis='index'))
  
  MAD_df = abs_diff_df.groupby(metric_df['GEV']).median()

  MAD_dfs.append(MAD_df)
  
  #%% Aggregate to institutional level
  
  # Create institutional predictions
  inst_gev_df = metric_df[['INSTITUTION_ID', 'GEV']]
  metric_pred_df = pd.merge(inst_gev_df, pred_df, 
                            left_index=True, right_index=True)
  
  inst_pred_df = metric_pred_df.groupby(['INSTITUTION_ID', 'GEV']).mean()

  inst_pred_df.columns = pd.MultiIndex.from_tuples(c for c in inst_pred_df.columns)
  
  inst_abs_diff_df = np.abs(inst_pred_df.subtract(inst_metric_df['REV_2_SCORE'], axis='index'))
  
  inst_perc_abs_diff_df = (
                           np.abs(inst_pred_df
                                 .multiply(inst_metric_df['n_pubs'], axis='index')
                                 .subtract(inst_metric_df['REV_2_SCORE']*inst_metric_df['n_pubs'], axis='index'))
                           .div(inst_metric_df['REV_2_SCORE']*inst_metric_df['n_pubs'], axis='index')
                          )
  
  inst_MAD_df = inst_abs_diff_df.groupby('GEV').median()
  inst_MAPD_df = inst_perc_abs_diff_df.groupby('GEV').median()
  
  inst_MAD_dfs.append(inst_MAD_df)
  inst_MAPD_dfs.append(inst_MAPD_df)

#%% Also add review MAD

draws_df = pd.read_csv(results_dir / citation_score / 'review_prediction' / 'draws.csv')     

# Change format
pred_df = extract_variable(draws_df, 'review_score_ppc', axis='columns', index_dtypes=[int])
pred_df = pred_df.rename(columns={'review_score_ppc': 'review_review_pred'})

# Reshape the dataframe such that it has the paper identifiers
# as the index, with the prediction types and the individuals draws
# as the columns
pred_df = (pred_df
           .T
           .unstack(level=0)
           .swaplevel(0, 1, axis='columns')
           )

abs_diff_df = np.abs(pred_df.subtract(metric_df['REV_2_SCORE'], axis='index'))

MAD_df = abs_diff_df.groupby(metric_df['GEV']).median()

MAD_dfs.append(MAD_df)

#%% Aggregate to institutional level

# Create institutional predictions
inst_gev_df = metric_df[['INSTITUTION_ID', 'GEV']]
metric_pred_df = pd.merge(inst_gev_df, pred_df, 
                          left_index=True, right_index=True)

inst_pred_df = metric_pred_df.groupby(['INSTITUTION_ID', 'GEV']).mean()

inst_pred_df.columns = pd.MultiIndex.from_tuples(c for c in inst_pred_df.columns)

inst_abs_diff_df = np.abs(inst_pred_df.subtract(inst_metric_df['REV_2_SCORE'], axis='index'))
  
inst_perc_abs_diff_df = (
                         np.abs(inst_pred_df
                               .multiply(inst_metric_df['n_pubs'], axis='index')
                               .subtract(inst_metric_df['REV_2_SCORE']*inst_metric_df['n_pubs'], axis='index'))
                         .div(inst_metric_df['REV_2_SCORE']*inst_metric_df['n_pubs'], axis='index')
                        )

inst_MAD_df = inst_abs_diff_df.groupby('GEV').median()
inst_MAPD_df = inst_perc_abs_diff_df.groupby('GEV').median()

inst_MAD_dfs.append(inst_MAD_df)
inst_MAPD_dfs.append(inst_MAPD_df)

#%% Concatenate all dataframes

MAD_df = pd.concat(MAD_dfs, axis=1)
inst_MAD_df = pd.concat(inst_MAD_dfs, axis=1)
inst_MAPD_df = pd.concat(inst_MAPD_dfs, axis=1)

#%% Create output directory
  
output_dir = results_dir / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)
  
#%% Plot MAD at individual level

plt_df = (MAD_df
          .rename(columns={'ncs_review_pred': 'NCS',
                           'njs_review_pred': 'NJS',
                           'PERCENTILE_INDICATOR_VALUE_review_pred': 'Perc. Journal',
                           'PERCENTILE_CITATIONS_review_pred': 'Perc. Cit',
                           'review_review_pred': 'Review'})
          .melt(value_name='MAD',
                ignore_index=False)
          .reset_index()
        )

sns.catplot(plt_df, 
            x='MAD', y='GEV', hue='variable_0',
            kind='bar', palette='Set1',
            errorbar=('sd', 1.96),
            height=12, aspect=0.9)

plt.savefig(output_dir / 'MAD_individual.pdf', bbox_inches='tight')

#%% Plot MAD at institutional level

plt_df = (inst_MAD_df
          .rename(columns={'ncs_review_pred': 'NCS',
                           'njs_review_pred': 'NJS',
                           'PERCENTILE_INDICATOR_VALUE_review_pred': 'Perc. Journal',
                           'PERCENTILE_CITATIONS_review_pred': 'Perc. Cit',
                           'review_review_pred': 'Review'})
          .melt(value_name='MAD',
                ignore_index=False)
          .reset_index()
        )

sns.catplot(plt_df, 
            x='MAD', y='GEV', hue='variable_0',
            kind='bar', palette='Set1',
            errorbar=('sd', 1.96),
            height=12, aspect=0.9)

plt.savefig(output_dir / 'MAD_institutional.pdf', bbox_inches='tight')

#%% Plot MAPD at institutional level

import matplotlib.ticker as mtick

plt_df = (inst_MAPD_df
          .rename(columns={'ncs_review_pred': 'NCS',
                           'njs_review_pred': 'NJS',
                           'PERCENTILE_INDICATOR_VALUE_review_pred': 'Perc. Journal',
                           'PERCENTILE_CITATIONS_review_pred': 'Perc. Cit',
                           'review_review_pred': 'Review'})
          .melt(value_name='MAPD',
                ignore_index=False)
          .reset_index()
        )

g = sns.catplot(plt_df, 
            x='MAPD', y='GEV', hue='variable_0',
            kind='bar', palette='Set1',
            errorbar=('sd', 1.96),
            height=12, aspect=0.9)

g.ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

plt.savefig(output_dir / 'MAPD_institutional.pdf', bbox_inches='tight')
