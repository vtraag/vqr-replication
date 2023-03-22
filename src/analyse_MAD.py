#%% Import libraries
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from common import gev_names_df, extract_variable, results_dir, data_dir

#%% Load the original data

inst_df = pd.read_csv(data_dir / 'institutional.csv')
metric_df = pd.read_csv(data_dir / 'metrics.csv')

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

avg_abs_diff_dfs = []
MAD_dfs = []
inst_avg_abs_diff_dfs = []
inst_MAD_dfs = []
inst_MAPD_dfs = []

for citation_score in citation_scores:

  draws_df = pd.read_csv(results_dir / citation_score / 'citation' / 'test_draws.csv')
  
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

  avg_abs_diff_df = abs_diff_df.mean(axis=1)
  avg_abs_diff_df.name = f'{citation_score}_review_pred'
  avg_abs_diff_dfs.append(avg_abs_diff_df)

  MAD_df = abs_diff_df.groupby(metric_df['GEV']).mean()
  MAD_dfs.append(MAD_df)

  #%% Aggregate to institutional level

  # Create institutional predictions
  inst_gev_df = metric_df[['INSTITUTION_ID', 'GEV']]
  metric_pred_df = pd.merge(inst_gev_df, pred_df,
                            left_index=True, right_index=True)

  inst_pred_df = metric_pred_df.groupby(['INSTITUTION_ID', 'GEV']).mean()

  inst_pred_df.columns = pd.MultiIndex.from_tuples(c for c in inst_pred_df.columns)

  inst_abs_diff_df = np.abs(inst_pred_df.subtract(inst_metric_df['REV_2_SCORE'], axis='index'))

  inst_avg_abs_diff_df = inst_abs_diff_df.mean(axis=1)
  inst_avg_abs_diff_df.name = f'{citation_score}_review_pred'
  inst_avg_abs_diff_dfs.append(inst_avg_abs_diff_df)

  inst_perc_abs_diff_df = (
                           np.abs(inst_pred_df
                                 .multiply(inst_metric_df['n_pubs'], axis='index')
                                 .subtract(inst_metric_df['REV_2_SCORE']*inst_metric_df['n_pubs'], axis='index'))
                           .div(inst_metric_df['REV_2_SCORE']*inst_metric_df['n_pubs'], axis='index')
                          )

  inst_MAD_df = inst_abs_diff_df.groupby('GEV').mean()
  inst_MAPD_df = inst_perc_abs_diff_df.groupby('GEV').mean()

  inst_MAD_dfs.append(inst_MAD_df)
  inst_MAPD_dfs.append(inst_MAPD_df)

#%% Also add review MAD

draws_df = pd.read_csv(results_dir / 'ncs' / 'review' / 'test_draws.csv')

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

avg_abs_diff_df = abs_diff_df.mean(axis=1)
avg_abs_diff_df.name = f'review_review_pred'
avg_abs_diff_dfs.append(avg_abs_diff_df)

MAD_df = abs_diff_df.groupby(metric_df['GEV']).mean()

MAD_dfs.append(MAD_df)

#%% Aggregate to institutional level

# Create institutional predictions
inst_gev_df = metric_df[['INSTITUTION_ID', 'GEV']]
metric_pred_df = pd.merge(inst_gev_df, pred_df,
                          left_index=True, right_index=True)

inst_pred_df = metric_pred_df.groupby(['INSTITUTION_ID', 'GEV']).mean()

inst_pred_df.columns = pd.MultiIndex.from_tuples(c for c in inst_pred_df.columns)

inst_abs_diff_df = np.abs(inst_pred_df.subtract(inst_metric_df['REV_2_SCORE'], axis='index'))

inst_avg_abs_diff_df = inst_abs_diff_df.mean(axis=1)
inst_avg_abs_diff_df.name = f'review_review_pred'
inst_avg_abs_diff_dfs.append(inst_avg_abs_diff_df)

inst_perc_abs_diff_df = (
                         np.abs(inst_pred_df
                               .multiply(inst_metric_df['n_pubs'], axis='index')
                               .subtract(inst_metric_df['REV_2_SCORE']*inst_metric_df['n_pubs'], axis='index'))
                         .div(inst_metric_df['REV_2_SCORE']*inst_metric_df['n_pubs'], axis='index')
                        )

inst_MAD_df = inst_abs_diff_df.groupby('GEV').mean()
inst_MAPD_df = inst_perc_abs_diff_df.groupby('GEV').mean()

inst_MAD_dfs.append(inst_MAD_df)
inst_MAPD_dfs.append(inst_MAPD_df)

#%% Concatenate all dataframes

avg_abs_diff_df = pd.concat(avg_abs_diff_dfs, axis=1)
MAD_df = pd.concat(MAD_dfs, axis=1)
inst_avg_abs_diff_df = pd.concat(inst_avg_abs_diff_dfs, axis=1)
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
          .rename(columns={'variable_0': 'variable'})
          .reset_index()
        )

sns.set_style('whitegrid')
sns.set_context('paper', rc={'lines.linewidth': 0.5,
                            'grid.linewidth': 0.5,
                            'axes.linewidth': 0.5,
                            'xtick.major.width': 0.5,
                            'ytick.major.width': 0.5,
                            'patch.linewidth': 1.0})
sns.set_palette('Set1')

g = sns.catplot(plt_df,
            x='MAD', y='GEV', hue='variable',
            kind='bar', palette='Set1',
            errorbar=('pi', 95), errwidth=0.5,
            height=5, aspect=1.2)

yticks = g.ax.get_yticks()
inbetween_y = (yticks[1:] + yticks[:-1])/2
for y in inbetween_y:
  g.ax.axhline(y, color='gray', linestyle='dotted')

sns.move_legend(g, 'upper right')
g.set_ylabels('')

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
          .rename(columns={'variable_0': 'variable'})
          .reset_index()
        )

g = sns.catplot(plt_df,
            x='MAD', y='GEV', hue='variable',
            kind='bar', palette='Set1',
            errorbar=('pi', 95), errwidth=0.5,
            height=5, aspect=1.2)

yticks = g.ax.get_yticks()
inbetween_y = (yticks[1:] + yticks[:-1])/2
for y in inbetween_y:
  g.ax.axhline(y, color='gray', linestyle='dotted')

sns.move_legend(g, 'upper right')
g.set_ylabels('')

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
          .rename(columns={'variable_0': 'variable'})
          .reset_index()
        )

g = sns.catplot(plt_df,
            x='MAPD', y='GEV', hue='variable',
            kind='bar', palette='Set1',
            errorbar=('pi', 95), errwidth=0.5,
            height=5, aspect=1.2)

yticks = g.ax.get_yticks()
inbetween_y = (yticks[1:] + yticks[:-1])/2
for y in inbetween_y:
  g.ax.axhline(y, color='gray', linestyle='dotted')

sns.move_legend(g, 'upper right')
g.set_ylabels('')

g.ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

plt.savefig(output_dir / 'MAPD_institutional.pdf', bbox_inches='tight')

#%% Plot distribution of average MADs at paper level

plt_df = avg_abs_diff_df.copy()
plt_df['GEV'] = metric_df['GEV']
plt_df = (plt_df
          .rename(columns={'ncs_review_pred': 'NCS',
                           'njs_review_pred': 'NJS',
                           'PERCENTILE_INDICATOR_VALUE_review_pred': 'Perc. Journal',
                           'PERCENTILE_CITATIONS_review_pred': 'Perc. Cit',
                           'review_review_pred': 'Review'})
          .melt(value_name='Abs. Diff.',
                id_vars=['GEV'],
                ignore_index=False)
          .rename(columns={'variable_0': 'variable'})
          .reset_index()
        )

clip = [0,15]
g = sns.catplot(plt_df,
            x='Abs. Diff.', y='GEV', hue='variable',
            kind='violin', scale='width',
            cut=0, clip=clip,
            height=7, aspect=0.8)

# Make edge colors of violins white
for ax in g.axes.flatten():
  for col in ax.collections:
    col.set_edgecolor('white')

yticks = g.ax.get_yticks()
inbetween_y = (yticks[1:] + yticks[:-1])/2
for y in inbetween_y:
  g.ax.axhline(y, color='gray', linestyle='dotted')

g.ax.set_xlim(clip)

sns.move_legend(g, 'upper right')
g.set_ylabels('')

plt.savefig(output_dir / 'avg_abs_diff_dist.pdf', bbox_inches='tight')

#%% Plot distribution of average MADs at institutional level

plt_df = (inst_avg_abs_diff_df
          .rename(columns={'ncs_review_pred': 'NCS',
                           'njs_review_pred': 'NJS',
                           'PERCENTILE_INDICATOR_VALUE_review_pred': 'Perc. Journal',
                           'PERCENTILE_CITATIONS_review_pred': 'Perc. Cit',
                           'review_review_pred': 'Review'})
          .melt(value_name='Abs. Diff.',
                ignore_index=False)
          .rename(columns={'variable_0': 'variable'})
          .reset_index()
        )

clip = [0,15]
g = sns.catplot(plt_df,
            x='Abs. Diff.', y='GEV', hue='variable',
            kind='violin', scale='width',
            cut=0, clip=clip,
            height=7, aspect=0.8)

# Make edge colors of violins white
for ax in g.axes.flatten():
  for col in ax.collections:
    col.set_edgecolor('white')

yticks = g.ax.get_yticks()
inbetween_y = (yticks[1:] + yticks[:-1])/2
for y in inbetween_y:
  g.ax.axhline(y, color='gray', linestyle='dotted')

g.ax.set_xlim(clip)

sns.move_legend(g, 'upper right')
g.set_ylabels('')

plt.savefig(output_dir / 'inst_avg_abs_diff_dist.pdf', bbox_inches='tight')

# %%
