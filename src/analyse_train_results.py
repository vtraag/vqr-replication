#%% Load libraries
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.transforms import blended_transform_factory
from common import extract_variable, percentile, results_dir, data_dir
import numpy as np

#%% Load the original data

inst_df = pd.read_csv(data_dir / 'institutional.csv')
metric_df = pd.read_csv(data_dir / 'metrics.csv')

# Ensure the dataframe is correctly sorted
metric_df.index.name = 'paper_id'
metric_df = metric_df.sort_index()

# Calculate the average review score
metric_df['REV_SCORE'] = metric_df[['REV_1_SCORE', 'REV_2_SCORE']].mean(axis=1)

#%% Plot figures

prediction_type = 'review'

citation_scores = {
  'ncs': 'NCS',
  'njs': 'NJS',
  'PERCENTILE_INDICATOR_VALUE': 'Perc. Journal',
  'PERCENTILE_CITATIONS': 'Perc. Cit'}

beta_dfs = []
sigma_dfs = []

for citation_score, citation_score_title in citation_scores.items():
  draws_df = pd.read_csv(results_dir / citation_score / prediction_type / 'train_draws.csv')
  
  # Create output dir
  output_dir = results_dir / 'figures' / citation_score / 'train'
  output_dir.mkdir(parents=True, exist_ok=True)
  
  #%% Plot results for beta distribution
  
  beta_df = extract_variable(draws_df, 'beta', axis='columns')
  alpha_nonzero_cit_df = extract_variable(draws_df, 'alpha_nonzero_cit', axis='columns')  
  beta_nonzero_cit_df = extract_variable(draws_df, 'beta_nonzero_cit', axis='columns')
  
  beta_df = pd.concat([beta_df, alpha_nonzero_cit_df, beta_nonzero_cit_df], axis=1)
  beta_df = pd.concat([beta_df], 
                       keys=[citation_score_title], 
                       names=['citation_score'], 
                       axis=1)
  beta_df.columns.names = ['citation_score', 'variable', 'GEV']
  beta_dfs.append(beta_df)

  sns.set_style('whitegrid')
  sns.set_palette('Set1')

  g = sns.catplot(beta_df.melt(), x='value', y='GEV', hue='variable',
              kind='violin', scale='width',
              linewidth=1, alpha=0.8, inner=None,
              height=5, aspect=0.7,
              palette='Set1').set(title=citation_score_title)
  
  yticks = g.ax.get_yticks()
  inbetween_y = (yticks[1:] + yticks[:-1])/2
  for y in inbetween_y:
    g.ax.axhline(y, color='gray', linestyle='dotted')

  # replace labels
  new_labels = [r'$\beta$', 
                r'$\alpha_{\bar{0}}$',
                r'$\beta_{\bar{0}}$']
  for t, l in zip(g._legend.texts, new_labels):
      t.set_text(l)

  sns.move_legend(g, "upper right")

  plt.savefig(output_dir / 'beta.pdf', bbox_inches='tight')
  plt.close()

  #%% Plot results for the various sigma distributions
  
  sigma_paper_value_df = extract_variable(draws_df, 'sigma_paper_value', axis='columns')
  sigma_cit_df = extract_variable(draws_df, 'sigma_cit', axis='columns')
  sigma_review_df = extract_variable(draws_df, 'sigma_review', axis='columns')
  
  sigma_df = pd.concat([sigma_paper_value_df,
                        sigma_cit_df,
                        sigma_review_df],axis=1)
  sigma_df = pd.concat([sigma_df], 
                       keys=[citation_score_title], 
                       names=['citation_score'], 
                       axis=1)  
  sigma_df.columns.names = ['citation_score', 'variable', 'GEV']
  
  sigma_df = sigma_df.rename(columns={'sigma_paper_value': 'Paper value',
                           'sigma_cit': 'Citation',
                           'sigma_review': 'Review'})
  sigma_dfs.append(sigma_df)

  g = sns.catplot(sigma_df.melt(), x='value', y='GEV', hue='variable',
              kind='violin', scale='width',
              linewidth=1, alpha=0.8, inner=None,
              height=5, aspect=0.7,
              palette='Set1').set(title=citation_score_title)

  yticks = g.ax.get_yticks()
  inbetween_y = (yticks[1:] + yticks[:-1])/2
  for y in inbetween_y:
    g.ax.axhline(y, color='gray', linestyle='dotted')
  
  # Make sure the x-limit are not larger than 2, to limit when
  # plotting results for the PERCENTILE indicators.
  xlim = g.ax.get_xlim()
  g.ax.set_xlim(xlim[0], min(xlim[1], 2))
  
  plt.xlabel(r'$\sigma$')

  sns.move_legend(g, "upper right")
  
  plt.savefig(output_dir / 'sigma.pdf', bbox_inches='tight')
  plt.close()

  #%% Example posterior distribution for reviews
  suitable_examples = metric_df.dropna().query('REV_1_SCORE != REV_2_SCORE')
  median_paper_id = suitable_examples[citation_score].sort_values().index[suitable_examples.shape[0]//2]  
  posterior_df = draws_df[f'review_score_ppc[{median_paper_id}]']
  observation = metric_df.loc[median_paper_id]

  min_rev = observation[['REV_1_SCORE', 'REV_2_SCORE']].astype(float).argmin() + 1

  fix, ax = plt.subplots(figsize=(4, 3))
  g = sns.histplot(posterior_df, binwidth=1, 
                   stat='probability',
                   element='step', 
                   label='Posterior',
                   alpha=0.3)
  
  trans = blended_transform_factory(x_transform=g.transData, y_transform=g.transAxes)

  g.axvline(observation['REV_1_SCORE'], color='black')
  g.annotate('Reviewer 1', 
             xy=(observation['REV_1_SCORE'], 0.9), 
             xycoords=trans,
             xytext=(-5 if min_rev == 1 else 5, 0), 
             textcoords="offset points",             
             ha='right' if min_rev == 1 else 'left')
  
  g.axvline(observation['REV_2_SCORE'], color='black')
  g.annotate('Reviewer 2', 
             xy=(observation['REV_2_SCORE'], 0.9), 
             xycoords=trans,             
             xytext=(-5 if min_rev == 2 else 5, 0), 
             textcoords="offset points",
             ha='right' if min_rev == 2 else 'left')

  g.set_xlabel('Review score')

  sns.despine()

  plt.savefig(output_dir / 'review_posterior_example.pdf', bbox_inches='tight')
  plt.close()

  #%% Example posterior distribution for citations
  posterior_df = draws_df[f'citation_ppc[{median_paper_id}]']
  observation = metric_df.loc[median_paper_id, citation_score]

  if 'PERCENTILE' in citation_score:
    clip = [0, 1]
    observation /= 100
  else:
    clip = [0, 5]

  fix, ax = plt.subplots(figsize=(4, 3))
  g = sns.kdeplot(posterior_df, fill=True, clip=clip, alpha=0.3)
  
  if 'PERCENTILE' in citation_score:  
    g.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
  
  g.axvline(observation, color='black')

  trans = blended_transform_factory(x_transform=g.transData, y_transform=g.transAxes)
  g.annotate('Observed citation score', 
             xy=(observation, 0.9), 
             xycoords=trans,
             xytext=(5, 0), 
             textcoords="offset points",
             ha='left')

  g.set_xlabel('Citation score')
  g.set_xlim(clip)

  sns.despine()

  plt.savefig(output_dir / 'citation_posterior_example.pdf', bbox_inches='tight')
  plt.close()  

  #%% Create summary dataframe
  
  summary_df = draws_df.agg(['mean', 'std', percentile(2.5), percentile(97.5)]).T
  
  #%% Plot results for review, observed vs. posterior
  extract_df = extract_variable(summary_df, 
                                'review_score_ppc', 
                                axis='index',
                                index_dtypes=[int])
  extract_df = (extract_df
                .sort_index()
                .droplevel(level=0, axis='index')
               )
  extract_df['observed'] = metric_df['REV_SCORE']
  
  sns.set_palette('Set1')
  sns.set_style('ticks')
  
  fig, ax = plt.subplots(figsize=(3.5,3.5))
  g = sns.scatterplot(extract_df, x='observed', y='mean', alpha=0.4)
  g.set_title(citation_score_title)
  g.axline(xy1=(25, 25), slope=1, color='black')

  sns.despine()

  plt.xlabel('Observed review score')
  plt.ylabel('Posterior predicted review score')
  plt.savefig(output_dir / 'review_score_ppc.pdf', bbox_inches='tight')
  plt.close()

  #%% Plot results for citation, observed vs. posterior
  extract_df = extract_variable(summary_df, 
                                'citation_ppc', 
                                axis='index',
                                index_dtypes=[int])
  extract_df = (extract_df
                .sort_index()
                .droplevel(level=0, axis='index')
               )
  extract_df['observed'] = metric_df[citation_score]
  
  fig, ax = plt.subplots(figsize=(3.5,3.5))
  g = sns.scatterplot(extract_df, x='observed', y='mean', alpha=0.4)
  g.set_title(citation_score_title)

  if not 'PERCENTILE' in citation_score:
    plt.xscale('log')
    plt.yscale('log')
    g.axline(xy1=(0.5, 0.5), xy2=(1.5, 1.5), color='black')
  else:
    g.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    g.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    g.axline(xy1=(40, 0.4), xy2=(60, 0.6), color='black')

  sns.despine()
  
  plt.xlabel('Observed citation score')
  plt.ylabel('Posterior predicted citation score')
  plt.savefig(output_dir / 'citation_ppc.pdf', bbox_inches='tight')
  plt.close()

  #%% Plot results for observed citations vs. inferred paper value 
  extract_df = extract_variable(summary_df, 
                                'value_per_paper', 
                                axis='index',
                                index_dtypes=[int])
  extract_df = (extract_df
                .sort_index()
                .droplevel(level=0, axis='index')
               )
  extract_df['observed'] = metric_df[citation_score]
  
  fig, ax = plt.subplots(figsize=(3.5,3.5))
  g = sns.scatterplot(extract_df, x='observed', y='mean', alpha=0.4)    
  g.set_title(citation_score_title)

  if not 'PERCENTILE' in citation_score:  
    plt.xscale('log')
  
  plt.xlabel('Observed citation score')
  plt.ylabel('Paper value')
  plt.savefig(output_dir / 'citation_paper_value.pdf', bbox_inches='tight')
  plt.close()
        
  #%% Plot results for observed review scores vs. inferred paper value 
  extract_df = extract_variable(summary_df, 
                                'value_per_paper', 
                                axis='index',
                                index_dtypes=[int])
  extract_df = (extract_df
                .sort_index()
                .droplevel(level=0, axis='index')
               )
  extract_df['observed'] = metric_df['REV_SCORE']
  
  fig, ax = plt.subplots(figsize=(3.5,3.5))  
  g = sns.scatterplot(extract_df, x='observed', y='mean', alpha=0.4)      
  g.set_title(citation_score_title)

  plt.xlabel('Observed review score')
  plt.ylabel('Paper value')
  plt.savefig(output_dir / 'review_score_paper_value.pdf', bbox_inches='tight')
  plt.close()

#%% Set output directory

output_dir = results_dir / 'figures'

#%% Create figure with all beta parameters for all GEV simultaneously

all_beta_df = pd.concat(beta_dfs)

sns.set_style('whitegrid')
sns.set_palette('Set1')

g = (
    sns.catplot(all_beta_df.melt(), x='value', y='GEV', hue='variable',
            col='citation_score', col_wrap=2,
            kind='violin', scale='width',
            linewidth=1, alpha=0.8, inner=None,
            height=5, aspect=0.7,
            palette='Set1')
    .set_titles("{col_name}")
    )

for ax in g.axes:
  yticks = ax.get_yticks()
  inbetween_y = (yticks[1:] + yticks[:-1])/2
  for y in inbetween_y:
    ax.axhline(y, color='gray', linestyle='dotted')

# replace labels
new_labels = [r'$\beta$', 
              r'$\alpha_{\bar{0}}$',
              r'$\beta_{\bar{0}}$']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)

sns.move_legend(g, "upper right")

plt.savefig(output_dir / 'beta.pdf', bbox_inches='tight')
plt.close()  

#%% Create figure with all sigma parameters for all GEV simultaneously

all_sigma_df = pd.concat(sigma_dfs)

sns.set_style('whitegrid')
sns.set_palette('Set1')

g = (
    sns.catplot(all_sigma_df.melt(), x='value', y='GEV', hue='variable',
            col='citation_score', col_wrap=2,
            kind='violin', scale='width',
            linewidth=1, alpha=0.8, inner=None,
            height=5, aspect=0.7,
            palette='Set1')
    .set_axis_labels('$\sigma$', 'GEV')
    .set_titles("{col_name}")
    )
  
for ax in g.axes:
  yticks = ax.get_yticks()
  inbetween_y = (yticks[1:] + yticks[:-1])/2
  for y in inbetween_y:
    ax.axhline(y, color='gray', linestyle='dotted')

  # Make sure the x-limit are not larger than 2, to limit when
  # plotting results for the PERCENTILE indicators.
  xlim = ax.get_xlim()
  ax.set_xlim(xlim[0], min(xlim[1], 2))

sns.move_legend(g, "upper right")

plt.savefig(output_dir / 'sigma.pdf', bbox_inches='tight')
plt.close()