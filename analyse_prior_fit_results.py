#%%
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from analysis_functions import extract_variable

#%% Set the directory we want to transform the fit results for

results_dir = Path('../results/20230118233207')

#%% Load the original data

inst_df = pd.read_csv('../data/public/institutional.csv')
metric_df = pd.read_csv('../data/public/metrics.csv')

#%%

prediction_type = 'prior'


citation_scores = ['ncs', 
                   'njs',
                   'PERCENTILE_INDICATOR_VALUE', 
                   'PERCENTILE_CITATIONS']

for citation_score in citation_scores:
  draws_df = pd.read_csv(results_dir / citation_score / prediction_type / 'draws.csv')
  
  output_dir = results_dir / 'figures' / citation_score / prediction_type
  output_dir.mkdir(parents=True, exist_ok=True)
  
  #%% Plot results for beta distribution
  
  beta_df = extract_variable(draws_df, 'beta', axis='columns')
  beta_nonzero_cit_df = extract_variable(draws_df, 'beta_nonzero_cit', axis='columns')
  
  beta_df = pd.concat([beta_df, beta_nonzero_cit_df], axis=1)
  beta_df.columns.names = ['variable', 'GEV']
  
  sns.set_palette('Set1')
  
  g = sns.catplot(beta_df.melt(), x='value', y='GEV', hue='variable',
              kind='violin',
              linewidth=1, alpha=0.8, inner=None,
              height=5, aspect=1,
              palette='Set1')
  
  yticks = g.ax.get_yticks()
  inbetween_y = (yticks[1:] + yticks[:-1])/2
  for y in inbetween_y:
    g.ax.axhline(y, color='gray', linestyle='dotted')
  
  plt.xlabel(r'$\beta$')
  
  plt.savefig(output_dir / 'beta.pdf', bbox_inches='tight')
  
  #%% Plot results for the various sigma distributions
  
  sigma_paper_value_df = extract_variable(draws_df, 'sigma_paper_value', axis='columns')
  sigma_cit_df = extract_variable(draws_df, 'sigma_cit', axis='columns')
  sigma_review_df = extract_variable(draws_df, 'sigma_review', axis='columns')
  
  sigma_df = pd.concat([sigma_paper_value_df,
                        sigma_cit_df,
                        sigma_review_df],axis=1)
  
  sigma_df.columns.names = ['variable', 'GEV']
  
  sigma_df = sigma_df.rename(columns={'sigma_paper_value': 'Paper value',
                           'sigma_cit': 'Citation',
                           'sigma_review': 'Review'})
  
  sns.set_palette('Set1')
  
  g = sns.catplot(sigma_df.melt(), x='value', y='GEV', hue='variable',
              kind='violin',
              linewidth=1, alpha=0.8, inner=None,
              height=8, aspect=0.7,
              palette='Set1')
  
  yticks = g.ax.get_yticks()
  inbetween_y = (yticks[1:] + yticks[:-1])/2
  for y in inbetween_y:
    g.ax.axhline(y, color='gray', linestyle='dotted')
  
  plt.xlabel(r'$\sigma$')
  
  plt.savefig(output_dir / 'sigma.pdf', bbox_inches='tight')