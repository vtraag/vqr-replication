#%% Load libraries
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from common import extract_variable, percentile

#%% Set the directory we want to transform the fit results for
results_dir = Path('../results/20230118233207')

#%% Load the original data

inst_df = pd.read_csv('../data/public/institutional.csv')
metric_df = pd.read_csv('../data/public/metrics.csv')

# Ensure the dataframe is correctly sorted
metric_df.index.name = 'paper_id'
metric_df = metric_df.sort_index()

# Calculate the average review score
metric_df['REV_SCORE'] = metric_df[['REV_1_SCORE', 'REV_2_SCORE']].mean(axis=1)

#%%

prediction_type = 'prior'

citation_scores = ['ncs', 
                   'njs',
                   'PERCENTILE_INDICATOR_VALUE', 
                   'PERCENTILE_CITATIONS']

for citation_score in citation_scores:
  draws_df = pd.read_csv(results_dir / citation_score / prediction_type / 'draws.csv')
  
  # Create output dir
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
  plt.close()

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
  
  # Make sure the x-limit are not larger than 2, to limit when
  # plotting results for the PERCENTILE indicators.
  xlim = g.ax.get_xlim()
  g.ax.set_xlim(xlim[0], min(xlim[1], 2))
  
  plt.xlabel(r'$\sigma$')
  
  plt.savefig(output_dir / 'sigma.pdf', bbox_inches='tight')
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
  
  g = sns.scatterplot(extract_df, x='observed', y='mean', alpha=0.4)

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
  
  g = sns.scatterplot(extract_df, x='observed', y='mean', alpha=0.4)  
  
  if not 'PERCENTILE' in citation_score:
  plt.xscale('log')
  plt.yscale('log')
  
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
  
  g = sns.scatterplot(extract_df, x='observed', y='mean', alpha=0.4)    
  
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
  
  g = sns.scatterplot(extract_df, x='observed', y='mean', alpha=0.4)      
  
  plt.xlabel('Observed review score')
  plt.ylabel('Paper value')
  plt.savefig(output_dir / 'review_score_paper_value.pdf', bbox_inches='tight')
  plt.close()