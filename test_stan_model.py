#%%
from cmdstanpy import CmdStanModel
import os
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from common import unique_id, nuniq, extract_variable

#%%

now = dt.datetime.now().strftime("%Y%m%d%H%M%S")
output_dir = Path(f'../results/{now}')

#%%

stan_file = os.path.abspath('./model.stan')
model = CmdStanModel(stan_file=stan_file)
print(model)

#%%

inst_df = pd.read_csv('../data/public/institutional.csv')
metric_df = pd.read_csv('../data/public/metrics.csv')

#%%

citation_score = 'ncs'

paper_df = (metric_df
            .query('GEV_id == "1"')\
               [['INSTITUTION_ID',
                 'REV_1_SCORE',
                 'REV_2_SCORE',
                 'ncs',
                 'njs',
                 'PERCENTILE_INDICATOR_VALUE',
                 'PERCENTILE_CITATIONS']]
            .sort_values('INSTITUTION_ID')
           )
paper_df['new_institution_id'] = unique_id(paper_df[['INSTITUTION_ID']])
paper_df['new_paper_id'] = np.arange(paper_df.shape[0]) + 1

paper_df['REV_SCORE'] = (paper_df['REV_1_SCORE'] + paper_df['REV_2_SCORE'])/2

citation_df = (
                paper_df[['new_paper_id', citation_score]]
                .rename(columns={citation_score: 
                                 'citation_score'})
                .dropna()
              )

review_df = (
                pd.concat([paper_df[['new_paper_id', 'REV_1_SCORE']]\
                           .rename(columns={'REV_1_SCORE': 'review_score'}),
                       paper_df[['new_paper_id', 'REV_2_SCORE']]\
                           .rename(columns={'REV_2_SCORE': 'review_score'})],
                      axis=0,
                      ignore_index=True)
                .dropna()
                .sort_values('new_paper_id')    
            )

#%%

data = {
    'N_papers':    paper_df.shape[0],
    'N_institutions':   nuniq(paper_df['new_institution_id']),
    'institution_per_paper': paper_df['new_institution_id'],
    
    'N_review_scores':    review_df.shape[0],        
    'review_score': (review_df['review_score'] - 2).astype('int'), # Should be between 1-28
    'paper_per_review_score': review_df['new_paper_id'],
    
    'N_citation_scores': citation_df.shape[0],
    'citation_score': citation_df['citation_score'],
    'paper_per_citation_score': citation_df['new_paper_id'],
    'citation_percentile_score': 0,
    
    'use_estimated_priors': 0,

    # The below are estimated coefficients from other models.

    # Coefficient of citation
    'beta_mu': 0,
    'beta_sigma': 1,

    # Standard deviation of citation
    'sigma_cit_mu': 0,
    'sigma_cit_sigma': 1,

    # Standard deviation of peer review.
    'sigma_review_mu': 0,
    'sigma_review_sigma': 1,
    
    'alpha_nonzero_cit_mu': 0,
    'alpha_nonzero_cit_sigma': 1,
    
    'beta_nonzero_cit_mu': 0,
    'beta_nonzero_cit_sigma': 1
}

prior_fit = model.sample(data=data, chains=1, 
                         output_dir = output_dir / 'prior')

#%%

print(prior_fit.diagnose())

#%%

prior_draws_df = prior_fit.draws_pd()

prior_summary_df = prior_fit.summary()

#%%
sns.set_palette('Set1')

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
ax = plt.subplot(1, 3, 1)
sns.distplot(prior_draws_df['beta'])
plt.xlabel(r'$\beta$')
plt.yticks([])

ax = plt.subplot(1, 3, 2)
sns.distplot(prior_draws_df['alpha_nonzero_cit'], label=r'$\alpha_{0^+}$')
sns.distplot(prior_draws_df['beta_nonzero_cit'], label=r'$\beta_{0^+}$')
plt.legend(loc='best')
plt.yticks([])
plt.ylabel('')

ax = plt.subplot(1, 3, 3)
sns.distplot(prior_draws_df['sigma_review'], label='Review')
sns.distplot(prior_draws_df['sigma_cit'], label='Citation')
plt.xlabel(r'$\sigma$')
plt.yticks([])
plt.ylabel('')
plt.legend(loc='best')

  
#%%

review_score_ppc_df = extract_variable(prior_summary_df, 'review_score_ppc', axis='index')

plt.plot(paper_df['REV_SCORE'], review_score_ppc_df['Mean'], '.')

plt.xlabel('Observed review score')
plt.ylabel('Posterior predicted review score')

#%%

value_df = extract_variable(prior_summary_df, 'value_per_paper', axis='index')

plt.plot(paper_df['REV_SCORE'], value_df['Mean'], '.')

plt.xlabel('Observed review score')
plt.ylabel('Inferred paper value')

#%%

value_df = extract_variable(prior_summary_df, 'value_per_paper', axis='index')

plt.plot(paper_df[citation_score], value_df['Mean'], '.')

plt.xlabel('Observed citation score')
plt.ylabel('Inferred paper value')

#%%

citation_ppc_df = extract_variable(prior_summary_df, 'citation_ppc', axis='index')

# plt.errorbar(x=paper_df[citation_score], 
#               y=citation_ppc_df['50%'], 
#               yerr=citation_ppc_df[['5%','95%']].T,
              # fmt='.')
plt.plot(paper_df[citation_score], citation_ppc_df['50%'], '.')
#plt.plot(paper_df['PERCENTILE_CITATIONS'], 100*citation_ppc_df['50%'], '.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Observed citation score')
plt.ylabel('Posterior predicted citation score')

#%%

prediction_type = 'citation'

data = {
    'N_papers':    paper_df.shape[0],
    'N_institutions':   nuniq(paper_df['new_institution_id']),
    'institution_per_paper': paper_df['new_institution_id'],
    }

if (prediction_type == 'citation'):
  data |= {
      'N_review_scores':    0,
      'review_score': [],
      'paper_per_review_score': [],
      
      'N_citation_scores': citation_df.shape[0],
      'citation_score': citation_df['citation_score'],
      'paper_per_citation_score': citation_df['new_paper_id'],
      'citation_percentile_score': 0
      }
elif (prediction_type == 'review'):
  single_review_df = (paper_df[['new_paper_id', 'REV_2_SCORE']]\
                      .rename(columns={'REV_2_SCORE': 'review_score'})
                      .dropna()
                      .sort_values('new_paper_id')
                     )  
  data |= {
      'N_review_scores':    single_review_df.shape[0],        
      'review_score': (single_review_df['review_score'] - 2).astype('int'), # Should be between 1-28
      'paper_per_review_score': single_review_df['new_paper_id'],
      
      'N_citation_scores': 0,
      'citation_score': [],
      'paper_per_citation_score': [],
      'citation_percentile_score': 1 if 'PERCENTILE' in citation_score else 0
      }
  
data |= {
    'use_estimated_priors': 1,

    # The below are estimated coefficients from other models.

    # Coefficient of citation
    'beta_mu': prior_summary_df.loc['beta', 'Mean'],
    'beta_sigma': prior_summary_df.loc['beta', 'StdDev'],

    # Standard deviation of citation
    'sigma_cit_mu': prior_summary_df.loc['sigma_cit', 'Mean'],
    'sigma_cit_sigma': prior_summary_df.loc['sigma_cit', 'StdDev'],

    # Standard deviation of peer review.
    'sigma_review_mu': prior_summary_df.loc['sigma_review', 'Mean'],
    'sigma_review_sigma': prior_summary_df.loc['sigma_review', 'StdDev'],
    
    'alpha_nonzero_cit_mu': prior_summary_df.loc['alpha_nonzero_cit', 'Mean'],
    'alpha_nonzero_cit_sigma': prior_summary_df.loc['alpha_nonzero_cit', 'StdDev'],
    
    'beta_nonzero_cit_mu': prior_summary_df.loc['beta_nonzero_cit', 'Mean'],
    'beta_nonzero_cit_sigma': prior_summary_df.loc['beta_nonzero_cit', 'StdDev']
}

fit = model.sample(data=data, chains=1, 
                   output_dir = output_dir / f'predict_{prediction_type}')

#%%

print(fit.diagnose())

#%%

draws_df = fit.draws_pd()

summary_df = fit.summary()

#%%

plt.plot(draws_df['beta'])

#%%
#paper_id = 385 # High ncs
paper_id = 106 # Low ncs
sns.distplot(prior_draws_df[f'citation_ppc[{paper_id}]'])
sns.distplot(draws_df[f'citation_ppc[{paper_id}]'])
plt.axvline(citation_df.query(f'new_paper_id == {paper_id}').iloc[0, 1], color='k')

  
#%%

review_score_ppc_df = extract_variable(summary_df, 'review_score_ppc', axis='index')

plt.plot(paper_df['REV_SCORE'], review_score_ppc_df['Mean'], '.')

plt.xlabel('Observed review score')
plt.ylabel('Posterior predicted review score')

#%%

value_df = extract_variable(summary_df, 'value_per_paper', axis='index')

plt.plot(paper_df['REV_SCORE'], value_df['Mean'], '.')

plt.xlabel('Observed review score')
plt.ylabel('Inferred paper value')

#%%

value_df = extract_variable(summary_df, 'value_per_paper', axis='index')

plt.plot(paper_df[citation_score], value_df['Mean'], '.')

plt.xlabel('Observed citation score')
plt.ylabel('Inferred paper value')

#%%

citation_ppc_df = extract_variable(summary_df, 'citation_ppc', axis='index')

# plt.errorbar(x=paper_df[citation_score], 
#               y=citation_ppc_df['50%'], 
#               yerr=citation_ppc_df[['5%','95%']].T,
              # fmt='.')
plt.plot(paper_df[citation_score], citation_ppc_df['50%'], '.')
# plt.plot(paper_df['PERCENTILE_CITATIONS'], 100*citation_ppc_df['50%'], '.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Observed citation score')
plt.ylabel('Posterior predicted citation score')

#%% Prior predictive check

use_estimated_priors = False

data = {
    'N_papers': paper_df.shape[0],
    'N_institutions': nuniq(paper_df['new_institution_id']),
    'institution_per_paper': paper_df['new_institution_id'],
    
    'N_review_scores':    0,
    'review_score': [],
    'paper_per_review_score': [],
    
    'N_citation_scores': 0,
    'citation_score': [],
    'paper_per_citation_score': [],
    'citation_percentile_score': 0
    }

if (use_estimated_priors):
  data |= {
      'use_estimated_priors': 1,
  
      # The below are estimated coefficients from other models.
  
      # Coefficient of citation
      'beta_mu': prior_summary_df.loc['beta', 'Mean'],
      'beta_sigma': prior_summary_df.loc['beta', 'StdDev'],
  
      # Standard deviation of citation
      'sigma_cit_mu': prior_summary_df.loc['sigma_cit', 'Mean'],
      'sigma_cit_sigma': prior_summary_df.loc['sigma_cit', 'StdDev'],
  
      # Standard deviation of peer review.
      'sigma_review_mu': prior_summary_df.loc['sigma_review', 'Mean'],
      'sigma_review_sigma': prior_summary_df.loc['sigma_review', 'StdDev'],
      
      'alpha_nonzero_cit_mu': prior_summary_df.loc['alpha_nonzero_cit', 'Mean'],
      'alpha_nonzero_cit_sigma': prior_summary_df.loc['alpha_nonzero_cit', 'StdDev'],
      
      'beta_nonzero_cit_mu': prior_summary_df.loc['beta_nonzero_cit', 'Mean'],
      'beta_nonzero_cit_sigma': prior_summary_df.loc['beta_nonzero_cit', 'StdDev']
  }
else:
  data |= {
      'use_estimated_priors': 0,
  
      # The below are estimated coefficients from other models.
      
      # Coefficient of citation
      'beta_mu': 0,
      'beta_sigma': 1,
  
      # Standard deviation of citation
      'sigma_cit_mu': 0,
      'sigma_cit_sigma': 1,
  
      # Standard deviation of peer review.
      'sigma_review_mu': 0,
      'sigma_review_sigma': 1,
      
      'alpha_nonzero_cit_mu': 0,
      'alpha_nonzero_cit_sigma': 1,
      
      'beta_nonzero_cit_mu': 0,
      'beta_nonzero_cit_sigma': 1
      }
  
prior_check_fit = model.sample(data=data, chains=1, 
                               output_dir = output_dir / 'prior_check')

#%%

print(prior_check_fit.diagnose())

#%%

prior_check_draws_df = prior_check_fit.draws_pd()

prior_check_summary_df = prior_check_fit.summary()

#%%

citation_ppc_df = extract_variable(prior_check_summary_df, 'citation_ppc', axis='index')
plt.hist(citation_ppc_df['Mean'])

#%%

value_per_paper_df = extract_variable(prior_check_summary_df, 'value_per_paper', axis='index')
plt.hist(value_per_paper_df['Mean'])

#%%

sigma_value_inst_df = extract_variable(prior_check_summary_df, 'sigma_value_inst', axis='index')
plt.hist(sigma_value_inst_df['Mean'])