#%%
from cmdstanpy import CmdStanModel
import os
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from common import unique_id, nuniq, extract_variable, group_kfold_partition, data_dir

#%%

output_dir = Path(f'test/')

#%%

stan_file = os.path.abspath('./model.stan')
model = CmdStanModel(stan_file=stan_file)
print(model)

#%%

inst_df = pd.read_csv(data_dir / 'institutional.csv')
metric_df = pd.read_csv(data_dir / 'metrics.csv')

#%%

citation_score = 'ncs'

paper_df = (metric_df
            .query('GEV_id == "6"')\
               [['INSTITUTION_ID',
                 'REV_1_SCORE',
                 'REV_2_SCORE',
                 'ncs',
                 'njs',
                 'PERCENTILE_INDICATOR_VALUE',
                 'PERCENTILE_CITATIONS']]
            .sample(frac=0.1)               
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

paper_df['fold'] = group_kfold_partition(paper_df['new_institution_id'], 5)

#%%

prediction_type = 'review'
fold = 0

train_df = paper_df[paper_df['fold'] != fold]
test_df = paper_df[paper_df['fold'] == fold]

review_df = pd.concat([train_df[['new_paper_id', 'REV_1_SCORE']]\
                        .rename(columns={'REV_1_SCORE': 'review_score'}),
                       train_df[['new_paper_id', 'REV_2_SCORE']]\
                        .rename(columns={'REV_2_SCORE': 'review_score'})],
                    axis=0,
                    ignore_index=True)

if (prediction_type == 'review'):
    review_df = pd.concat([review_df, 
                           test_df[['new_paper_id', 'REV_1_SCORE']]\
                           .rename(columns={'REV_1_SCORE': 'review_score'})],
                          axis=0,
                          ignore_index=True)

review_df = review_df.dropna()

citation_df = (
                train_df[['new_paper_id', citation_score]]
                .rename(columns={citation_score:
                                'citation_score'})
                )

if (prediction_type == 'citation'):
    citation_df = pd.concat([citation_df,
                             (test_df[['new_paper_id', citation_score]]
                              .rename(columns={citation_score: 'citation_score'})
                             )
                            ]
                           )

citation_df = citation_df.dropna()

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
    'citation_percentile_score': 'PERCENTILE' in citation_score,
}

#%% Fit model to data

fit = model.sample(data=data, chains=1, 
                   output_dir = output_dir / f'{prediction_type}' / f'{fold}')

#%%

print(fit.diagnose())

#%%

draws_df = fit.draws_pd()

summary_df = fit.summary()

#%%
sns.set_palette('Set1')

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
ax = plt.subplot(1, 3, 1)
sns.distplot(draws_df['beta'])
plt.xlabel(r'$\beta$')
plt.yticks([])

ax = plt.subplot(1, 3, 2)
sns.distplot(draws_df['alpha_nonzero_cit'], label=r'$\alpha_{0^+}$')
sns.distplot(draws_df['beta_nonzero_cit'], label=r'$\beta_{0^+}$')
plt.legend(loc='best')
plt.yticks([])
plt.ylabel('')

ax = plt.subplot(1, 3, 3)
sns.distplot(draws_df['sigma_review'], label='Review')
sns.distplot(draws_df['sigma_cit'], label='Citation')
plt.xlabel(r'$\sigma$')
plt.yticks([])
plt.ylabel('')
plt.legend(loc='best')

#%%
#paper_id = 385 # High ncs
paper_id = 1 # Low ncs
sns.distplot(draws_df[f'citation_ppc[{paper_id}]'])
plt.axvline(citation_df.query(f'new_paper_id == {paper_id}').iloc[0, 1], color='k')

  #%%

def extract_and_merge(variable):
    review_score_ppc_df = extract_variable(summary_df, 
                                        variable, 
                                        axis='index',
                                        index_dtypes=[int])

    plt_df = pd.merge(review_score_ppc_df.loc[variable,:], paper_df, 
                    left_index=True, right_on='new_paper_id')

    plt_df['type'] = pd.Categorical.from_codes(1*(plt_df['fold'] == fold), 
                                            categories=['Training', 'Test'])

    return plt_df

#%%

plt_df = extract_and_merge('review_score_ppc')

sns.set_palette('Set1')
sns.scatterplot(plt_df, x='REV_2_SCORE', y='Mean', hue='type')

plt.xlabel('Observed review score')
plt.ylabel('Posterior predicted review score')

#%%

plt_df = extract_and_merge('value_per_paper')

sns.set_palette('Set1')
sns.scatterplot(plt_df, x='REV_2_SCORE', y='Mean', hue='type')

plt.xlabel('Observed review score')
plt.ylabel('Inferred paper value')

#%%

plt_df = extract_and_merge('value_per_paper')

sns.set_palette('Set1')
sns.scatterplot(plt_df, x=citation_score, y='Mean', hue='type')

plt.xlabel('Observed citation score')
plt.ylabel('Inferred paper value')

#%%

plt_df = extract_and_merge('citation_ppc')
citation_ppc_df = extract_variable(summary_df, 'citation_ppc', axis='index')

sns.set_palette('Set1')
sns.scatterplot(plt_df, x=citation_score, y='Mean', hue='type')

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

plt.plot(value_per_paper_df['Mean'],
         citation_ppc_df['Mean'],
         '.')
plt.xlabel('Value')
plt.ylabel('Citation')

#%%
paper_id = 2
plt.plot(prior_check_draws_df[f'citation_ppc[{paper_id}]'], 
         prior_check_draws_df[f'value_per_paper[{paper_id}]'], 
         '.')

#%%
sns.kdeplot(prior_check_draws_df[f'citation_ppc[{paper_id}]'])
# %%
