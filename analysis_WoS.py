import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import re
import statsmodels.formula.api as smf
import os
from datetime import date

colors = sns.palettes.mpl_palette('Set1', 8)

def get_conn():
  conn = pyodbc.connect(
              driver='{SQL Server Native Client 11.0}',
              server='p-cwts-010260',
              database='userdb_traagva1',
              trusted_connection='yes',
              unicode_results=False);
  return conn;

gev_names_df = pd.DataFrame(
  [['1', "Mathematics and Computer Sciences"],
  ['2', "Physics"],
  ['3', "Chemistry"],
  ['4', "Earth Sciences"],
  ['5', "Biology"],
  ['6', "Medicine"],
  ['7', "Agricultural and veterinary sciences"],
  ['8b', "Civil Engineering"],
  ['9', "Industrial and Information Engineering"],
  ['11b', "Psychology"],
  ['13', "Economics and Statistics"]], columns=['GEV_id', 'GEV'])

today = date.today().strftime('%Y%m%d')
#%%
get_data_from_SQL = False
if get_data_from_SQL:
    try:
      conn = get_conn()
    except:
      conn = None
    
    if conn:
      sql = """
          SELECT *
            FROM VQR
            LEFT JOIN VQR_WoS_indic AS indic
              ON indic.ut = VQR.ut
        """;
      df = pd.io.sql.read_sql(sql, conn);
      conn.close()
    
      # Randomly pick either reviewer 1 or reviewer 2 as reviewer 1.
      np.random.seed(235)
      df['reviewer'] = np.random.randint(1, 3, size=df.shape[0])
      rev_1_cols = [c for c in df.columns if 'REV_1' in c]
      rev_2_cols = [c for c in df.columns if 'REV_2' in c]
      df[rev_1_cols + rev_2_cols] = df[rev_1_cols + rev_2_cols].where(
                                      df['reviewer'] == 1,
                                      df[rev_2_cols + rev_1_cols].values)
    
      df['GEV_numeric'] = df['GEV'].apply(lambda x: int(re.match(r'\d*', x)[0]))
      df = df.set_index(['INSTITUTION_ID', 'GEV']);
    
      df = df.sort_values('GEV_numeric')
    
      df['REV_1_SCORE'] = df[['REV_1_ORIGINALITY', 'REV_1_RIGOR', 'REV_1_IMPACT']].sum(axis=1, skipna=False)
      df['REV_2_SCORE'] = df[['REV_2_ORIGINALITY', 'REV_2_RIGOR', 'REV_2_IMPACT']].sum(axis=1, skipna=False)
      df['REV_SCORE'] = df[['REV_1_SCORE', 'REV_2_SCORE']].mean(axis=1, skipna=False)
    
      df.to_csv('../data/WoS_indic.csv')
    
#%%
np.random.seed(0)
min_n_per_institution = 5

data_type = 'all' # all | universities

results_dir = '../results/min={min}/{data}/'.format(min=min_n_per_institution, data=data_type)
if not os.path.exists(results_dir):
  os.makedirs(results_dir)

#%% Read data
df = pd.read_csv('../data/WoS_indic.csv')
if data_type == 'universities':
  df = df[df['INSTITUTION_TYPE'] == 'U']
df = pd.merge(gev_names_df, df, left_on='GEV_id', right_on='GEV')
del df['GEV_y']
df = df.rename(columns={'GEV_x': 'GEV'})
df = df.set_index(['INSTITUTION_ID', 'GEV', 'ID_OUTPUT'])

# Remove items with missing review and metrics
df = df[~pd.isna(df['REV_1_SCORE']) & ~pd.isna(df['REV_2_SCORE']) & ~pd.isna(df['ncs'])]
#%%

stratification_df = pd.read_excel('../data/dati_leida_campione.xlsx')
#stratification_df = stratification_df[stratification_df['Typology'] == 'U']
#%%

plt.figure(figsize=(4, 3))
bins = np.linspace(0, 15, 16)
plt.hist(stratification_df['WoS_Universe'],
         bins=bins, color=colors[1],histtype='stepfilled')
plt.grid(axis='y', ls=':')
plt.ylabel('Frequency')
plt.xlabel('Sample %')
sns.despine()
plt.savefig(os.path.join(results_dir, 'stratification.pdf'), bbox_inches='tight')
#%%
missing_uni_df = stratification_df.query('WoS_Leiden_sample == 0')
#%%
cols = ['cs', 'ncs', 'p_top_prop',
        'CITATIONS_NUMBER', 'PERCENTILE_CITATIONS',
        'js', 'njs', 'jpp_top_prop',
        'INDICATOR_VALUE', 'PERCENTILE_INDICATOR_VALUE',
        'REV_1_SCORE', 'REV_2_SCORE', 'REV_SCORE'];

#%% Correlation at individual levels (overall)
corr_individual = df[cols].corr('spearman')
#%% Calculate outcome results on institutional level  (overall)
inst_df = df[cols].groupby(level=['INSTITUTION_ID', 'GEV'], sort=False).mean().reset_index()

#%% #%% Sample size per GEV per institutions
n_per_institution = df.groupby(level=['INSTITUTION_ID', 'GEV'], sort=False).size()
n_per_institution.name = 'n_pubs'
n_per_institution.to_excel('../results/number_of_publications.xlsx')
#%% Scatter plot at individual level (njs individual level)
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(4*2, 3*2))
for idx, (gev, gev_df) in enumerate(df.groupby(level='GEV', sort=False)):
  row = int(idx / 4)
  col = idx % 4
  ax = axs[row][col]
  ax.plot(gev_df['REV_1_SCORE'], gev_df['ncs'], '.', alpha=0.5, mew=0)

  ax.set_yscale('log')
  if (row == 2):
      if (col == 0):
          ax.set_xticks([0, 10, 20, 30])
      else:
          ax.set_xticks([10, 20, 30])
      ax.tick_params(axis='x', which='both', direction='in', left=True, bottom=True)
  else:
      ax.set_xticks([])

  if (col == 0):
      if (row == 2):
          ax.set_yticks([0.1, 10])
      else:
          ax.set_yticks([10])
      ax.tick_params(axis='y', which='both', direction='in', left=True, bottom=True)
  else:
      ax.set_yticks([])

  ax.set_xlim(0, 35)
  ax.set_ylim(0.1, 20)
  ax.text(0.05, 0.9, '{0}'.format(gev[:30]), transform=ax.transAxes, ha='left', va='baseline')

axs[2][3].set_xticks([])
axs[2][3].set_yticks([])

fig.subplots_adjust(hspace=0,wspace=0)

fig.text(0.5, 0.05, 'Reviewer 1 score', ha='center')
fig.text(0.05, 0.5, 'NCS', va='center', rotation='vertical')
fig.savefig(os.path.join(results_dir, 'scatter_pub_ncs.pdf'))
#%% Scatter plot at individual level (reviewer individual level)
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(4*2, 3*2))
for idx, (gev, gev_df) in enumerate(df.groupby(level='GEV', sort=False)):
  row = int(idx / 4)
  col = idx % 4
  ax = axs[row][col]
  ax.plot(gev_df['REV_1_SCORE'], gev_df['REV_2_SCORE'], '.', alpha=0.5, mew=0)

  if (row == 2):
      if (col == 0):
          ax.set_xticks([0, 10, 20, 30])
      else:
          ax.set_xticks([10, 20, 30])
      ax.tick_params(axis='x', which='both', direction='in', left=True, bottom=True)
  else:
      ax.set_xticks([])

  if (col == 0):
      if (row == 2):
          ax.set_yticks([0, 10, 20, 30])
      else:
          ax.set_yticks([10, 20, 30])
      ax.tick_params(axis='y', which='both', direction='in', left=True, bottom=True)
  else:
      ax.set_yticks([])

  ax.set_xlim(0, 35)
  ax.set_ylim(0, 35)
  ax.text(0.05, 0.9, '{0}'.format(gev), transform=ax.transAxes, ha='left', va='baseline')

axs[2][3].set_xticks([])
axs[2][3].set_yticks([])

fig.subplots_adjust(hspace=0,wspace=0)

fig.text(0.5, 0.05, 'Reviewer 1 score', ha='center')
fig.text(0.05, 0.5, 'Reviewer 2 score', va='center', rotation='vertical')
fig.savefig(os.path.join(results_dir, 'scatter_pub_reviewer_uncertainty.pdf'))

#%% Calculate MAD and MAPD on institutional level
def calc_MAD_and_MAPD_inst(df):

  inst_df = df.groupby(['INSTITUTION_ID', 'GEV_id', 'GEV_numeric'])\
                      .aggregate({'REV_1_SCORE': 'mean',
                                  'REV_2_SCORE': 'mean',
                                  'ncs': 'mean',
                                  'njs': 'mean',
                                  'PERCENTILE_CITATIONS': 'mean',
                                  'PERCENTILE_INDICATOR_VALUE': 'mean',
                                  'n_pubs': 'mean'})

  MADs = []
  MAPDs = []
  for GEV_id, GEV_df in inst_df.groupby('GEV_id'):

    res_ncs = smf.ols(formula='REV_1_SCORE ~ ncs', data=GEV_df).fit()
    res_njs = smf.ols(formula='REV_1_SCORE ~ njs', data=GEV_df).fit()

    if GEV_id != '13':
      res_perc_cit = smf.ols(formula='REV_1_SCORE ~ PERCENTILE_CITATIONS', data=GEV_df).fit()
      res_perc_ind = smf.ols(formula='REV_1_SCORE ~ PERCENTILE_INDICATOR_VALUE', data=GEV_df).fit()

    GEV_df['pred_ncs'] = res_ncs.predict()
    GEV_df['pred_njs'] = res_njs.predict()
    if GEV_id != '13':
      GEV_df['pred_perc_cit'] = res_perc_cit.predict(GEV_df['PERCENTILE_CITATIONS']) # To account for NA
      GEV_df['pred_perc_ind'] = res_perc_ind.predict(GEV_df['PERCENTILE_INDICATOR_VALUE']) # To account for NA
    else:
      GEV_df['pred_perc_cit'] = np.nan
      GEV_df['pred_perc_ind'] = np.nan

    GEV_df['abs_diff_ncs'] = np.abs(GEV_df['pred_ncs'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_njs'] = np.abs(GEV_df['pred_njs'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_perc_cit'] = np.abs(GEV_df['pred_perc_cit'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_perc_ind'] = np.abs(GEV_df['pred_perc_ind'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_rev_2'] = np.abs(GEV_df['REV_2_SCORE'] - GEV_df['REV_1_SCORE'])

    MAD = GEV_df.median()[['abs_diff_ncs', 'abs_diff_njs', 'abs_diff_perc_cit', 'abs_diff_perc_ind', 'abs_diff_rev_2']]
    MAD['GEV_id'] = GEV_id
    MADs.append(MAD)

    cols = ['REV_1_SCORE', 'REV_2_SCORE', 'ncs', 'njs',
            'pred_ncs', 'pred_njs', 'pred_perc_cit', 'pred_perc_ind']

    GEV_df = GEV_df[cols + ['n_pubs']].copy()
    GEV_df[cols] = GEV_df[cols].mul(GEV_df['n_pubs'], axis=0)

    def abs_perc_diff(x, y):
      d = np.abs(x - y)/y
      d[pd.isna(d)] = 0
      return d

    GEV_df['abs_perc_diff_ncs'] = abs_perc_diff(GEV_df['pred_ncs'], GEV_df['REV_1_SCORE'])
    GEV_df['abs_perc_diff_njs'] = abs_perc_diff(GEV_df['pred_njs'], GEV_df['REV_1_SCORE'])
    GEV_df['abs_perc_diff_perc_cit'] = abs_perc_diff(GEV_df['pred_perc_cit'], GEV_df['REV_1_SCORE'])
    GEV_df['abs_perc_diff_perc_ind'] = abs_perc_diff(GEV_df['pred_perc_ind'], GEV_df['REV_1_SCORE'])
    GEV_df['abs_perc_diff_rev_2'] = abs_perc_diff(GEV_df['REV_2_SCORE'], GEV_df['REV_1_SCORE'])

    MAPD = 100*GEV_df.median()[['abs_perc_diff_ncs', 'abs_perc_diff_njs', 'abs_perc_diff_perc_cit', 'abs_perc_diff_perc_ind', 'abs_perc_diff_rev_2']]
    MAPD['GEV_id'] = GEV_id
    MAPDs.append(MAPD)

  MAD = pd.concat(MADs, axis=1).T.set_index('GEV_id')
  MAPD = pd.concat(MAPDs, axis=1).T.set_index('GEV_id')
  return MAD, MAPD

# For debugging purposes we make it a global variable
bootstrap_df = None

def bootstrap_MAD_MAPD_inst(df, N):
  n = df.shape[0]
  global bootstrap_df
  for _ in range(N):
    bootstrap_df = df.iloc[np.random.randint(n, size=n)]
    yield(calc_MAD_and_MAPD_inst(bootstrap_df))
#%%
limited_df = pd.merge(df.reset_index(), n_per_institution[n_per_institution >= min_n_per_institution].reset_index(), on=['INSTITUTION_ID', 'GEV'])

#%%

inst_gev_df = limited_df.groupby(['INSTITUTION_ID', 'GEV_id', 'GEV_numeric'])\
                    .aggregate({'REV_1_SCORE': 'mean',
                                'REV_2_SCORE': 'mean',
                                'ncs': 'mean',
                                'njs': 'mean',
                                'PERCENTILE_CITATIONS': 'mean',
                                'PERCENTILE_INDICATOR_VALUE': 'mean',
                                'n_pubs': 'mean'})

def percentile(n):
  def percentile_(x):
    if not isinstance(x,pd.Series):
      raise ValueError('need Series argument')
    return np.percentile(x, n)
  percentile_.__name__ = 'percentile_%s' % n
  return percentile_

GEV_df = inst_gev_df.groupby('GEV_id').aggregate([percentile(2.5), percentile(50), percentile(97.5)])
GEV_df = pd.merge(gev_names_df, GEV_df, left_on='GEV_id', right_index=True)
#%%
x = np.arange(GEV_df.shape[0])
y = GEV_df[('n_pubs', 'percentile_50')]
ybounds = GEV_df[[('n_pubs', 'percentile_2.5'), ('n_pubs', 'percentile_97.5')]]

plt.bar(x, y)
plt.errorbar(x=x, y=y, fmt='none', lw=1,
             yerr=np.abs(ybounds.subtract(y, axis='index').values.T),
             alpha=0.3)
plt.ylim(0, 30)
plt.xticks(x, GEV_df['GEV'], rotation=45, ha='right')
plt.savefig(os.path.join(results_dir, 'field_n_pubs.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(results_dir, 'field_n_pubs.png'), dpi=300, bbox_inches='tight')

#%%
MAD_MAPD_bootstrap = list(bootstrap_MAD_MAPD_inst(limited_df, N=1000))
#%%
MAD_bootstrap, MAPD_bootstrap = zip(*MAD_MAPD_bootstrap)
#%%
MAD, MAPD = calc_MAD_and_MAPD_inst(limited_df)
#%%
conf_interval = 0.95
low_bound = pd.concat(MAD_bootstrap).groupby('GEV_id').aggregate([lambda x: np.percentile(x, 100*(1 - conf_interval)/2)])
low_bound.columns.set_levels(['lb'], level=1, inplace=True)
up_bound = pd.concat(MAD_bootstrap).groupby('GEV_id').aggregate([lambda x: np.percentile(x, 100*((1 - conf_interval)/2 + conf_interval))])
up_bound.columns.set_levels(['ub'], level=1, inplace=True)
MAD.columns = pd.MultiIndex.from_tuples([(c, 'empirical') for c in MAD.columns])
MAD = pd.concat([MAD, low_bound, up_bound], axis=1)
MAD = MAD.sort_index(axis=1)
MAD = pd.merge(gev_names_df, MAD, left_on='GEV_id', right_index=True)
#%%

low_bound = pd.concat(MAPD_bootstrap).groupby('GEV_id').aggregate([lambda x: np.percentile(x, 100*(1 - conf_interval)/2)])
low_bound.columns.set_levels(['lb'], level=1, inplace=True)
up_bound = pd.concat(MAPD_bootstrap).groupby('GEV_id').aggregate([lambda x: np.percentile(x, 100*((1 - conf_interval)/2 + conf_interval))])
up_bound.columns.set_levels(['ub'], level=1, inplace=True)
MAPD.columns = pd.MultiIndex.from_tuples([(c, 'empirical') for c in MAPD.columns])
MAPD = pd.concat([MAPD, low_bound, up_bound], axis=1)
MAPD = MAPD.sort_index(axis=1)
MAPD = pd.merge(gev_names_df, MAPD, left_on='GEV_id', right_index=True)
#%%

MAD.to_csv(os.path.join(results_dir, 'MAD.csv'))
MAPD.to_csv(os.path.join(results_dir, 'MAPD.csv'))

#%%

sns.set_style('white')
sns.set_palette('Set1')
plt.figure(figsize=(9, 4))
xlabels = MAD['GEV']
x = np.arange(len(xlabels))
ycols = [('abs_diff_ncs', 'NCS'),
         ('abs_diff_njs', 'NJS'),
         ('abs_diff_perc_cit', 'Perc. Cit'),
         ('abs_diff_perc_ind', 'Perc. Journal'),
         ('abs_diff_rev_2', 'Review')]
gap = 0.01
group_gap = 0.3
width = (1 - group_gap - gap*(len(ycols) - 1))/len(ycols)
x = x - (len(ycols)*width + (len(ycols) - 1)*gap)/2 + width/2
bars = []
for idx, (ycol, ylabel) in enumerate(ycols):
  y = MAD[(ycol, 'empirical')]
  ybounds = MAD[[(ycol, 'lb'), (ycol, 'ub')]]
  b = plt.bar(x=x, height=y, color=colors[idx], alpha=0.5, width=width, label=ylabel)
  bars.append(b)
  plt.errorbar(x=x, y=y, fmt='none', capsize=2,
               yerr=np.abs(ybounds.subtract(y, axis='index').values.T),
               color=colors[idx])
  x += width + gap
plt.xticks(np.arange(len(xlabels)), xlabels, rotation='vertical')
#plt.xlabel('GEV')
plt.ylabel('MAD')
plt.legend(bars, list(zip(*ycols))[1],
           bbox_to_anchor=(0.5, 1.02), loc='center',
           ncol=5)
plt.grid(axis='y', ls=':')
sns.despine()
plt.savefig(os.path.join(results_dir, 'MAD.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(results_dir, 'MAD.png'), dpi=300, bbox_inches='tight')

#%%

sns.set_style('white')
sns.set_palette('Set1')
plt.figure(figsize=(9, 4))
xlabels = MAPD['GEV']
x = np.arange(len(xlabels))
ycols = [('abs_perc_diff_ncs', 'NCS'),
         ('abs_perc_diff_njs', 'NJS'),
         ('abs_perc_diff_perc_cit', 'Perc. Cit'),
         ('abs_perc_diff_perc_ind', 'Perc. Journal'),
         ('abs_perc_diff_rev_2', 'Review')]
gap = 0.01
group_gap = 0.3
width = (1 - group_gap - gap*(len(ycols) - 1))/len(ycols)
x = x - (len(ycols)*width + (len(ycols) - 1)*gap)/2 + width/2
bars = []
for idx, (ycol, ylabel) in enumerate(ycols):
  y = MAPD[(ycol, 'empirical')]
  ybounds = MAPD[[(ycol, 'lb'), (ycol, 'ub')]]
  b = plt.bar(x=x, height=y, color=colors[idx], alpha=0.5, width=width, label=ylabel)
  bars.append(b)
  plt.errorbar(x=x, y=y, fmt='none', capsize=2,
               yerr=np.abs(ybounds.subtract(y, axis='index').values.T),
               color=colors[idx])
  x += width + gap
plt.xticks(np.arange(len(xlabels)), xlabels, rotation='vertical')
#plt.xlabel('GEV')
plt.ylabel('MAPD')
plt.legend(bars, list(zip(*ycols))[1],
           bbox_to_anchor=(0.5, 1.02), loc='center',
           ncol=5)
plt.grid(axis='y', ls=':')
sns.despine()
plt.savefig(os.path.join(results_dir, 'MAPD.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(results_dir, 'MAPD.png'), dpi=300, bbox_inches='tight')
#%% Calculate MAD on individual level
def calc_MAD_ind(df):

  ##%%
  MADs = []
  for GEV_id, GEV_df in df.groupby('GEV_id'):
    ##%%
    res_ncs = smf.ols(formula='REV_1_SCORE ~ ncs', data=GEV_df).fit()
    res_njs = smf.ols(formula='REV_1_SCORE ~ njs', data=GEV_df).fit()
    if GEV_id != '13':
      res_perc_cit = smf.ols(formula='REV_1_SCORE ~ PERCENTILE_CITATIONS', data=GEV_df).fit()
      res_perc_ind = smf.ols(formula='REV_1_SCORE ~ PERCENTILE_INDICATOR_VALUE', data=GEV_df).fit()

    GEV_df['pred_ncs'] = res_ncs.predict()
    GEV_df['pred_njs'] = res_njs.predict()
    if GEV_id != '13':
      GEV_df['pred_perc_cit'] = res_perc_cit.predict(GEV_df['PERCENTILE_CITATIONS']) # To account for NA
      GEV_df['pred_perc_ind'] = res_perc_ind.predict(GEV_df['PERCENTILE_INDICATOR_VALUE']) # To account for NA
    else:
      GEV_df['pred_perc_cit'] = np.nan
      GEV_df['pred_perc_ind'] = np.nan

    GEV_df['abs_diff_ncs'] = np.abs(GEV_df['pred_ncs'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_njs'] = np.abs(GEV_df['pred_njs'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_perc_cit'] = np.abs(GEV_df['pred_perc_cit'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_perc_ind'] = np.abs(GEV_df['pred_perc_ind'] - GEV_df['REV_1_SCORE'])
    GEV_df['abs_diff_rev_2'] = np.abs(GEV_df['REV_2_SCORE'] - GEV_df['REV_1_SCORE'])

    MAD = GEV_df.median()[['abs_diff_ncs', 'abs_diff_njs', 'abs_diff_perc_cit', 'abs_diff_perc_ind', 'abs_diff_rev_2']]
    MAD['GEV_id'] = GEV_id
    MADs.append(MAD)
  ##%%
  MAD = pd.concat(MADs, axis=1).T.set_index('GEV_id')
  return MAD

def bootstrap_MAD_ind(df, N):
  n = df.shape[0]
  for _ in range(N):
    bootstrap_df = df.iloc[np.random.randint(n, size=n)]
    yield(calc_MAD_ind(bootstrap_df))

#%%
MAD_ind_bootstrap = list(bootstrap_MAD_ind(df, N=1000))
#%%
MAD_ind = calc_MAD_ind(df)
#%%
conf_interval = 0.95
low_bound = pd.concat(MAD_ind_bootstrap).groupby('GEV_id').aggregate([lambda x: np.percentile(x, 1 - conf_interval/2)])
low_bound.columns.set_levels(['lb'], level=1, inplace=True)
up_bound = pd.concat(MAD_ind_bootstrap).groupby('GEV_id').aggregate([lambda x: np.percentile(x, (1 - conf_interval/2) + conf_interval)])
up_bound.columns.set_levels(['ub'], level=1, inplace=True)
MAD_ind.columns = pd.MultiIndex.from_tuples([(c, 'empirical') for c in MAD_ind.columns])
MAD_ind = pd.concat([MAD_ind, low_bound, up_bound], axis=1)
MAD_ind = MAD_ind.sort_index(axis=1)
MAD_ind = pd.merge(gev_names_df, MAD_ind, left_on='GEV_id', right_index=True)

#%%
MAD_ind.to_csv(os.path.join(results_dir, 'MAD_ind.csv'))

#%%

sns.set_style('white')
sns.set_palette('Set1')
plt.figure(figsize=(9, 4))
xlabels = MAD_ind['GEV']
x = np.arange(len(xlabels))
ycols = [('abs_diff_ncs', 'NCS'),
         ('abs_diff_njs', 'NJS'),
         ('abs_diff_perc_cit', 'Perc. Cit'),
         ('abs_diff_perc_ind', 'Perc. Journal'),
         ('abs_diff_rev_2', 'Review')]
gap = 0.01
group_gap = 0.3
width = (1 - group_gap - gap*(len(ycols) - 1))/len(ycols)
x = x - (len(ycols)*width + (len(ycols) - 1)*gap)/2 + width/2
bars = []
for idx, (ycol, ylabel) in enumerate(ycols):
  y = MAD_ind[(ycol, 'empirical')]
  ybounds = MAD_ind[[(ycol, 'lb'), (ycol, 'ub')]]
  b = plt.bar(x=x, height=y, color=colors[idx], alpha=0.5, width=width, label=ylabel)
  bars.append(b)
  plt.errorbar(x=x, y=y, fmt='none', capsize=2,
               yerr=np.abs(ybounds.subtract(y, axis='index').values.T),
               color=colors[idx])
  x += width + gap
plt.xticks(np.arange(len(xlabels)), xlabels, rotation='vertical')
#plt.xlabel('GEV')
plt.ylabel('MAD')
plt.legend(bars, list(zip(*ycols))[1],
           bbox_to_anchor=(0.5, 1.02), loc='center',
           ncol=5)
plt.grid(axis='y', ls=':')
sns.despine()
plt.savefig(os.path.join(results_dir, 'MAD_ind.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(results_dir, 'MAD_ind.png'), dpi=300, bbox_inches='tight')

#%% Scatter plot at institutional level (ncs)
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(4*2, 3*2))
for idx, (gev, gev_df) in enumerate(inst_df.groupby('GEV', sort=False)):
  row = int(idx / 4)
  col = idx % 4
  ax = axs[row][col]
  ax.plot(gev_df['REV_1_SCORE'], gev_df['ncs'], '.', alpha=0.5, mew=0)

  ax.set_yscale('log')
  if (row == 2):
      if (col == 0):
          ax.set_xticks([0, 10, 20, 30])
      else:
          ax.set_xticks([10, 20, 30])
      ax.tick_params(axis='x', which='both', direction='in', left=True, bottom=True)
  else:
      ax.set_xticks([])

  ax.set_xlim(0, 35)
  ax.set_ylim(0.02, 200)

  if (col == 0):
      ax.set_yticks([1e-1, 1e0, 1e1, 1e2])
      ax.tick_params(axis='y', which='both', direction='in', left=True, bottom=True)
  else:
      ax.set_yticks([])

  ax.text(0.05, 0.9, '{0}'.format(gev[:30]), transform=ax.transAxes, ha='left', va='baseline')

ax = axs[2][3]
ax.set_xlim(0, 35)
ax.set_xticks([10, 20, 30])
ax.set_yticks([])

fig.subplots_adjust(hspace=0,wspace=0)

fig.text(0.5, 0.05, 'Reviewer 1 score', ha='center')
fig.text(0.05, 0.5, 'NCS', va='center', rotation='vertical')
fig.savefig(os.path.join(results_dir, 'scatter_inst_ncs.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(results_dir, 'scatter_inst_ncs.png'), dpi=300, bbox_inches='tight')
#%% Scatter plot at institutional level (ncs)
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(4*2, 3*2))
for idx, (gev, gev_df) in enumerate(inst_df.groupby('GEV', sort=False)):
  row = int(idx / 4)
  col = idx % 4
  ax = axs[row][col]
  ax.plot(gev_df['REV_1_SCORE'], gev_df['REV_2_SCORE'], '.', alpha=0.5, mew=0)

  if (row == 2):
      if (col == 0):
          ax.set_xticks([0, 10, 20, 30])
      else:
          ax.set_xticks([10, 20, 30])
      ax.tick_params(axis='x', which='both', direction='in', left=True, bottom=True)
  else:
      ax.set_xticks([])

  ax.set_xlim(0, 35)
  ax.set_ylim(0, 35)

  if (col == 0):
      if (row == 2):
          ax.set_yticks([0, 10, 20, 30])
      else:
          ax.set_yticks([10, 20, 30])
      ax.tick_params(axis='y', which='both', direction='in', left=True, bottom=True)
  else:
      ax.set_yticks([])

  ax.text(0.05, 0.9, '{0}'.format(gev[:30]), transform=ax.transAxes, ha='left', va='baseline')

ax = axs[2][3]
ax.set_xlim(0, 35)
ax.set_xticks([10, 20, 30])
ax.set_yticks([])

fig.subplots_adjust(hspace=0,wspace=0)

fig.text(0.5, 0.05, 'Reviewer 1 score', ha='center')
fig.text(0.05, 0.5, 'Reviewer 2 score', va='center', rotation='vertical')
fig.savefig(os.path.join(results_dir, 'scatter_inst_reviewer_uncertainty.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(results_dir, 'scatter_inst_reviewer_uncertainty.png'), dpi=300, bbox_inches='tight')
