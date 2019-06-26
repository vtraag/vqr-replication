import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import re

def get_conn():
  conn = pyodbc.connect(
              driver='{SQL Server Native Client 11.0}',
              server='SPCWTDBS02',
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

#%%
sql = """  
    SELECT *
      FROM VQR
      INNER JOIN VQR_WoS_indic AS indic
        ON indic.ut = VQR.ut
  """;        
df = pd.io.sql.read_sql(sql, get_conn());

# Randomly pick either reviewer 1 or reviewer 2 as reviewer 1.
np.random.seed(0)
df['reviewer'] = np.random.randint(1, 3, size=df.shape[0])
rev_1_cols = [c for c in df.columns if 'REV_1' in c]
rev_2_cols = [c for c in df.columns if 'REV_2' in c]
df[rev_1_cols + rev_2_cols] = df[rev_1_cols + rev_2_cols].where(
                                df['reviewer'] == 1, 
                                df[rev_1_cols + rev_2_cols]);

# Limit to universities
df = df[df['INSTITUTION_TYPE'] == 'U']

# Limit to only WoS publications
#df = df[df['USED_DB'] == 'WoS']

df['GEV_numeric'] = df['GEV'].apply(lambda x: int(re.match(r'\d*', x)[0]))
df = df.set_index(['INSTITUTION_ID', 'GEV']);

df = df.sort_values('GEV_numeric')
#%%

df['REV_1_SCORE'] = df[['REV_1_ORIGINALITY', 'REV_1_RIGOR', 'REV_1_IMPACT']].sum(axis=1)
df['REV_2_SCORE'] = df[['REV_2_ORIGINALITY', 'REV_2_RIGOR', 'REV_2_IMPACT']].sum(axis=1)
df['REV_SCORE'] = df[['REV_1_SCORE', 'REV_2_SCORE']].mean(axis=1)
#%%
cols = ['cs', 'ncs', 'p_top_prop', 
        'CITATIONS_NUMBER', 'PERCENTILE_CITATIONS',         
        'js', 'njs', 'jpp_top_prop',
        'INDICATOR_VALUE', 'PERCENTILE_INDICATOR_VALUE',
        'REV_1_SCORE', 'REV_2_SCORE', 'REV_SCORE'];

#%% Correlation at individual levels (overall)
corr_individual = df[cols].corr('spearman')
#%% Calculate outcome results on institutional level  (overall)
inst_df = df[cols].groupby(level=[0, 1]).mean()
corr_institutional = inst_df.corr('spearman')
#%% Sample size per GEV
n = df.groupby(level=['GEV'], sort=False).size()
#%% #%% Sample size per GEV per institutions
n_per_institution = df.groupby(level=['INSTITUTION_ID', 'GEV'], sort=False).size()
n_per_institution.name = 'n_pubs'
n_per_institution.to_excel('../results/number_of_publications.xlsx')
#%% Correlations on individual level, per GEV.
cols = ['ncs', 'p_top_prop', 'PERCENTILE_CITATIONS', 'njs', 'jpp_top_prop', 'PERCENTILE_INDICATOR_VALUE', 'REV_1_SCORE', 'REV_2_SCORE']
corr_individual_per_gev = df.groupby(level=['GEV'])[cols].corr('spearman')
corr_individual_per_gev = corr_individual_per_gev.unstack(level=1)
corr_individual_per_gev = corr_individual_per_gev['REV_1_SCORE']
corr_individual_per_gev = corr_individual_per_gev[[col for col in cols if col != 'REV_1_SCORE']]

corr_individual_per_gev = pd.merge(corr_individual_per_gev, gev_names_df, left_index=True, right_on='GEV_id')
corr_individual_per_gev['GEV_numeric'] = [int(''.join(filter(str.isdigit, g))) for g in corr_individual_per_gev['GEV_id']]
corr_individual_per_gev = corr_individual_per_gev.sort_values('GEV_numeric')
#%% Plot correlations
sns.set_style('whitegrid')
sns.set_palette('Set1')

x = np.arange(corr_individual_per_gev.shape[0])
width = 1.0/3 - 0.1;
plt.figure(figsize=(15, 9))
plt.bar(x,          corr_individual_per_gev['ncs'], width=width, label='WoS Citation Score')
plt.bar(x+width,    corr_individual_per_gev['njs'], width=width, label='WoS Journal Score')
plt.bar(x+2*width,  corr_individual_per_gev['REV_2_SCORE'], width=width, label='Reviewer agreement')

from textwrap import wrap

plt.xticks(np.arange(corr_individual_per_gev.shape[0]) + width, ['\n'.join(wrap(g, 12)) for g in corr_individual_per_gev['GEV']])

sns.despine(top=True, right=True, bottom=True, left=False)
plt.axhline(0, color='k')
plt.ylim(-0.1, 0.7)
plt.ylabel('Spearman Correlation')

plt.legend(loc='upper right')

plt.savefig('../results/art_corr_wos.png', dpi=300, transparent=True)

#%% Scatter plot at individual level (njs individual level)
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(4*2, 3*2))
for idx, (gev, gev_df) in enumerate(df.groupby(level='GEV', sort=False)):
  row = int(idx / 4)
  col = idx % 4
  ax = axs[row][col]
  ax.plot(gev_df['REV_1_SCORE'], gev_df['njs'], '.', alpha=0.5, mew=0)
  
  ax.set_yscale('log')  
  if (row == 2):
      if (col == 0):
          ax.set_xticks([0, 30])
      else:
          ax.set_xticks([30])
  else:
      ax.set_xticks([])

  if (col == 0):
      if (row == 2):
          ax.set_yticks([0.1, 10])
      else:
          ax.set_yticks([10])
  else:
      ax.set_yticks([])
  
  ax.set_xlim(0, 30)
  ax.set_ylim(0.1, 20)
  ax.text(0.05, 0.9, '{0}'.format(gev), transform=ax.transAxes, ha='left', va='baseline')  
    
axs[2][3].set_xticks([])
axs[2][3].set_yticks([])
  
fig.subplots_adjust(hspace=0,wspace=0)

fig.text(0.5, 0.05, 'Reviewer 1 score', ha='center')
fig.text(0.05, 0.5, 'NJS', va='center', rotation='vertical')
fig.savefig('../results/scatter_pub_njs.pdf')
#%% Scatter plot at individual level (reviewer individual level)
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(4*2, 3*2))
for idx, (gev, gev_df) in enumerate(df.groupby(level='GEV', sort=False)):
  row = int(idx / 4)
  col = idx % 4
  ax = axs[row][col]
  ax.plot(gev_df['REV_1_SCORE'], gev_df['REV_2_SCORE'], '.', alpha=0.5, mew=0)
    
  if (row == 2):
      if (col == 0):
          ax.set_xticks([0, 30])
      else:
          ax.set_xticks([30])
  else:
      ax.set_xticks([])

  if (col == 0):
      if (row == 2):
          ax.set_yticks([0, 30])
      else:
          ax.set_yticks([30])
  else:
      ax.set_yticks([])
  
  ax.set_xlim(0, 30)
  ax.set_ylim(0, 30)
  ax.text(0.05, 0.9, '{0}'.format(gev), transform=ax.transAxes, ha='left', va='baseline')  
    
axs[2][3].set_xticks([])
axs[2][3].set_yticks([])
  
fig.subplots_adjust(hspace=0,wspace=0)

fig.text(0.5, 0.05, 'Reviewer 1 score', ha='center')
fig.text(0.05, 0.5, 'Reviewer 2 score', va='center', rotation='vertical')
fig.savefig('../results/scatter_pub_reviewer_uncertainty.pdf')

#%% #%% Sample size per GEV per institutions
# Limit to only those institutions that have at least x publications in a GEV
limited_df = pd.merge(df, n_per_institution[n_per_institution >= 1].reset_index(), left_index=True, right_on=['INSTITUTION_ID', 'GEV'])
inst_df = limited_df.groupby(['INSTITUTION_ID', 'GEV', 'GEV_numeric']).mean()
inst_df = inst_df.sort_index(level='GEV_numeric')
corr_institution_per_gev = inst_df.groupby(level=['GEV'])[cols].corr('spearman')
corr_institution_per_gev = corr_institution_per_gev.unstack(level=1)
corr_institution_per_gev = corr_institution_per_gev['REV_1_SCORE']
corr_institution_per_gev = corr_institution_per_gev[[col for col in cols if col != 'REV_1_SCORE']]

corr_institution_per_gev = pd.merge(corr_institution_per_gev, gev_names_df, left_index=True, right_on='GEV_id')
corr_institution_per_gev['GEV_numeric'] = [int(''.join(filter(str.isdigit, g))) for g in corr_institution_per_gev['GEV_id']]
corr_institution_per_gev = corr_institution_per_gev.sort_values('GEV_numeric')

#%% Scatter plot of peer review

sns.set_style('white')
sns.set_palette('Set1')

plt.figure(figsize=(15, 9))
plt.scatter(inst_df['REV_1_SCORE'], inst_df['REV_2_SCORE'], marker='.')

plt.xlabel('Reviewer 1')
plt.ylabel('Reviewer 2')

sns.despine()

plt.savefig('../results/scatter_inst.png', dpi=300, transparent=True)

#%% Plot correlations
sns.set_style('whitegrid')
sns.set_palette('Set1')

x = np.arange(corr_institution_per_gev.shape[0])
width = 1.0/3 - 0.1;
plt.figure(figsize=(15, 9))
plt.bar(x,          corr_institution_per_gev['ncs'], width=width, label='WoS Citation Score')
plt.bar(x+width,    corr_institution_per_gev['njs'], width=width, label='WoS Journal Score')
plt.bar(x+2*width,  corr_institution_per_gev['REV_2_SCORE'], width=width, label='Reviewer agreement')

from textwrap import wrap

plt.xticks(np.arange(corr_institution_per_gev.shape[0]) + width, ['\n'.join(wrap(g, 12)) for g in corr_institution_per_gev['GEV']])

sns.despine(top=True, right=True, bottom=True, left=False)
plt.axhline(0, color='k')
plt.ylim(-0.1, 0.7)
plt.ylabel('Spearman Correlation')

plt.legend(loc='upper right')

plt.savefig('../results/inst_corr_wos.png', dpi=300, transparent=True)

#%% Plot correlations
sns.set_style('whitegrid')
sns.set_palette('Set1')

x = np.arange(corr_institution_per_gev.shape[0])
width = 1.0/3 - 0.1;
plt.figure(figsize=(15, 9))
plt.bar(x,          corr_institution_per_gev['PERCENTILE_CITATIONS'], width=width, label='VQR Citation Percentile')
plt.bar(x+width,    corr_institution_per_gev['PERCENTILE_INDICATOR_VALUE'], width=width, label='VQR Journal Percentile')
plt.bar(x+2*width,  corr_institution_per_gev['REV_2_SCORE'], width=width, label='Reviewer agreement')

from textwrap import wrap

plt.xticks(np.arange(corr_institution_per_gev.shape[0]) + width, ['\n'.join(wrap(g, 12)) for g in corr_institution_per_gev['GEV']])

sns.despine(top=True, right=True, bottom=True, left=False)
plt.axhline(0, color='k')
plt.ylim(-0.1, 0.7)
plt.legend(loc='upper right')
plt.ylabel('Spearman Correlation')

plt.savefig('../results/inst_corr_vqr.png', dpi=300, transparent=True)

#%% Calculate MAP and MAPD

limited_df = pd.merge(df, n_per_institution[n_per_institution >= 1].reset_index(), left_index=True, right_on=['INSTITUTION_ID', 'GEV'])

limited_df = pd.get_dummies(limited_df, prefix=['REV_1', 'REV_2'],
                            columns=['REV_1_CLASS', 'REV_2_CLASS'], dtype=np.float)
#%%
total_top_pubs = limited_df.groupby('GEV').sum()[['REV_1_Excellent', 'p_top_prop']]

p_top_per_excellent = total_top_pubs['REV_1_Excellent'] / total_top_pubs['p_top_prop']
p_top_per_excellent.name = 'p_top_per_excellent'

#%%
inst_df = limited_df.groupby(['INSTITUTION_ID', 'GEV', 'GEV_numeric']).sum()
inst_df = inst_df.sort_index(level='GEV_numeric')
inst_df = inst_df.join(p_top_per_excellent, on='GEV').reset_index()
inst_df['predicted_Excellent'] = inst_df['p_top_per_excellent']*inst_df['p_top_prop']

inst_df['predicted_APD'] = np.abs(inst_df['predicted_Excellent'] - inst_df['REV_1_Excellent'])/inst_df['REV_1_Excellent']
inst_df['predicted_APD'] = inst_df['predicted_APD'].fillna(0)

inst_df['REV_2_APD'] = np.abs(inst_df['REV_2_Excellent'] - inst_df['REV_1_Excellent'])/inst_df['REV_1_Excellent']
inst_df['REV_2_APD'] = inst_df['REV_2_APD'].fillna(0)

inst_df.groupby('GEV').median()[['predicted_APD', 'REV_2_APD']]
#%% Sample size per GEV at institutional level
n = inst_df.groupby(level=['GEV'], sort=False).size()

#%% Scatter plot at individual level (njs institutional level)
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(4*2, 3*2))
for idx, (gev, gev_df) in enumerate(inst_df.groupby(level='GEV', sort=False)):
  row = int(idx / 4)
  col = idx % 4
  ax = axs[row][col]
  ax.plot(gev_df['REV_1_SCORE'], gev_df['njs'], '.', alpha=0.5, mew=0)
  
  ax.set_yscale('log')  
  if (row == 2):
      if (col == 0):
          ax.set_xticks([0, 30])
      else:
          ax.set_xticks([30])
  else:
      ax.set_xticks([])

  if (col == 0):
      if (row == 2):
          ax.set_yticks([0.1, 10])
      else:
          ax.set_yticks([10])
  else:
      ax.set_yticks([])
  
  ax.set_xlim(0, 30)
  ax.set_ylim(0.1, 20)
  ax.text(0.05, 0.9, '{0}'.format(gev), transform=ax.transAxes, ha='left', va='baseline')  
    
axs[2][3].set_xticks([])
axs[2][3].set_yticks([])
  
fig.subplots_adjust(hspace=0,wspace=0)

fig.text(0.5, 0.05, 'Reviewer 1 score', ha='center')
fig.text(0.05, 0.5, 'NJS', va='center', rotation='vertical')
fig.savefig('../results/scatter_inst_njs.pdf')
#%% Scatter plot at individual level (reviewer individual level)
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(4*2, 3*2))
for idx, (gev, gev_df) in enumerate(inst_df.groupby(level='GEV', sort=False)):
  row = int(idx / 4)
  col = idx % 4
  ax = axs[row][col]
  ax.plot(gev_df['REV_1_SCORE'], gev_df['REV_2_SCORE'], '.', alpha=0.5, mew=0)
    
  if (row == 2):
      if (col == 0):
          ax.set_xticks([0, 30])
      else:
          ax.set_xticks([30])
  else:
      ax.set_xticks([])

  if (col == 0):
      if (row == 2):
          ax.set_yticks([0, 30])
      else:
          ax.set_yticks([30])
  else:
      ax.set_yticks([])
  
  ax.set_xlim(0, 30)
  ax.set_ylim(0, 30)
  ax.text(0.05, 0.9, '{0}'.format(gev), transform=ax.transAxes, ha='left', va='baseline')  
    
axs[2][3].set_xticks([])
axs[2][3].set_yticks([])
  
fig.subplots_adjust(hspace=0,wspace=0)

fig.text(0.5, 0.05, 'Reviewer 1 score', ha='center')
fig.text(0.05, 0.5, 'Reviewer 2 score', va='center', rotation='vertical')
fig.savefig('../results/scatter_inst_reviewer_uncertainty.pdf')