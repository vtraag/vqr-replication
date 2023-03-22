from pathlib import Path
import pandas as pd
import numpy as np

gev_names_df = pd.DataFrame(
  [['1',  "Mathematics & Computer Sciences"],
  ['2',   "Physics"],
  ['3',   "Chemistry"],
  ['4',   "Earth Sciences"],
  ['5',   "Biology"],
  ['6',   "Medicine"],
  ['7',   "Agricultural & veterinary sciences"],
  ['8b',  "Civil Engineering"],
  ['9',   "Industrial & Information Engineering"],
  ['11b', "Psychology"],
  ['13',  "Economics & Statistics"]], columns=['GEV_id', 'GEV'])

# Set directories
data_dir = Path('data')
results_dir = Path('results')
figure_dir = Path('figures')

# Create directories if they do not yet exist
for dir in [data_dir, results_dir, figure_dir]:
  dir.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt

def errorbar(ax, x, y, ylow, yhigh=None, xlow=None, xhigh=None, **kwargs):
    yerr = None
    if yhigh is None:
        yerr = ylow
    else:
        yerr = [y - ylow, yhigh - y]
        
    xerr = None
    if xlow is not None and xhigh is None:
        xerr = xlow
    elif xlow is not None and xhigh is not None:
        xerr= [x - xlow, xhigh - x]
        
    return ax.errorbar(x, y, yerr=yerr, xerr=xerr, **kwargs)

def extract_variable(df, variable, axis, index_dtypes=None):
  """ Extracts a variable from a summary dataframe from stan, and vectorizes
  the indices. That is, if there is a variable x[1], x[2], etc.. in a stan
  dataframe, it is extracted by this function to 
  id1    x
  1      value
  2
  """
  
  import re
  
  # Find relevant indices (rows in this case)
  
  var_re = re.compile(f'{variable}\[[\w,]*\]')
  
  if axis == 'index' or axis == 0:
    
    variable_index = [idx for idx in df.index if var_re.match(idx) is not None]
    
    # Split the name of each matching index, i.e.
    # variable[1,2] will be split into 1, 2
    split_variable_indices = [idx[len(variable)+1:-1].split(',')
                                   for idx in variable_index]
    
    if not index_dtypes is None:
      n_indices = len(index_dtypes)
      for indices in split_variable_indices:
        for i in range(n_indices):
          indices[i] = index_dtypes[i](indices[i])
          
    variable_df = df.loc[variable_index,]
    
    variable_df.index = pd.MultiIndex.from_tuples(
      [(variable, ) + tuple(indices) for indices in split_variable_indices])
    
  elif axis == 'columns' or axis == 1:

    variable_columns = [col for col in df.columns if var_re.match(col) is not None]

    # Split the name of each matching index, i.e.
    # variable[1,2] will be split into 1, 2
    split_variable_columns = [idx[len(variable)+1:-1].split(',')
                                     for idx in variable_columns]
    
    # Ensure the type is correct for each index (if specified)
    if not index_dtypes is None:
      n_indices = len(index_dtypes)
      for indices in split_variable_columns:
        for i in range(n_indices):
          indices[i] = index_dtypes[i](indices[i])
          
    variable_df = df.loc[:,variable_columns]
      
    variable_df.columns = pd.MultiIndex.from_tuples(
      [(variable, ) + tuple(indices) for indices in split_variable_columns])

  else:
    raise ValueError(f"Unknown axis: {axis}")
    
  return variable_df

def extract_indices(lst):
  """ Extracts indices from a list of variable names from stan. That is, if there 
  is a variable x[1], x[2], etc.. in a stan dataframe, it is extracted by this function 
  to a list of
  [('x', '1'),
   ('x', '2'),
   ...]
  """

  var_re = re.compile(f'(\w*)(\[[^\]]*\])?')

  split_list = []

  # Check each element
  for el in lst:

    # See if it matches the regex
    m = var_re.match(el)   

    # If it does
    if m:

      # We split it into variables and indices
      variable, indices = m.groups()

      # If we did find any indices
      if indices:
        # We will split all indices into separate elements
        indices = indices[1:-1].split(',')
        split_list.append([variable] + indices)
      else:
        # Else we just add the variable
        split_list.append([variable])
    else:
      # We only add the variable itself
      split_list.append([el])

  return split_list

def percentile(n):
  """"Named percentile function to be used for aggregation of pandas DataFrames"""
  def percentile_(x):
    if not isinstance(x,pd.Series):
      raise ValueError('need Series argument')
    return np.percentile(x, n)
  percentile_.__name__ = f'percentile_{n}'
  return percentile_

from scipy import stats

#%% Define distributions
def normal(mu, sigma):
  return stats.norm(loc=mu, scale=sigma)

def exponential(inv_rate):
  return stats.expon(scale=inv_rate)

def lognormal(mu, sigma):
  return stats.lognorm(scale=np.exp(mu), s=sigma)

#%% Define functions for creating unique IDs

def unique_id(df, start_id=1):
    """ Create unique identifiers for each unique row in df."""
    return df.groupby(list(df.columns)).ngroup() + start_id

def nuniq(df):
    return df.drop_duplicates().shape[0]

#%% Define functions for creating a group k-fold partition

from sklearn.model_selection import GroupKFold
def group_kfold_partition(groups, n_splits):
  """Partition into folds based on the groups. 
  
  This function is a simple helper function to translate results 
  from sklearn to a vector that contains in which fold an element 
  is in the test set.

  Returns a vector indicating for each element in which fold it is in the test set.
  """
  gkf = GroupKFold(n_splits=n_splits)
  n = len(groups)
  folds = np.repeat(-1, n)
  for i, (train, test) in enumerate(gkf.split(range(n), range(n), groups=groups)):
      folds[test] = i

  return folds
