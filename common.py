import pandas as pd

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

import matplotlib.pyplot as plt

def errorbar(x, y, ylow, yhigh=None, xlow=None, xhigh=None, **kwargs):
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
        
    return plt.errorbar(x, y, yerr=yerr, xerr=xerr, **kwargs)