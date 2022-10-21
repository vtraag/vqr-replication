from cmdstanpy import CmdStanModel
import os
#%%

stan_file = os.path.abspath('./model.stan')
model = CmdStanModel(stan_file=stan_file)
print(model)

#%%