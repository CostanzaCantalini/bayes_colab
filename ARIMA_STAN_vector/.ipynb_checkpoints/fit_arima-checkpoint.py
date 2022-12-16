

import pandas as pd
import numpy as np


from cmdstanpy import cmdstan_path, CmdStanModel


import open_data
import arviz as az



def compute(stazione, p, q, samples_per_chain = 1250, catene = 4):
    df = open_data.open()
    data = {
        "T":len(df[stazione]),
        "p":int(p),
        "q":int(q),
        "y":np.nan_to_num(df[stazione].to_numpy(), nan=1.0),
        "is_missing":np.isnan(df[stazione].to_numpy()).astype(int)
    }        
    stan_model = CmdStanModel(exe_file='./code')
    stan_fit = stan_model.sample(data=data, chains=catene, 
                                 parallel_chains=catene, 
                                 iter_warmup=1000,  iter_sampling=samples_per_chain, adapt_delta=0.999)
    inference_data = az.from_cmdstanpy(stan_fit)

    missing_entries = np.isnan(df[stazione].to_numpy())
    y_missing = inference_data.posterior.y_missing.values[:,:,missing_entries]
    date_mancanti = df.index[missing_entries]
    
    y_missing_dict = {}
    for i in range(len(date_mancanti)):
        y_missing_dict[str(date_mancanti[i])] = y_missing[:,:,i]
            
    return {'inference_data': inference_data, 'reconstructed_y': y_missing_dict}

