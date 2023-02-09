

import pandas as pd
import numpy as np


from cmdstanpy import cmdstan_path, CmdStanModel


import open_data
import arviz as az



def compute(p, q, samples_per_chain = 1250, catene = 4, burnin = 1000):

    ritorno = open_data.open()
    df = ritorno[0]
    reg_matrix = ritorno[1]
    data = {
        "T":df.shape[0],
        "S":df.shape[1],
        "reg":reg_matrix.shape[1],
        "p":int(p),
        "q":int(q),
        "y":np.nan_to_num(df.to_numpy(), nan=1.0),
        "X":reg_matrix.to_numpy(),
        "is_missing":np.isnan(df.to_numpy()).astype(int),
        "missing_size": np.sum(np.isnan(df.to_numpy()).astype(int))
    }        
    stan_model = CmdStanModel(exe_file='./code')
    stan_fit = stan_model.sample(data=data, chains=catene, 
                                 parallel_chains=catene, 
                                 iter_warmup=burnin,  iter_sampling=samples_per_chain)
    inference_data = az.from_cmdstanpy(stan_fit)

    y_missing_list = []
    u = inference_data.posterior.missing_index_time.values[0,0,:] - 1
    v = inference_data.posterior.missing_index_station.values[0,0,:] - 1
    for i in range(data['missing_size']):
        y_missing_list.append({'Stazione':df.columns[v[i]],'Data':df.index[u[i]],'Samples':inference_data.posterior.w.values[:,:,i].reshape(catene*samples_per_chain)})

    return {'inference_data': inference_data, 'missing': y_missing_list}
