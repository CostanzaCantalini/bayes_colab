import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as spst
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import datetime

import arviz as az
from cmdstanpy import cmdstan_path, CmdStanModel


def fit_all_arma(df):
    code = """
    
    data {
        int<lower=1> T;
        array[T] real y;
    }
    
    parameters {
        real<lower=-1,upper=1> phi;                  // autoregression coeff
        real<lower=-1,upper=1> theta;                // moving avg coeff
        real<lower=0> sigma;       // noise scale
    }
    
    model {
        vector[T] nu;              // prediction for time t
        vector[T] err;             // error for time t
        nu[1] = 0;                 // assume err[0] == 0
        err[1] = y[1] - nu[1];

        
        for (t in 2:T) {
            nu[t] = phi * y[t - 1] + theta * err[t - 1];
            err[t] = y[t] - nu[t];
        }
        phi ~ normal(0, 2) T[-1,1];
        theta ~ normal(0, 2) T[-1,1];
        sigma ~ cauchy(0, 5) T[0,];
        err ~ normal(0, sigma);    // likelihood
    }

    
    """

    stan_file = "./code.stan"

    with open(stan_file, "w") as f:
        print(code, file=f)

    stan_model = CmdStanModel(stan_file=stan_file)
    
    
    ritorno = {}
    
    
    stazioni = df.columns
    for stazione in stazioni:
        data = {
            "T":len(df[stazione]),
            "y":df[stazione].to_numpy()
        }


        stan_fit = stan_model.sample(data=data, chains=4, 
                                     parallel_chains=4, 
                                     iter_warmup=1000,  iter_sampling=10000, adapt_delta=0.99999)

        ritorno[stazione] = az.from_cmdstanpy(stan_fit)
        
    return ritorno


