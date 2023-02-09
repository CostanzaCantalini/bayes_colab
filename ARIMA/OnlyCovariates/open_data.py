
import pandas as pd
import numpy as np



def open():

    df = pd.read_csv('../data/data.csv')
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    df.replace(0.0, 1.0, inplace=True)
    df.fillna(np.NaN, inplace=True)
    
    fields = ["alt", "tipo", "zoning"]
    reg_matrix=pd.read_csv('../data/regression.csv',sep=";",names=fields)
    reg_matrix.iloc[:,0] = (reg_matrix.iloc[:,0]-reg_matrix.iloc[:,0].mean())/reg_matrix.iloc[:,0].std()
    dummies = pd.get_dummies(reg_matrix, columns=['tipo','zoning'], drop_first=True)
    
    return (df,dummies)

