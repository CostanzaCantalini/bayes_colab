
import pandas as pd
import numpy as np



def open():
    df = pd.read_csv('data.csv')
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    df.fillna(np.NaN, inplace=True)
    
    fields = ["alt", "tipo", "zoning"]
    reg_matrix=pd.read_csv('regression.csv',sep=";",names=fields)
    
    dummies = pd.get_dummies(reg_matrix, columns=['tipo','zoning'], drop_first=True)
    
    
    return (df,dummies)

