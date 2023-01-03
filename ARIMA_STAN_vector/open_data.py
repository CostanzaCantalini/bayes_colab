
import pandas as pd
import numpy as np



def open():
    df = pd.read_csv('data.csv')
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    df.fillna(np.NaN, inplace=True)
    return df
