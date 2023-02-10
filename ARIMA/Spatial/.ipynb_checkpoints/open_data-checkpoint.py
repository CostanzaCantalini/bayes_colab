
import pandas as pd
import numpy as np



def open():

    df = pd.read_csv('../data/data.csv')
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df.set_index('Date', inplace=True)
    df.replace(0.0, 1.0, inplace=True)
    df.fillna(np.NaN, inplace=True)
    
    reg_matrix=pd.read_csv('../data/regression.csv')    
    reg_matrix = reg_matrix.drop('Stazione', axis=1)
    reg_matrix.iloc[:,0] = (reg_matrix.iloc[:,0]-reg_matrix.iloc[:,0].mean())/reg_matrix.iloc[:,0].std()
    dummies = pd.get_dummies(reg_matrix)
    dummies = dummies.drop('TipoStazione_Fondo', axis=1)
    dummies = dummies.drop('TipoArea_Urbano', axis=1)
    dummies = dummies.drop('Zonizzazione_Pianura Ovest', axis=1)
    dummies.columns = ['Altitude','Type_Industrial','Type_Traffic','Type_Rural','Type_SubUrban','Zone_Agglomerato','Zone_Appennino','Zone_Est']

    coordinates = ["latitude","longitude"]
    coord_matrix = pd.read_csv('../data/coord.csv',sep=",",names=coordinates)
    coord_matrix = (coord_matrix - coord_matrix.mean())
    
    return (df,dummies,coord_matrix)

