# -*- coding=utf-8 -*-

import pandas as pd
import numpy as np

def gen(row, col, nbCluster = 3, isOverlap = False, errPercentage = 0):
    """Create a dataframe of zeros with clusters of 1"""
    df = pd.DataFrame(np.zeros(shape = (row+1,col+1), dtype = int))
    temp = df.copy(deep = True)

    for i in range(nbCluster):
        cl = temp.sample(frac=1/(nbCluster-i)).sample(frac=1/(nbCluster-i),axis = 1)
        r,c = cl.index.values, cl.columns.values
        if isOverlap:
            temp = temp.drop(columns = c)
        else:
            temp = temp.drop(index= r)
            temp = temp.drop(columns = c)
        df.loc[r,c] = 1

    return df

if __name__ == '__main__':
    m,n,c = 100,100,
    df = gen(m,n,c)
    df.to_csv(f'data/artificial_data_{m}x{n}-{c}.csv')