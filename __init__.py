# -*- coding=utf-8 -*-

from pulp_model import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================================ #
#                                  GET DATA                                    #
# ============================================================================ #


def get_data(filePath: str):
    """Read the matrix and parse it to a DataFrame"""
    df = pd.read_csv(filePath, index_col=0)
    df.columns = [f'c{c}' for c in df.columns.values]
    df.index = [f'r{i}_{r}' for r, i in enumerate(df.index.values)]
    return df


def stat_on_artificial_data():
    """Run all artificial data to get the stat"""
    FILES = [file for file in os.listdir(
        'data') if file.startswith('artificial_data')]
    print('Dim', 'Step', '|V|', '|E|', 'Time', 'Opt',
          'Root relaxation', 'Iterations', 'Nodes explored', sep=',')
    for file in FILES:
        df = get_data('data/'+file)
        cut_solve(df, idFile=file[16:-4])


# ============================================================================ #
#                                ILLUSTRATION                                  #
# ============================================================================ #


def illustrate_matrix(figure, dataframe):
    plt.figure(figure, figsize=(10, 10))
    sns.heatmap(dataframe, cbar=False, cmap='binary', square=True)


def illustrate_solutions(figure, dataframe):
    return

# ============================================================================ #
#                                    MAIN                                      #
# ============================================================================ #


if __name__ == '__main__':

    df = get_data('data/problem_2.csv')
    solutions = single_solve(df, printLog=True, printVar = False, min_col=5, min_row=5) 
  
    reads,cols,rem_r, rem_c = recluster(solutions,df)   


    print(df.mode().mode())

    # plotting
    # plt.rcParams.update({'font.size': 6})
    # illustrate_matrix(0,df)
    # illustrate_matrix(1,df.loc[reads+rem_r,cols+rem_c])
    # plt.show()

    # stat_on_artificial_data()

