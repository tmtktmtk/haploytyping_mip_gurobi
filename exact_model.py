# -*- coding=utf-8 -*-

import gurobipy as gp
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import *
from __init__ import get_data

def rearrange_matrix_model(dataframe):
    model = gp.Model('Rearrange_model')
    return model

def cell_activation_model(dataframe):
    # MODEL
    model =  gp.Model('Cell_activation_model')

    # Param
    model.Params.MIPGap = 0.01

    # DATA
    rows = dataframe.index.values
    cols = dataframe.columns.values
    cells= [(r,c) for r in rows for c in cols]
    most_freq = 1
    df_ = dataframe.replace(-1,most_freq)

    # VARIABLES
    lpRows = model.addVars(rows, lb = 0, ub = 1, vtype = gp.GRB.INTEGER,name = 'rw')
    lpCols = model.addVars(cols, lb = 0, ub = 1, vtype = gp.GRB.INTEGER,name = 'cl')
    lpCells= model.addVars(cells,lb = 0, ub = 1, vtype = gp.GRB.INTEGER,name = 'ce')

    # OBJECTIVE
    if most_freq == 0:
        model.setObjective(gp.quicksum([(1-df_.loc[c[0],c[1]]) * lpCells[c] for c in lpCells]), gp.GRB.MAXIMIZE)
    else:
        model.setObjective(gp.quicksum([(df_.loc[c[0],c[1]]) * lpCells[c] for c in lpCells]), gp.GRB.MAXIMIZE)

    # CONSTRAINTS
    for cell in lpCells:
        model.addConstr(1 - lpRows[cell[0]] >= lpCells[cell], f'{cell}_cr')
        model.addConstr(1 - lpCols[cell[1]] >= lpCells[cell], f'{cell}_cc')
        model.addConstr(2 - lpRows[cell[0]] - lpCols[cell[1]] <= 1 + lpCells[cell], f'{cell}_ccr')

    if most_freq == 0:
        model.addConstr(gp.quicksum([lpCells[coord]*df_.loc[coord[0],coord[1]] for coord in lpCells]) <= 0.025*gp.quicksum([lpCells[coord]*(1-df_.loc[coord[0],coord[1]]) for coord in lpCells]), 'err_thrshld')
    else:
        model.addConstr(0.02*gp.quicksum([lpCells[coord]*df_.loc[coord[0],coord[1]] for coord in lpCells]) >= gp.quicksum([lpCells[coord]*(1-df_.loc[coord[0],coord[1]]) for coord in lpCells]), 'err_thrshld')

    return model


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 6})

    df = get_data('data/problem_7.csv')

    grid = sns.clustermap(df,cmap='binary', yticklabels= 1, cbar=True, figsize=(5,10),metric='hamming', method='complete')
    reordered_rows = grid.dendrogram_row.reordered_ind
    reordered_cols = grid.dendrogram_col.reordered_ind
    df = df.iloc[reordered_rows,reordered_cols]
    df = df.iloc[:df.shape[0]//3,:]
    model = cell_activation_model(df)
    model.optimize()
    # print(model.getVars())
    rw = []
    cl= []
    solutions = {}
    for sol in range(model.solCount):
        model.setParam(gp.GRB.Param.SolutionNumber, sol)
        rw = []
        cl= []
        for var in model.getVars():
            if var.Xn == 0:
                name = var.VarName
                if name[0:2]=='rw':
                    rw += [name[3:-1]]
                elif name[0:2]=='cl':
                    cl += [name[3:-1]]
        solutions[sol] = (rw, cl)


    print(solutions)
    print(rw)
    print(len(rw))
    print()
    print(cl)
    print(len(cl))
    print(df.shape)
    print(model.runtime)

    plt.figure('original', figsize=(5, 10))
    sns.heatmap(df, cmap = 'binary', yticklabels= 1, cbar=False,square=False)

    # plt.figure('clst', figsize=(5, 10))
    # sns.heatmap(df.loc[rw,cl], cmap = 'binary', yticklabels= 1, cbar=False,square=True)

    plt.figure('sol', figsize=(5, 10))
    illustrate(df, (rw,cl), bold=True)

    plt.show()