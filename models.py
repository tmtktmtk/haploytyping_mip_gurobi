# -*- coding=utf-8 -*-

import gurobipy as grb

# ============================================================================ #
#                                      MODELS                                  #
# ============================================================================ #


def grb_weighted_konig_dual_model(dataframe):
    """Get the minimum vertex cover of a graph """
    # MODEL
    model = grb.Model('konig_dual_model')

    # Params
    model.Params.MIPGap = 0.01
    model.Params.PoolSearchMode = 2

    # DATA
    rows = dataframe.index.values
    cols = dataframe.columns.values
    nb_1 = (dataframe == 1).sum().sum()
    nb_0 = (dataframe == 0).sum().sum()
    most_freq = 1 if nb_1 > nb_0 else 0

    print('Clustering', most_freq)

    # VARIABLES
    lpRows = model.addVars(rows, lb=0, ub=1, vtype=grb.GRB.INTEGER, name='rw')
    lpCols = model.addVars(cols, lb=0, ub=1, vtype=grb.GRB.INTEGER, name='cl')

    # OBJECTIVE
    model.setObjective(grb.quicksum([(dataframe.loc[row] == most_freq).sum()*lpRows[row] for row in lpRows])+grb.quicksum(
        [(dataframe.loc[:, col] == most_freq).sum()*lpCols[col] for col in lpCols]), grb.GRB.MINIMIZE)

    # CONSTRAINTS
    for row in rows:
        for col in cols:
            if dataframe.loc[row, col] == (most_freq-1)*-1:
                model.addConstr(lpRows[row]+lpCols[col]
                                >= 1, name=f'edge_{row}-{col}')

    res = {
        'model': model,
        'most_freq': most_freq
    }

    return res


def grb_restrained_konig_dual_model(dataframe, minRows=5, minCols=5):
    """Get the minimum vertex cover of a graph """
    # MODEL
    model = grb.Model('konig_dual_model')

    # Params
    model.Params.MIPGap = 0.01
    model.Params.PoolSearchMode = 2

    # DATA
    rows = dataframe.index.values
    cols = dataframe.columns.values
    nb_1 = (dataframe == 1).sum().sum()
    nb_0 = (dataframe == 0).sum().sum()
    most_freq = 1 if nb_1 > nb_0 else 0

    # VARIABLES
    lpRows = model.addVars(rows, lb=0, ub=1, vtype=grb.GRB.INTEGER, name='rw')
    lpCols = model.addVars(cols, lb=0, ub=1, vtype=grb.GRB.INTEGER, name='cl')

    # OBJECTIVE
    model.setObjective(grb.quicksum([lpRows[row] for row in lpRows])+grb.quicksum(
        [lpCols[col] for col in lpCols]), grb.GRB.MINIMIZE)

    # CONSTRAINTS
    for row in rows:
        for col in cols:
            if dataframe.loc[row, col] == (most_freq-1)*-1:
                model.addConstr(lpRows[row]+lpCols[col]
                                >= 1, name=f'edge_{row}-{col}')

    #Difficult constraints
    model.addConstr(grb.quicksum(lpRows) <= len(
        rows) - minRows, name='minRows')
    model.addConstr(grb.quicksum(lpCols) <= len(
        cols) - minCols, name='minCols')

    res = {
        'model': model,
        'most_freq': most_freq
    }

    return res


def grb_exact_model(dataframe, errThreshold=0.025):
    # MODEL
    model = grb.Model('grb_exact_model')

    # Param
    model.Params.MIPGap = 0.01
    model.Params.TimeLimit = 60

    # DATA
    rows = dataframe.index.values
    cols = dataframe.columns.values
    cells = [(r, c) for r in rows for c in cols]
    nb_1 = (dataframe == 1).sum().sum()
    nb_0 = (dataframe == 0).sum().sum()
    most_freq = 1 if nb_1 > nb_0 else 0
    df_ = dataframe.replace(-1, most_freq)

    print('Clustering', most_freq)

    # VARIABLES
    lpRows = model.addVars(rows, lb=0, ub=1, vtype=grb.GRB.INTEGER, name='rw')
    lpCols = model.addVars(cols, lb=0, ub=1, vtype=grb.GRB.INTEGER, name='cl')
    lpCells = model.addVars(
        cells, lb=0, ub=1, vtype=grb.GRB.INTEGER, name='ce')

    # OBJECTIVE
    if most_freq == 0:
        model.setObjective(grb.quicksum(
            [(1-df_.loc[c[0], c[1]]) * lpCells[c] for c in lpCells]), grb.GRB.MAXIMIZE)
    else:
        model.setObjective(grb.quicksum(
            [(df_.loc[c[0], c[1]]) * lpCells[c] for c in lpCells]), grb.GRB.MAXIMIZE)

    # CONSTRAINTS
    for cell in lpCells:
        model.addConstr(1 - lpRows[cell[0]] >= lpCells[cell], f'{cell}_cr')
        model.addConstr(1 - lpCols[cell[1]] >= lpCells[cell], f'{cell}_cc')
        model.addConstr(2 - lpRows[cell[0]] - lpCols[cell[1]]
                        <= 1 + lpCells[cell], f'{cell}_ccr')

    if most_freq == 0:
        model.addConstr(grb.quicksum([lpCells[coord]*df_.loc[coord[0], coord[1]] for coord in lpCells]) <= errThreshold *
                        grb.quicksum([lpCells[coord]*(1-df_.loc[coord[0], coord[1]]) for coord in lpCells]), 'err_thrshld')
    else:
        model.addConstr(errThreshold*grb.quicksum([lpCells[coord]*df_.loc[coord[0], coord[1]] for coord in lpCells]) >= grb.quicksum(
            [lpCells[coord]*(1-df_.loc[coord[0], coord[1]]) for coord in lpCells]), 'err_thrshld')

    res = {
        'model': model,
        'most_freq': most_freq
    }

    return res