# -*- coding=utf-8 -*-

from multiprocessing import pool
import gurobipy as gp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import model_selection
from utilities import *
from __init__ import get_data
from sklearn.metrics.pairwise import pairwise_distances


def grb_konig_dual_model(dataframe, min_rows=1, min_cols=1):
    """Get the minimum vertex cover of a graph """
    # MODEL
    model = gp.Model('konig_dual_model')

    # DATA
    rows = dataframe.index.values
    cols = dataframe.columns.values
    nb_1 = (dataframe == 1).sum().sum()
    nb_0 = (dataframe == 0).sum().sum()
    most_freq = 1 if nb_1 > nb_0 else 0
    most_freq = 1

    df_ = dataframe.replace(-1,most_freq)
 
    # VARIABLES
    lpRows = model.addVars(rows, lb=0, ub=1, vtype=gp.GRB.INTEGER, name='r')
    lpCols = model.addVars(cols, lb=0, ub=1, vtype=gp.GRB.INTEGER, name='c')

    # OBJECTIVE
    model.setObjective(gp.quicksum([(df_.loc[row]==most_freq).sum()*lpRows[row] for row in lpRows])+gp.quicksum(
        [(df_.loc[:,col]==most_freq).sum()*lpCols[col] for col in lpCols]), gp.GRB.MINIMIZE)

    # CONSTRAINTS
    for row in rows:
        for col in cols:
            if df_.loc[row, col] == (most_freq-1)*-1:
                model.addConstr(lpRows[row]+lpCols[col]
                                >= 1, name=f'edge_{row}-{col}')
    # model.addConstr(gp.quicksum(lpRows) <= len(
    #     rows) - min_rows, name='minRows')
    # model.addConstr(gp.quicksum(lpCols) <= len(
    #     cols) - min_cols, name='minCols')

    return (most_freq,model)


def grb_single_solve(model, poolSolve=True, printLog=True, stats=True, GAPTol=0.01):

    print()
    # set parameter
    if poolSolve:
        model.Params.PoolSearchMode = 2
        model.Params.MIPGap = GAPTol
    if not printLog:
        model.Params.OutputFlag = 0

    # optimize
    model.optimize()

    # status
    status = model.Status
    if status in (gp.GRB.INF_OR_UNBD, gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED):
        print('The model cannot be solved because it is infeasible or unbounded')
        return {}
    if status != gp.GRB.OPTIMAL:
        print('Optimization was stopped with status ' + str(status))
        return {}

    # save 10 best solutions // default num of solutions
    solutions = {}
    print(model.solCount)
    for sol in range(model.solCount):
        model.setParam(gp.GRB.Param.SolutionNumber, sol)
        r = []
        c = []
        for var in model.getVars():
            if var.Xn == 0:
                name = var.VarName
                if name[0] == 'r':
                    r += [name[2:-1]]
                else:
                    c += [name[2:-1]]

        solutions[sol] = (r, c)


    # for sol in solutions.values():
    #     print(sol)
    #     df.loc[sol[0],sol[1]]=df.loc[sol[0],sol[1]] + 2

    # plt.figure(np.random.randint(0,10,size=(1, 10)).sum(),figsize=(8,8))
    # sns.heatmap(df,cmap='binary', yticklabels= 1, cbar=False)

    # stats
    if stats:
        print()
        print('New solve')
        print()
        print('Stats')
        print('__'*40)
        print()
        print(f'- Runtime: {model.runtime}')

    return solutions


def merge_sols(dataframe, solutions):
    """return a cluster coordinate from overlapping solutions"""

    # check empty sols
    if not solutions:
        return ([], [])

    # count and sort
    rows = pd.DataFrame(0, index=dataframe.index.values,
                        columns=solutions.keys())
    cols = pd.DataFrame(0, index=dataframe.columns.values,
                        columns=solutions.keys())
    for key, comb in solutions.items():
        if comb[0] and comb[1]:
            r, c = comb
            rows.loc[r, key] = 1
            cols.loc[c, key] = 1
            # dataframe.loc[r,c] = dataframe.loc[r,c] + 1
    rows = rows.sum(axis=1)
    cols = cols.sum(axis=1)
    rows = rows.sort_values(ascending=False)
    cols = cols.sort_values(ascending=False)

    threshold_r = 0.8*rows.max()
    threshold_c = 0.8*cols.max()

    anchor_rows = rows[rows >= threshold_r+1].index.values
    anchor_cols = cols[cols >= threshold_c+1].index.values

    merged_rows = list(anchor_rows)
    merged_cols = list(anchor_cols)

    if merged_cols and merged_rows:
        for r, c in solutions.values():
            jaccard_rows = len(set(anchor_rows).intersection(
                set(r)))/len(set(list(anchor_rows)+r))
            jaccard_cols = len(set(anchor_cols).intersection(
                set(c)))/len(set(list(anchor_cols)+c))
            if jaccard_rows >= 0.7:
                merged_rows = merged_rows + r
            if jaccard_cols >= 0.7:
                merged_cols = merged_cols + c

    return list(set(merged_rows)), list(set(merged_cols))


def full_solve(dataframe, min_rows=5, min_cols=5, poolSolve=True, printLog=True, stats=True):

    # stopping condition
    r, c = dataframe.shape
    print(r,c)
    if r < min_rows or c < min_cols:
        print('dim too small')
        return []

    # model
    edge_num, model = grb_konig_dual_model(dataframe, min_rows, min_cols)

    # main solve
    sols = grb_single_solve(model, poolSolve, printLog, stats)
    clst = merge_sols(dataframe, sols)

    df_ = dataframe.copy(deep=True)
    if clst[0] and clst[1]:
        # sub solve 1
        df_1 = df_.drop(clst[1], axis=1).loc[clst[0]]
        clst1 = full_solve(df_1, min_rows, min_cols, poolSolve, printLog, stats)

        # sub solve 2
        df_2 = df_.drop(clst[0])
        clst2 = full_solve(df_2, min_rows, min_cols, poolSolve, printLog, stats)

    else:
        # sub solve 1
        df_1 = df_.iloc[:,:df_.shape[1]//2]
        clst1 = full_solve(df_1, min_rows, min_cols, poolSolve, printLog, stats)

        # sub solve 2
        df_2 = df_.iloc[:,df_.shape[1]//2:]
        clst2 = full_solve(df_2, min_rows, min_cols, poolSolve, printLog, stats)


    return [clst] + clst1 + clst2

# TODO improve code, reduce nested loop and repetition, there are bugs!!!


def merge_clusters(dataframe, clusters, simThreshold=0.8):

    print()
    print('Clusters merging...')
    print(f'Matrix dimension: {dataframe.shape}')
    print()
    # distinct split the clusters
    df_ = pd.DataFrame([], index=dataframe.index.values,
                       columns=dataframe.columns.values)
    for i, clst in enumerate(clusters):
        if len(clst[0])>=5 and len(clst[1])>=5:
            df_.loc[clst[0], clst[1]] = i+1
    df_ = df_.dropna(axis=1, how = 'all')
    df_ = df_.fillna(0)    
    df_ = df_.astype(str).apply(''.join, axis=1)

    rows_clsts = []
    for splitted_clst in df_.unique():
        if int(splitted_clst) != 0:  # if classified
            rows_clsts = rows_clsts + \
                [list(df_[df_ == splitted_clst].index.values)]
        else:
            for r in df_[df_ == splitted_clst].index.values:
                rows_clsts = rows_clsts + [[r]]

    isDistinct = False

    while not isDistinct:
        # merge similar cluster until all of them is distinct (lower than a threshold)
        df_mode = pd.DataFrame([dataframe.loc[clst, :].mode().iloc[0] for clst in rows_clsts], index=range(
            len(rows_clsts)), columns=dataframe.columns.values)
        df_mode = df_mode.drop_duplicates()

        # pairwise similarity between clusters
        ham_sim = 1 - pairwise_distances(df_mode, metric="hamming")
        ham_sim = pd.DataFrame(
            ham_sim, index=df_mode.index, columns=df_mode.index)

        print(ham_sim)

        sim = ham_sim[(ham_sim > simThreshold) & (ham_sim < 1)].sum().sum()

        if sim == 0:

            # classify smaller clusters/outsiders with relaxed threshold
            major_clsts = pd.DataFrame.from_dict({i: dataframe.loc[clst, :].mode().iloc[0] for i, clst in enumerate(
                rows_clsts) if len(clst) >= 5}, columns=dataframe.columns.values, orient='index')
            minor_clsts = pd.DataFrame.from_dict({i: dataframe.loc[clst, :].mode().iloc[0] for i, clst in enumerate(
                rows_clsts) if len(clst) < 5}, columns=dataframe.columns.values, orient='index')

            if minor_clsts.size != 0:
                ham_sim = 1 - \
                    pairwise_distances(
                        minor_clsts, major_clsts, metric="hamming")
                ham_sim = pd.DataFrame(
                    ham_sim, index=minor_clsts.index, columns=major_clsts.index)
                print(ham_sim)

                sim = ham_sim[(ham_sim > 0.7) & (ham_sim < 1)].sum().sum()
                if sim == 0:
                    isDistinct = True
                else:
                    temp_clsts = {}
                    for clst_id in ham_sim.columns.values:
                        temp_clsts[clst_id] = rows_clsts[clst_id]
                    for mi, ma in ham_sim.idxmax(axis=1).to_dict().items():
                        if ham_sim.loc[mi, ma] > 0.7:
                            temp_clsts[ma] = temp_clsts[ma] + rows_clsts[mi]
                        else:
                            temp_clsts[mi] = rows_clsts[mi]
                    rows_clsts = list(temp_clsts.values())

            else:
                isDistinct = True

        else:

            sim_clsts = {tuple(row[1][row[1] > simThreshold].index.values)
                         for row in ham_sim.iterrows()}

            temp_clsts = []
            for clst_tple in sim_clsts:
                temp = []
                for i in clst_tple:
                    temp = temp + rows_clsts[i]
                temp_clsts = temp_clsts + [temp]

            rows_clsts = temp_clsts

    return [(clst, list(dataframe.columns.values)) for clst in rows_clsts]


def pairwise_combination(dataframe, clusters, simThreshold=0.8):

    # id the rows
    df_ = pd.DataFrame(0, index=dataframe.index.values,
                       columns=dataframe.columns.values)
    for i, clst in enumerate(clusters):
        df_.loc[clst[0], clst[1]] = i+1
    df_ = df_.astype(str).apply(''.join, axis=1)

    # split horizontally
    rows_clsts = []
    unclassified = []
    for splitted_clst in df_.unique():
        if int(splitted_clst) != 0:  # if classified
            clst = list(df_[df_ == splitted_clst].index.values)

            if len(clst) > 1:
                rows_clsts = rows_clsts + [clst]
            else:
                unclassified = unclassified + clst
        else:
            for r in df_[df_ == splitted_clst].index.values:
                unclassified = unclassified + [r]

    for i, clst in enumerate(rows_clsts):
        print()
        print(i)
        print(clst)
        print()

    sim_ = []
    for i in range(len(rows_clsts)):
        temp = []
        for j in range(len(rows_clsts)):
            dist = [[1-((dataframe.loc[x]+dataframe.loc[y]) == 1).sum() /
                     dataframe.shape[1] for y in rows_clsts[j]] for x in rows_clsts[i]]
            # dist = 1 - pairwise_distances(dataframe.loc[rows_clsts[i]],dataframe.loc[rows_clsts[j]],metric='hamming')
            dist = pd.DataFrame(
                dist, index=rows_clsts[i], columns=rows_clsts[j])
            # print(dist)
            temp = temp + [dist.mean().mean()]
        sim_ = sim_ + [temp]

    sim_ = pd.DataFrame(sim_, index=range(len(rows_clsts)),
                        columns=range(len(rows_clsts)))
    for i in range(sim_.shape[0]):
        print(str(i)+': ', sim_.iloc[i, i])
    print(sim_)

    print('Unclassified')
    print(unclassified)
    # print()

    # for r in unclassified:
    #     for i in range(len(rows_clsts)):
    #         dist = 1 - pairwise_distances(dataframe.loc[r,:],dataframe.loc[rows_clsts[j]],metric='hamming')
    #         dist = pd.DataFrame(dist, index=[r], columns=rows_clsts[j])
    #         print(dist)

    return []


def split_solve(dataframe, min_rows=5, min_cols=5, poolSolve=True, printLog=True, stats=True):
    m, n = dataframe.shape
    clst_1 = full_solve(df.iloc[:, :n//2], min_rows,
                        min_cols, poolSolve, printLog, stats)
    clst_2 = full_solve(df.iloc[:, n//2:], min_rows,
                        min_cols, poolSolve, printLog, stats)
    return clst_1+clst_2


if __name__ == '__main__':

    # plt.rcParams.update({'font.size': 6})

    # # solve
    # df = get_data('data/problem_7.csv')
    # # df = pd.DataFrame(np.random.randint(0,2,size=(200, 3000)))
    # # df = pd.DataFrame([[1,0,0,1,1],[0,1,0,1,0],[0,0,1,1,1]])
    # # df = pd.read_csv('data/random.csv')
    # # df.to_csv('data/random.csv')
    # # print(df)

    # grid = sns.clustermap(df,cmap='binary', yticklabels= 1, cbar=True, figsize=(8,8),metric='hamming', method='complete')
    # reordered_rows = grid.dendrogram_row.reordered_ind
    # reordered_cols = grid.dendrogram_col.reordered_ind
    # df = df.iloc[reordered_rows,reordered_cols]

    # df = df.iloc[df.shape[0]//2:,df.shape[1]//2:]

    # plt.figure('original', figsize=(8,8))
    # sns.heatmap(df,cmap='binary', yticklabels= 1, cbar=False)

    # model = grb_konig_dual_model(df)

    # clsts = grb_single_solve(model,poolSolve=True,printLog=True)

    # for sol in clsts.values():
    #     print(sol)
    #     df.loc[sol[0],sol[1]]=df.loc[sol[0],sol[1]] + 2

    # plt.figure('sols',figsize=(8,8))
    # sns.heatmap(df,cmap='binary', yticklabels= 1, cbar=False)

    # # clsts = full_solve(df,printLog=True) 
    # # graph_result(df,clsts,interm_graph=False)

    # # clsts = merge_clusters(df,clsts,simThreshold=0.85)

    # # graph_result(df,clsts)

    # plt.show()

    ####################################

    plt.rcParams.update({'font.size': 6})

    # solve
    df = get_data('data/problem_2.csv')

    grid = sns.clustermap(df,cmap='binary', yticklabels= 1, cbar=True, figsize=(8,8),metric='hamming', method='complete')
    reordered_rows = grid.dendrogram_row.reordered_ind
    reordered_cols = grid.dendrogram_col.reordered_ind
    df = df.iloc[reordered_rows,reordered_cols]

    # df = df.iloc[df.shape[0]//2:,df.shape[1]//2:]

    plt.figure('original', figsize=(8,8))
    sns.heatmap(df,cmap='binary', yticklabels= 1, cbar=False)

    clsts = full_solve(df,printLog=True) 
    graph_result(df,clsts,interm_graph=False)

    clsts = merge_clusters(df,clsts,simThreshold=0.85)

    graph_result(df,clsts)

    plt.show()