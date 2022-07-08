# -*- coding=utf-8 -*-

import gurobipy as grb
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser
# import matplotlib.pyplot as plt
# from solve import graph_result

def get_data(filePath: str):
    """Read the matrix and parse it to a DataFrame"""
    df = pd.read_csv(filePath, header = None)
    df.columns = [f'cl_{i}' for i in range(len(df.columns.values))]
    df.index = [f'rw_{i}' for i in range(len(df.index.values))]
    return df

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

# ============================================================================ #
#                             SOLVE THE MODEL                                  #
# ============================================================================ #


def single_solve(dataframe, useModel='weighted_konig', minRows=5, minCols=15, errThreshold=0.025, printLog=False, printStat=False):

    if useModel == 'weighted_konig':
        m = grb_weighted_konig_dual_model(dataframe)
    elif useModel == 'exact':
        m = grb_exact_model(dataframe, errThreshold)
    else:
        raise ValueError(
            'Invalid model, available models: weighted_konig, exact')

    model = m['model']
    most_freq = m['most_freq']

    if not printLog:
        model.Params.OutputFlag = 0

    # optimize
    model.optimize()

    # status check
    status = model.Status
    if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
        print('The model cannot be solved because it is infeasible or unbounded')
        return (most_freq, ([], []))
    if status != grb.GRB.OPTIMAL:
        print('Optimization was stopped with status ' + str(status))
        return (most_freq, ([], []))

    # stats
    if printStat:
        print()
        print('New solve')
        print()
        print('Stats')
        print('__'*40)
        print()
        print(f'- Runtime: {model.runtime}')

    # process solutions
    solutions = {}
    for sol in range(model.solCount):
        model.setParam(grb.GRB.Param.SolutionNumber, sol)
        rw = []
        cl = []
        for var in model.getVars():
            if var.Xn == 0:
                name = var.VarName
                if name[0:2] == 'rw':
                    rw += [name[3:-1]]
                elif name[0:2] == 'cl':
                    cl += [name[3:-1]]
        solutions[sol] = (rw, cl)

    # merge solutions, return the most_freq value in the submatrix along with the clst found
    clst = (most_freq, ([], []))
    if useModel == 'exact':
        clst = (most_freq, solutions[0])
    else:
        clst = (most_freq, solMerge(dataframe, solutions))

    return clst


def hybrid_single_solve(dataframe, minRows=5, minCols=15, errThreshold=0.025, printLog=False, printStat=False):

    print('Solving...')
    most_freq, clst = single_solve(
        dataframe, useModel='weighted_konig', minRows=minRows, minCols=minCols, printLog=printLog)
    quality = evaluate(dataframe, clst[0])
    print('Cluster coordinates:')
    print(f'Reads: {clst[0]}')
    print(f'on {len(clst[1])} columns')
    print(
        f'Cluster internal pairwise hamming similarity (on the current matrix): {quality}')
    if len(clst[0]) < minRows or len(clst[1]) < minCols:
        print('Insignificant cluster found (small dimension).')
        return (most_freq, ([], []))
    if quality >= 0.8:
        sim_rows, sim_cols = get_similar_rows_cols(dataframe, clst)
        print(f'Similar reads: {sim_rows}')
        if sim_rows:
            print('Trying to improve the result cluster...')
            most_freq, clst = single_solve(
                dataframe.loc[clst[0]+sim_rows, clst[1]+sim_cols], useModel='exact', errThreshold=errThreshold, printLog=printLog)
            print('New cluster coordinates:')
            print(f'Reads: {clst[0]}')
            print(f'on {len(clst[1])} columns')
            quality = evaluate(dataframe, clst[0])

    return (most_freq, clst)


def hybrid_full_solve(dataframe, minRows=5, minCols=15, errThreshold=0.025, printLog=False, printStat=False):
    # stopping condition
    r, c = dataframe.shape
    print()
    print(f'Dimension {r}x{c}')
    if r < minRows or c < minCols:
        print('Dimension of matrix too small.')
        return []

    # main solve
    clst = hybrid_single_solve(dataframe, minRows, minCols,errThreshold, printLog, printStat)

    df_ = dataframe.copy(deep=True)

    if not clst[1][0] or not clst[1][1]:
        # in case of invalid cluster found, try to look for a solution on the 1st half of the domain
        clst = hybrid_single_solve(df_.iloc[:, :df_.shape[1]//2], minRows, minCols,errThreshold, printLog, printStat)

    if clst[1][0] and clst[1][1]:
        # sub solve 1
        df_1 = df_.drop(clst[1][1], axis=1).loc[clst[1][0]]
        clst1 = hybrid_full_solve(df_1, minRows, minCols,errThreshold, printLog, printStat)

        # sub solve 2
        df_2 = df_.drop(clst[1][0])
        clst2 = hybrid_full_solve(df_2, minRows, minCols,errThreshold, printLog, printStat)
    else:
        #if still invalid cluster found, try to look for a solution on the 2nd half of the domain
        return hybrid_full_solve(df_.iloc[:, df_.shape[1]//2:], minRows, minCols,errThreshold, printLog, printStat)

    return [clst] + clst1 + clst2


# ============================================================================ #
#                      MERGING CLSTS & SOLUTIONS POOL                          #
# ============================================================================ #

def solMerge(dataframe, solutions):
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

    threshold_r = 0.7*rows.max()
    threshold_c = 0.7*cols.max()

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
            if jaccard_rows >= 0.6:
                merged_rows = merged_rows + r
            if jaccard_cols >= 0.6:
                merged_cols = merged_cols + c

    return (list(set(merged_rows)), list(set(merged_cols)))


def cplt_hierarchical_clstering(dataframe, clusters, estimation=3):

    print()
    print('Clusters merging...')
    print(f'Matrix dimension: {dataframe.shape}')
    print()

    # preprocess the cluster by cutting horizontally
    df_ = pd.DataFrame([], index=dataframe.index.values,
                       columns=dataframe.columns.values)
    for i, clst in enumerate(clusters):
        df_.loc[clst[1][0], clst[1][1]] = i+1
    df_ = df_.dropna(axis=1, how='all')
    df_ = df_.fillna(0)
    df_id = df_.astype(str).apply(''.join, axis=1)

    clsts = []
    unclassified = []
    for splitted_clst in df_id.unique():
        if int(splitted_clst) != 0:  # if classified
            clst = df_[df_id == splitted_clst]
            clst = clst.loc[:, (clst != 0).any(axis=0)]
            clst = (list(clst.index.values), list(clst.columns.values))

            if len(clst) > 1:
                clsts = clsts + [clst]
            else:
                unclassified = unclassified + clst
        else:
            for r in df_[df_id == splitted_clst].index.values:
                unclassified = unclassified + [r]

    # calculate the initial similarity matrix with hamming distance
    sim_df = pd.DataFrame([], index=[(i,) for i in range(len(clsts))], columns=[
                          (i,) for i in range(len(clsts))])
    print('Calculating proximity matrix...')
    for i in range(len(clsts)):
        print(sim_df)
        for j in range(i, len(clsts)):
            if i == j:
                sim_df.iloc[i, j] = 1.0
            else:
                union_cols = list(set(clsts[i][1]+clsts[j][1]))
                clst1_mode = dataframe.loc[clsts[i]
                                           [0], union_cols].mode().iloc[0]
                clst2_mode = dataframe.loc[clsts[j]
                                           [0], union_cols].mode().iloc[0]
                sim_score = ((clst1_mode+clst2_mode) ==
                             1).sum()/len(union_cols)
                sim_df.iloc[i, j] = sim_score
                sim_df.iloc[j, i] = sim_score

    print(sim_df)

    # clustering loop:
    # get the most similar position
    l = len(sim_df.index)
    while l > estimation:
        min_cell = (0, 0)
        for i in range(l):
            for j in range(i, l):
                if sim_df.iat[i, j] < sim_df.iat[min_cell[0], min_cell[1]]:
                    min_cell = (i, j)
        name1 = sim_df.iloc[:, min_cell[0]].name
        name2 = sim_df.iloc[:, min_cell[1]].name
        new_clst = tuple(set(name1) | set(name2))

        new_sim = list(
            sim_df.iloc[:, [min_cell[0], min_cell[1]]].max(axis=1).values)
        sim_df.insert(l, new_clst, new_sim)
        df_ = pd.DataFrame([new_sim+[1]], index=[new_clst],
                           columns=sim_df.columns)

        sim_df = pd.concat([sim_df, df_], axis=0)
        sim_df.drop([name1, name2], inplace=True)
        sim_df.drop([name1, name2], axis=1, inplace=True)
        l = len(sim_df.index)
        print(sim_df)
        print()

    result = []
    for clst in sim_df.index.values:
        clst_rows = []
        clst_cols = []
        for idx in clst:
            clst_rows = clst_rows + clsts[idx][0]
            clst_cols = list(set(clst_cols + clsts[idx][1]))
        result = result + [(clst_rows, clst_cols)]
    for res in result:
        print(res[0])
        print('Quality', evaluate(dataframe, res[0]))
    # classify the outsiders
    print('Classifying outsiders...')
    for r in unclassified:
        hamming = []
        for i, clst in enumerate(result):
            clst_mode = dataframe.loc[clst[0], clst[1]].mode().iloc[0]
            outsider = dataframe.loc[r, clst[1]].mode().iloc[0]
            sim_score = ((clst_mode+outsider) == 1).sum()/len(union_cols)
            hamming = hamming + [sim_score]
        print(f'Classifying {r}')
        closest_clst = min(hamming)
        if closest_clst < 0.1:
            idx = hamming.index(closest_clst)
            result[idx] = (result[idx][0]+[r], result[idx][1])

    return result


# ============================================================================ #
#                            CLST EVALUATION                                   #
# ============================================================================ #

def evaluate(dataframe, reads):
    """Check the quality of the read found with pairwise distance between reads"""
    if not reads:
        return 0
    l = dataframe.shape[1]
    dist = [[1 - ((dataframe.loc[x]+dataframe.loc[y]) ==
                  1).sum()/l for y in reads] for x in reads]
    dist = pd.DataFrame(dist, index=reads, columns=reads)
    return dist.mean().mean()


def get_similar_rows_cols(dataframe, clst):
    """Look for similar reads to the mode of a clst"""

    rows, cols = clst

    if not rows and not cols:
        return []

    # get similar cols:
    mode_cols = dataframe.loc[rows, cols].mode(axis=1)
    l_cols = len(rows)
    df_ = dataframe.drop(cols, axis=1)
    dist = [(1 - ((df_.loc[rows, y]+mode_cols[0]) == 1).sum()/l_cols)
            for y in df_.columns.values]
    sim_cols = pd.Series(dist, index=df_.columns.values, dtype='float64')
    sim_cols = list(sim_cols[sim_cols > 0.8].index.values)

    mode_rows = dataframe.loc[rows, cols+sim_cols].mode()
    l_rows = len(cols)+len(sim_cols)
    df_ = dataframe.drop(rows)
    dist = [(1 - ((df_.loc[x]+mode_rows) == 1).sum(axis=1)/l_rows)[0]
            for x in df_.index.values]
    sim_rows = pd.Series(dist, index=df_.index.values, dtype='float64')
    sim_rows = list(sim_rows[sim_rows > 0.9].index.values)

    return (sim_rows, sim_cols)


# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #
if __name__ == '__main__':

    # Arguments Parser
    parser = ArgumentParser()

    parser.add_argument('--filename', type=str, required=True, help='Path to the problem matrix')
    parser.add_argument('--minRows', type=int, default=5, help='Cluster minimum rows')
    parser.add_argument('--minCols', type=int, default=10, help='Cluster minimum cols')
    parser.add_argument('--errThreshold', type=float, default=0.025, help='Error rate to tolerate')
    parser.add_argument('--preSolve',default = False, help='Presolving, can improve the quality of clusters')
    parser.add_argument('--merge',default = True, help='Clusters merging')      

    args = parser.parse_args()

    df = get_data(args.filename)

    # presolve
    if args.preSolve:
        grid = sns.clustermap(df, cmap='binary', yticklabels=1, cbar=True, figsize=(
            8, 8), metric='hamming', method='single')
        reordered_rows = grid.dendrogram_row.reordered_ind
        reordered_cols = grid.dendrogram_col.reordered_ind
        df = df.iloc[reordered_rows, reordered_cols]

    clsts_tuple = hybrid_full_solve(df,minRows = args.minRows, minCols = args.minCols, errThreshold=args.errThreshold)

    clsts = [c[1] for c in clsts_tuple if c[1][0] and c[1][1]]
    for i,clst in enumerate(clsts):
        print(f'Cluster {i}')
        print('Dimension:', f'{len(clst[0])}x{len(clst[1])}')
        print('Quality:', evaluate(df,clst[0]))

    if args.merge:
        clsts = cplt_hierarchical_clstering(df, clsts_tuple, estimation=3)
    #     graph_result(df,clsts)        

    # plt.show()




