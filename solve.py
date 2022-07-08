# -*- coding=utf-8 -*-

from models import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_data(filePath: str):
    """Read the matrix and parse it to a DataFrame"""
    df = pd.read_csv(filePath, index_col=0)
    df.columns = [f'cl_{i}_{c}' for i, c in enumerate(df.columns.values)]
    df.index = [f'rw_{i}_{r}' for i, r in enumerate(df.index.values)]
    return df

# ============================================================================ #
#                             SOLVE THE MODEL                                  #
# ============================================================================ #


def single_solve(dataframe, useModel='weighted_konig', minRows=10, minCols=15, errThreshold=0.025, printLog=False, printStat=False, drawGraph=False):

    if useModel == 'weighted_konig':
        m = grb_weighted_konig_dual_model(dataframe)
    elif useModel == 'konig':
        m = grb_restrained_konig_dual_model(dataframe, minRows, minCols)
    elif useModel == 'exact':
        m = grb_exact_model(dataframe, errThreshold)
    else:
        raise ValueError(
            'Invalid model, available models: weighted_konig, konig, exact')

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

    # illustrate the result
    if drawGraph:
        df = dataframe.copy(deep=True)
        for sol in solutions.values():
            df.loc[sol[0], sol[1]] = df.loc[sol[0], sol[1]] + 2

        plt.figure(f'model sols {useModel}', figsize=(8, 8))
        illustrate(df, clst[1], bold=True)

    print('Dimension:', f'{len(clst[1][0])}x{len(clst[1][1])}')
    print('Cluster internal pairwise hamming similarity (on the current matrix):', evaluate(dataframe,clst[1][0]))

    return clst


def full_solve(dataframe, useModel='weighted_konig', minRows=10, minCols=15, errThreshold=0.025, printLog=False, printStat=False):
    # stopping condition
    r, c = dataframe.shape
    print()
    print(f'Dimension {r}x{c}')
    if r < minRows or c < minCols:
        print('dim too small')
        return []

    # main solve
    clst = single_solve(dataframe, useModel, minRows, minCols,
                        errThreshold, printLog, printStat)

    df_ = dataframe.copy(deep=True)
    if clst[1][0] and clst[1][1]:
        # sub solve 1
        df_1 = df_.drop(clst[1][1], axis=1).loc[clst[1][0]]
        clst1 = full_solve(df_1, useModel, minRows, minCols,
                           errThreshold, printLog, printStat)

        # sub solve 2
        df_2 = df_.drop(clst[1][0])
        clst2 = full_solve(df_2, useModel, minRows, minCols,
                           errThreshold, printLog, printStat)
    else:
        # in case of invalid cluster found, cut the current matrix in 2 and retry
        # sub solve 1
        df_1 = df_.iloc[:, :df_.shape[1]//2]
        clst1 = full_solve(df_1, useModel, minRows, minCols,
                           errThreshold, printLog, printStat)

        # sub solve 2
        df_2 = df_.iloc[:, df_.shape[1]//2:]
        clst2 = full_solve(df_2, useModel, minRows, minCols,
                           errThreshold, printLog, printStat)

    return [clst] + clst1 + clst2


def hybrid_single_solve(dataframe, minRows=5, minCols=15, errThreshold=0.025, printLog=False, printStat=False, drawGraph=False):

    print('Solving...')
    most_freq, clst = single_solve(
        dataframe, useModel='weighted_konig', minRows=minRows, minCols=minCols,printLog=printLog,printStat=printStat)
    quality = evaluate(dataframe, clst[0])
    print('Cluster coordinates:')
    print(f'Reads: {clst[0]}')
    print(f'on {len(clst[1])} columns')
    if len(clst[0])<minRows or len(clst[1])<minCols:
        print('Insignificant cluster found (small dimension).')
        return (most_freq, ([], []))
    # quality check here, crude threshold
    if quality >= 0.8:
        sim_rows,sim_cols = get_similar_rows_cols(dataframe, clst)
        print(f'Similar reads: {sim_rows}')
        if sim_rows:
            print('Trying to improve the result cluster...')
            most_freq, clst = single_solve(
                dataframe.loc[clst[0]+sim_rows, clst[1]+sim_cols], useModel='exact', errThreshold=errThreshold,printLog=printLog,printStat=printStat)
            print('New cluster coordinates:')
            print(f'Reads: {clst[0]}')
            print(f'on {len(clst[1])} columns')
            quality = evaluate(dataframe, clst[0])

    # illustrate the result
    if drawGraph:
        df = dataframe.copy(deep=True)
        df.loc[clst[0], clst[1]] = df.loc[clst[0], clst[1]] + 2

        plt.figure('model sols hybrid', figsize=(8, 8))
        illustrate(df, clst, bold=True)

    print('Dimension:', f'{len(clst[0])}x{len(clst[1])}')
    print('Hybrid solve cluster internal pairwise hamming similarity (on the current matrix):', quality)

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
#                             MERGING CLSTS                                    #
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

def clstsMerge(dataframe, clusters):

    print()
    print('Clusters merging...')
    print(f'Matrix dimension: {dataframe.shape}')
    print()

    # distinct split the clusters
    df_ = pd.DataFrame([], index=dataframe.index.values,
                       columns=dataframe.columns.values)
    for i, clst in enumerate(clusters):
        if len(clst[1][0]) >= 5 and len(clst[1][1]) >= 5:
            df_.loc[clst[1][0], clst[1][1]] = i+1
    df_ = df_.dropna(axis=1, how='all')
    df_ = df_.fillna(0)
    df_id = df_.astype(str).apply(''.join, axis=1)

    # split horizontally
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
            for r in df_[df_ == splitted_clst].index.values:
                unclassified = unclassified + [r]

    # calculate the pairwise distance between clsts and merge
    isDistinct = False

    print(unclassified)

    while not isDistinct:
        sim_ = distance_matrix(dataframe, clsts)

        print(sim_)

        sim_clsts = {tuple(row[1][row[1] > 0.85].index.values)
                     for row in sim_.iterrows()}

        # check if clusts are distinct
        clstsDistinct = sim_[(sim_ > 0.85) & (sim_ < 1)].sum().sum() == 0

        if clstsDistinct:
            # classify outsiders
            if unclassified:
                outsiders_sim = []
                for i in range(len(clsts)):
                    dist = [[1-((dataframe.loc[x, clsts[i][1]]+dataframe.loc[y, clsts[i][1]])
                                == 1).sum()/len(clsts[i][1]) for y in clsts[i][0]] for x in unclassified]
                    dist = pd.DataFrame(
                        dist, index=unclassified, columns=clsts[i][0])
                    outsiders_sim = outsiders_sim + [list(dist.mean(axis=1))]

                outsiders_sim = pd.DataFrame(
                    outsiders_sim, index=range(len(clsts)), columns=unclassified)

                print(outsiders_sim)

                if outsiders_sim[(outsiders_sim > 0.75) & (outsiders_sim < 1)].sum().sum() == 0:
                    isDistinct = True
                else:
                    temp_clsts = {}
                    for clst_id in outsiders_sim.index.values:
                        temp_clsts[clst_id] = clsts[clst_id]

                    for mi, ma in outsiders_sim.idxmax().to_dict().items():
                        if outsiders_sim.loc[ma, mi] > 0.75:
                            tmp_rows = temp_clsts[ma][0] + [mi]
                            temp_clsts[ma] = (tmp_rows, temp_clsts[ma][1])
                            unclassified.remove(mi)

                    clsts = list(temp_clsts.values())
            else:
                isDistinct = True
        else:

            temp_clsts = []
            for clst_tple in sim_clsts:
                tmp_rows = []
                tmp_cols = []
                for i in clst_tple:
                    tmp_rows = tmp_rows + clsts[i][0]
                    tmp_cols = list(set(tmp_cols + clsts[i][1]))
                temp_clsts = temp_clsts + [(tmp_rows, tmp_cols)]

            clsts = temp_clsts

    return clsts + [([r, list(dataframe.columns.values)]) for r in unclassified]

def clsts_distance(dataframe, clst1, clst2):
    if clst1[0] == clst2[0] and clst1[1] == clst2[1]:
        return 1
    union_cols = list(set(clst1[1]+clst2[1]))
    dist = [[1-((dataframe.loc[x, union_cols]+dataframe.loc[y, union_cols]) == 1).sum() /
             len(union_cols) for y in clst2[0]] for x in clst1[0]]
    dist = pd.DataFrame(dist, index=clst1[0], columns=clst2[0])
    return dist.mean().mean()

def distance_matrix(dataframe, clsts):

    sim_ = []
    for i in range(len(clsts)):
        temp = []
        for j in range(len(clsts)):
            dist = clsts_distance(dataframe, clsts[i], clsts[j])
            temp = temp + [dist]
        sim_ = sim_ + [temp]

    sim_ = pd.DataFrame(sim_, index=range(len(clsts)),
                        columns=range(len(clsts)))

    return sim_

def jaccard_merge(dataframe, clsts):
    clsts1 = []
    clsts0 = []
    for num, coord in clsts:
        if coord[0] and coord[1]:
            if num == 1:
                clsts1 = clsts1 + [coord]
            else:
                clsts0 = clsts0 + [coord]

    jacc_dist1 = []
    for i in range(len(clsts1)):
        temp = []
        for j in range(len(clsts1)):
            jacc_score = len(set(clsts1[i][1]).intersection(
                set(clsts1[j][1])))/len(set(clsts1[i][1]+clsts1[j][1]))
            temp = temp + [jacc_score]
        jacc_dist1 = jacc_dist1 + [temp]

    jacc_dist1 = pd.DataFrame(jacc_dist1, index=range(
        len(clsts1)), columns=range(len(clsts1)))

    jacc = {tuple(row[1][row[1] > 0.6].index.values)
            for row in jacc_dist1.iterrows()}

    print(jacc)
    temp_clsts = []
    for clst_tple in jacc:
        tmp_rows = []
        tmp_cols = []
        for i in clst_tple:
            tmp_rows = list(set(tmp_rows + clsts1[i][0]))
            tmp_cols = list(set(tmp_cols + clsts1[i][1]))
        temp_clsts = temp_clsts + [(tmp_rows, tmp_cols)]

    clsts1_merged = temp_clsts

    print(jacc_dist1[jacc_dist1 > 0.5])
    return clsts1, clsts1_merged


def cplt_hierarchical_clstering(dataframe, clusters, estimation = 3):

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

    print('Number of distinct cluster after horizontally cutting:', len(clsts))            

    # calculate the initial similarity matrix with hamming distance
    sim_df = pd.DataFrame([], index = [(i,) for i in range(len(clsts))], columns= [(i,) for i in range(len(clsts))])
    print('Calculating proximity matrix...')
    for i in range(len(clsts)):
        print(sim_df)
        for j in range(i,len(clsts)):
            if i == j:
                sim_df.iloc[i,j] = 1.0
            else:
                union_cols = list(set(clsts[i][1]+clsts[j][1]))
                clst1_mode = dataframe.loc[clsts[i][0], union_cols].mode().iloc[0]
                clst2_mode = dataframe.loc[clsts[j][0], union_cols].mode().iloc[0]
                sim_score = ((clst1_mode+clst2_mode) == 1).sum()/len(union_cols)
                sim_df.iloc[i,j] = sim_score
                sim_df.iloc[j,i] = sim_score 

    print(sim_df)

    # clustering loop:
    # get the most similar position
    l = len(sim_df.index)
    while l>estimation:
        min_cell = (0,0)
        for i in range(l):
            for j in range(i,l):
                if sim_df.iat[i,j] < sim_df.iat[min_cell[0],min_cell[1]]:
                    min_cell = (i,j)
        name1 = sim_df.iloc[:,min_cell[0]].name
        name2 = sim_df.iloc[:,min_cell[1]].name
        new_clst = tuple(set(name1) | set(name2))

        new_sim = list(sim_df.iloc[:,[min_cell[0],min_cell[1]]].max(axis=1).values)
        sim_df.insert(l, new_clst, new_sim)
        df_ = pd.DataFrame([new_sim+[1]],index = [new_clst], columns = sim_df.columns)
        
        sim_df = pd.concat([sim_df,df_],axis = 0)
        sim_df.drop([name1,name2],inplace = True)
        sim_df.drop([name1,name2], axis = 1, inplace = True)
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
        result = result + [(clst_rows,clst_cols)]
    for res in result:
        print(res[0])
        print('Quality', evaluate(dataframe,res[0]))
    # classify the outsiders
    print('Classifying outsiders...')
    for r in unclassified: 
        hamming = [] 
        for i,clst in enumerate(result):
            clst_mode = dataframe.loc[clst[0], clst[1]].mode().iloc[0]
            outsider = dataframe.loc[r, clst[1]].mode().iloc[0]
            sim_score = ((clst_mode+outsider) == 1).sum()/len(union_cols)
            hamming = hamming + [sim_score]
        print(f'Classifying {r}')
        closest_clst = min(hamming)
        if closest_clst < 0.1:
            idx = hamming.index(closest_clst)
            result[idx] = (result[idx][0]+[r],result[idx][1])

    return result

# ============================================================================ #
#                            CLST EVALUATION                                  #
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
    mode_cols = dataframe.loc[rows,cols].mode(axis=1)
    l_cols = len(rows)
    df_ = dataframe.drop(cols,axis=1)
    dist = [(1 - ((df_.loc[rows,y]+mode_cols[0]) == 1).sum()/l_cols)for y in df_.columns.values]
    sim_cols = pd.Series(dist, index=df_.columns.values,dtype='float64')
    sim_cols = list(sim_cols[sim_cols > 0.8].index.values)
    
    mode_rows = dataframe.loc[rows,cols+sim_cols].mode()
    l_rows = len(cols)+len(sim_cols)
    df_ = dataframe.drop(rows)
    dist = [(1 - ((df_.loc[x]+mode_rows) == 1).sum(axis=1)/l_rows)[0] for x in df_.index.values]
    sim_rows = pd.Series(dist, index=df_.index.values,dtype='float64')
    sim_rows = list(sim_rows[sim_rows > 0.9].index.values)

    return (sim_rows,sim_cols)


# ============================================================================ #
#                              ILLUSTRATION                                    #
# ============================================================================ #


def illustrate(dataframe, coordinate, bold=False):
    df_ = dataframe.copy(deep=True)
    rows, cols = coordinate
    rem_r = [r for r in dataframe.index.values if r not in rows]
    rem_c = [c for c in dataframe.columns.values if c not in cols]
    if bold:
        df_.loc[rows, cols] = df_.loc[rows, cols]+3
    sns.heatmap(df_.loc[rows+rem_r, cols+rem_c],
                cmap='binary', yticklabels=1, cbar=False)


def graph_result(dataframe, clusters, interm_graph=False):
    # plotting sol
    df_sol = pd.DataFrame([], index=dataframe.index.values,
                          columns=dataframe.columns.values)
    for i, clst in enumerate(clusters):
        df_sol.loc[clst[0], clst[1]] = i+1

        if interm_graph:
            # plotting single cluster
            plt.figure(f'clst_{i+1}', figsize=(5, 5))
            illustrate(dataframe, clst, bold=True)

    df_sol = df_sol.dropna(axis=1, how='all')
    df_sol = df_sol.fillna(0)
    color = sns.color_palette("Paired", len(clusters)+1)
    grid = sns.clustermap(df_sol, cmap=color, yticklabels=1, cbar=True, figsize=(
        10, 10), metric='hamming', method='complete')
    reordered_rows = grid.dendrogram_row.reordered_ind
    reordered_cols = grid.dendrogram_col.reordered_ind
    plt.figure('reordered_matrix', figsize=(8, 8))
    sns.heatmap(dataframe.iloc[reordered_rows, reordered_cols],
                cmap='binary', yticklabels=1, cbar=False)

# ============================================================================ #
#                                   MAIN                                       #
# ============================================================================ #


if __name__ == '__main__':

    plt.rcParams.update({'font.size': 6})

    # solve
    df = get_data('data/kmers_problem.csv')

    # for i in range(3):
    #     df_ = get_data('data/problem_7.csv')
    #     df_.columns = [f'{c}_{i}' for c in df_.columns.values]
    #     df = pd.concat([df,df_],axis=1)

    # df = df.sample(frac=1).sample(frac=1,axis = 1)

    plt.figure('original', figsize=(8, 8))
    sns.heatmap(df, cmap='binary', yticklabels=1, cbar=False)

    # presolve
    grid = sns.clustermap(df, cmap='binary', yticklabels=1, cbar=True, figsize=(
        8, 8), metric='hamming', method='single')
    reordered_rows = grid.dendrogram_row.reordered_ind
    reordered_cols = grid.dendrogram_col.reordered_ind
    df = df.iloc[reordered_rows, reordered_cols]
    df = df.iloc[:,:df.shape[1]//4]

    clsts_tpl = hybrid_full_solve(df,minRows = 5, minCols =15, errThreshold=0.1)
    # clsts_tpl = single_solve(df,useModel='weighted_konig',minRows = 5, minCols =5, drawGraph = True,printStat=True)

    # clsts_tpl = single_solve(df,useModel='exact', drawGraph = True,printStat=True)

    # clsts_tpl = hybrid_single_solve(df,minRows = 5, minCols =5, drawGraph = True,printStat=True)

    clsts = [c[1] for c in clsts_tpl if c[1][0] and c[1][1]]
    print()
    print(f'Found {len(clsts)} cluster(s) satisfied the threshold of significance.')
    for i,clst in enumerate(clsts):
        print(f'Cluster {i}')
        print('Dimension:', f'{len(clst[0])}x{len(clst[1])}')
        print('Quality:', evaluate(df,clst[0]))
    graph_result(df,clsts)

    clsts = cplt_hierarchical_clstering(df, clsts_tpl, estimation=10)
    graph_result(df,clsts)

    plt.show()
