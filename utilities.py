
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def illustrate(dataframe, cluster, bold = False):
    df_ = dataframe.copy(deep=True)
    rows,cols = cluster
    rem_r = [r for r in dataframe.index.values if r not in rows]
    rem_c = [c for c in dataframe.columns.values if c not in cols]
    if bold:
       df_.loc[rows,cols] = df_.loc[rows,cols]+3
    sns.heatmap(df_.loc[rows+rem_r,cols+rem_c], cmap = 'binary', yticklabels= 1, cbar=False)

def graph_result(dataframe,clusters, interm_graph = False):
    #plotting sol
    df_sol = pd.DataFrame([], index = dataframe.index.values, columns= dataframe.columns.values)
    for i,clst in enumerate(clusters):
        df_sol.loc[clst[0],clst[1]] = i+1

        if interm_graph:
            # plotting single cluster
            plt.figure(f'clst_{i+1}', figsize=(5, 5))
            illustrate(dataframe, clst, bold = True)

    df_sol = df_sol.dropna(axis=1, how = 'all')
    df_sol = df_sol.fillna(0)
    color = sns.color_palette("Paired", len(clusters)+1)
    grid = sns.clustermap(df_sol, cmap = color, yticklabels= 1, cbar=True, figsize=(10,10),metric='hamming', method='complete')
    reordered_rows = grid.dendrogram_row.reordered_ind
    reordered_cols = grid.dendrogram_col.reordered_ind
    plt.figure('reordered_matrix', figsize=(8, 8))
    sns.heatmap(dataframe.iloc[reordered_rows,reordered_cols], cmap = 'binary', yticklabels= 1, cbar=False)
