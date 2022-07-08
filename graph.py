# -*- coding=utf-8 -*-

import networkx as nx

def to_bipartite(df):
    G = nx.Graph()
    G.add_nodes_from(df.index.values, bipartite = 0)
    G.add_nodes_from(df.columns.values, bipartite = 1)
    for col, rows in df.to_dict().items():
        for row,val in rows.items():
            if val==0:
                G.add_edge(row,col)
    return G