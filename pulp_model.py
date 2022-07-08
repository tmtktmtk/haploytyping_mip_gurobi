# -*- coding=utf-8 -*-

import pulp as lp
from graph import to_bipartite

# ============================================================================ #
#                                   MODEL                                      #
# ============================================================================ #


def konig_primal_model(graph, min_row, min_col):
    """Get the maximum cardinality matching of a graph """
    # Linear problem for to find maximum cardinality matching of a bipartite graph
    prob = lp.LpProblem(name='konig_primal_model', sense=lp.LpMaximize)

    # Variables
    edges = {(row,col):lp.LpVariable(f'edge_{row}-{col}', cat='Continuous', lowBound=0, upBound=1) for row,col in graph.edges()} 
    rows = {row: lp.LpVariable(f'row_{row}', cat='Continuous', lowBound=0, upBound=1)
            for row, data in graph.nodes(data=True) if data['bipartite'] == 0}
    cols = {col: lp.LpVariable(f'col_{col}', cat='Continuous', lowBound=0, upBound=1)
            for col, data in graph.nodes(data=True) if data['bipartite'] == 1}    
    minRow = lp.LpVariable('minRow', cat = 'Continuous', upBound=0)
    minCol = lp.LpVariable('minCol', cat = 'Continuous', upBound=0)
    ROW = len(rows) - min_row
    COL = len(cols) - min_col
    # Objective function
    prob += lp.lpSum([edge for edge in edges.values()]+[ROW*minRow, COL*minCol]), 'max_card_matching'

    # Constraints
    for row, rowLpVar in rows.items():
        prob += (lp.lpSum([minRow] + [edges[edge] for edge in graph.edges(row)])) <= 1, f'endpoint at {row}'
    for col, colLpVar in cols.items():
        prob += (lp.lpSum([minCol] + [edges[row,col] for col,row in graph.edges(col)])) <= 1, f'endpoint at {col}'       

    return prob


def konig_dual_model(graph, num_row, num_col, min_row, min_col):
    """Get the minimum vertex cover of a graph """
    # Linear problem for to find minimim vertex cover of a bipartite graph
    prob = lp.LpProblem(name='konig_dual_model', sense=lp.LpMinimize)

    # Variables
    rows = {node: lp.LpVariable(f'row_{node}', cat='Integer', lowBound=0, upBound=1)
            for node, data in graph.nodes(data=True) if data['bipartite'] == 0}
    cols = {node: lp.LpVariable(f'col_{node}', cat='Integer', lowBound=0, upBound=1)
            for node, data in graph.nodes(data=True) if data['bipartite'] == 1}

    # Objective function
    # weight = num_row//num_col +1
    weight = 1
    prob += lp.lpSum([node for node in rows.values()] +
                     [weight*node for node in cols.values()]), 'min_num_vertices'

    # Constraints
    for row, col in graph.edges():
        prob += (rows[row] + cols[col] >= 1), f'edge_{row}_{col}'
    prob += (lp.lpSum(node for node in rows.values())
             ) <= len(rows) - min_row, 'max_rows_delete'
    prob += (lp.lpSum(node for node in cols.values())
             ) <= len(cols) - min_col, 'max_cols_delete'

    return prob

# ============================================================================ #
#                               SOLVE WITH DATA                                #
# ============================================================================ #


def primal_solve(dataframe, printLog=False, printVar = False, min_col=5, min_row=5):
    """Solve for maximum cardinality matching"""
    graph = to_bipartite(dataframe)
    prob = konig_primal_model(graph,min_row, min_col)
    prob.solve(lp.GUROBI_CMD(msg=False, logPath='log/stat_log.log'))
    if printLog:
        print_log_output(prob, printVar)
    return prob

def dual_solve(dataframe, printLog=False, printVar = False, min_col=5, min_row=5):
    graph = to_bipartite(dataframe)
    num_row, num_col = dataframe.shape
    prob = konig_dual_model(graph, num_row, num_col, min_row, min_col)
    if printLog:
        print_log_output(prob, printVar)    
    return prob

def single_solve(dataframe, printLog=False, printVar = True, min_row=5, min_col=5):
    """Solve for single best solution"""
    graph = to_bipartite(dataframe)
    num_row, num_col = dataframe.shape
    prob = konig_dual_model(graph, num_row, num_col, min_row, min_col)
    prob.solve(lp.GUROBI_CMD(msg=True, logPath='log/stat_log.log'))

    if printLog:
        print_log_output(prob, printVar)

    row_sols = [v.name[4:]
                for v in prob.variables() if v.varValue == 0 and v.name[0] == 'r']
    col_sols = [v.name[4:]
                for v in prob.variables() if v.varValue == 0 and v.name[0] == 'c']

    return {1: (row_sols, col_sols)}


def multiple_solve(dataframe, iters=3, min_row=5, min_col=5):
    """Solve for multiple solutions on the same matrix"""
    graph = to_bipartite(dataframe)
    num_row, num_col = dataframe.shape
    prob = konig_dual_model(graph, num_row, num_col, min_row, min_col)
    i = 1
    solutions = {}

    while lp.LpStatus[prob.status] != "Optimal" or i <= iters:
        prob.solve(lp.GUROBI_CMD(msg=False))
        row_sols = [v.name[4:]
                    for v in prob.variables() if v.varValue == 0 and v.name[0] == 'r']
        col_sols = [v.name[4:]
                    for v in prob.variables() if v.varValue == 0 and v.name[0] == 'c']
        prob += ((lp.lpSum(v if v.varValue == 0 else 1 -
                 v for v in prob.variables()))) >= 1, f'forbid_sol_{i}'
        solutions[i] = row_sols, col_sols
        i += 1

    return solutions


def cut_solve(dataframe, min_row=5, min_col=5, mulSolve=False, isOverlap=False, idFile=''):
    """solve for the best the solution, then cut the matrix"""
    df_ = dataframe.copy(deep=True)
    graph = to_bipartite(df_)
    num_row, num_col = df_.shape
    prob = konig_dual_model(graph, num_row, num_col, min_row, min_col)
    solutions = {}
    i = 1
    isFeasible = True

    while isFeasible:
        open("log/stat_log.log", "w").close()
        prob.solve(lp.GUROBI_CMD(msg=False, logPath='log/stat_log.log'))

        if lp.LpStatus[prob.status] != "Optimal" or num_col == 0 or num_row == 0:
            isFeasible = False
        else:
            row_sols = [v.name[4:]
                        for v in prob.variables() if v.varValue == 0 and v.name[0] == 'r']
            col_sols = [v.name[4:]
                        for v in prob.variables() if v.varValue == 0 and v.name[0] == 'c']

            solutions[i] = row_sols, col_sols

            if not isOverlap:
                df_ = df_.drop(index=row_sols)
            df_ = df_.drop(columns=col_sols)

            root_relax, nodes_explored, iterations = get_stat_from_log(
                'log/stat_log.log')

            # some stat
            print(idFile, f'Solve {i}', prob.numVariables(), prob.numConstraints(), prob.solutionCpuTime, prob.objective.value(), root_relax, iterations, nodes_explored, sep=',')

            # reset the lp prob
            graph = to_bipartite(df_)
            num_row, num_col = df_.shape
            if num_col != 0 and num_row != 0:
                prob = konig_dual_model(
                    graph, num_row, num_col, min_row, min_col)
            i += 1

    return solutions

# ============================================================================ #
#                                   UTILITIES                                  #
# ============================================================================ #


def print_log_output(prob,printVar=True):
    """Print the log output and problem solutions."""
    print()
    print('-' * 40)
    print('Stats')
    print('-' * 40)
    print()
    print(f'Number variables: {prob.numVariables()}')
    print(f'Number constraints: {prob.numConstraints()}')
    print()
    print('Time:')
    print(f'- (real) {prob.solutionTime}')
    print(f'- (CPU) {prob.solutionCpuTime}')
    print()
    print(f'Solve status: {lp.LpStatus[prob.status]}')
    print(f'Objective value: {prob.objective.value()}')
    if printVar:
        for v in prob.variables():
            print(v.name, ': ', v.varValue)


# TODO rewrite to better deal with overlapping
# FIXME bug when test with case dim 3x3 3 clusters
def recluster(solutions, original_dataframe):
    """reconstruct the matrix"""
    reads = []
    cols = []
    remaining_r = list(original_dataframe.index)
    remaining_c = list(original_dataframe.columns)
    for rs, ks in solutions.values():
        for r in rs:
            if r not in reads:
                reads = reads + [r]
                remaining_r.remove(r)
        for k in ks:
            if k not in cols:
                cols = cols + [k]
                remaining_c.remove(k)

    return reads, cols, remaining_r, remaining_c


def get_stat_from_log(logPath: str):
    log = open(logPath, 'r')
    content = log.readlines()
    infos = {
        'root relaxation' : '-',
        'node' : '-',
        'iterations': '-'
    }
    for line in content:
        if line.startswith('Root relaxation'):
            infos['root relaxation'] = line.split()[3][:-1]
        if line.startswith('Explored'):
            infos['node'] = line.split()[1]
            infos['iterations'] = line.split()[3][1:]
    return infos['root relaxation'], infos['node'], infos['iterations']
