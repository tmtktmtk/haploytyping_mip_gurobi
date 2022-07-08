# -*- coding=utf-8 -*-

import pandas as pd
import pysam as ps

#TODO Optimize the algorithm
samfile = ps.AlignmentFile('data/nanopore_sorted.bam','rb')

# find suspicious positions in range
start = 400000
end = start + 50000
coverage = samfile.count_coverage('NC_000913.3', start, end)
df = pd.DataFrame(coverage, index = ['A','C','G', 'T'], columns = range(start,end))
df_ = df
df_sum = df.sum()
df = df/df_sum
df = (df>0.95)*1
df = df.sum()
sus_pos = df[df==0].index.values
print(len(sus_pos), 'suspicious positions found')

print('Parsing data...')
# align the reads
def read_parser(cigar, seq):
    res = []
    p = 0
    for x in cigar:
        # operation and length
        o,l = x
        if o == 0:
            res = res + [b for b in seq[p:p+l]]
            p = p+l
        if o == 2 or o == 3:
            res = res + (['x']*l)
        if o==4 or o==1:
            p = p+l
    return res

matrix = []
sol = []

for read in samfile.fetch('NC_000913.3', start, end):
    q_seq = read.query_sequence
    if q_seq is not None:
        cigar = read.cigartuples
        ref_strt = read.reference_start
        r = read_parser(cigar, q_seq)
        if ref_strt>start:
            r = ['x']*(ref_strt - start) + r
        else:
            r = r[start - ref_strt:] + ['x']*(end - read.reference_end)
        matrix = matrix + [r[:end-start]]
        sol = sol + [read.query_name[:1]]

# select sus position
mat = pd.DataFrame(matrix, index = sol, columns=range(start,end))
mat_sus = mat[sus_pos]
print('Dimension of matrix before data cleaning:', mat_sus.shape)

# cleaning the data taking the biggest window of covering
m,n = mat_sus.shape
# take rows with less than 20% empty cells
mat_sus = mat_sus.dropna(thresh = m*10//10)
mat_sus = mat_sus.dropna(how='any', axis = 1)
# delete extremely ambiguous rows >95%
mask = (mat_sus == 'x')*1
mask = mask.sum()/mat_sus.shape[0]
mask = mask[mask<0.95].index.values
mat_sus = mat_sus[mask]
mat_sus.to_csv('data/gen_matrix.csv')
print('Dimension of matrix after data cleaning:',mat_sus.shape)

#transform into (1,0) matrix
for col in mat_sus.columns.values:
    #get the coverage distribution of the position
    dis = df_[col].sort_values().index.values
    bases_map = {
        'x' : -1,
        dis[0]: -1,
        dis[1]: -1,
        dis[2]: 0,
        dis[3]: 1
    }
    mat_sus[col] = mat_sus[col].transform(lambda base: bases_map[base])

mat_sus.columns = [x for x in range(mat_sus.shape[1])]
# write to output
mat_sus.to_csv('data/problem.csv')