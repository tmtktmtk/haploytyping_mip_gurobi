# -*- coding=utf-8 -*-

import pandas as pd
import pysam as ps
from random import sample


samfile = ps.AlignmentFile('data/nanopore_sorted.bam','rb')

# crude parameters, taken from rust-mdbg can be changed
start = 0
end = start + 200
l = 8
d = 0.0003
k = 4
H = 256

print('k =', k)
print('l =', l)

# find universe l-mers in reads
kmers = set()
n = 0
total_l = 0
for read in samfile.fetch('NC_000913.3', start, end):
    read_sequence = read.query_sequence
    if read_sequence is not None:
        n+=1
        total_l=+len(read_sequence)
        for i in range(len(read_sequence) - l + 1):
            kmer = read_sequence[i: i+l]
            kmers.add(kmer)

print('Total number of kmer:', len(kmers))

# hashing
hashed_dict = {}
for kmer in kmers:
    key = hash(kmer) % H
    if key in hashed_dict:
        hashed_dict[key].add(kmer)
    else:
        hashed_dict[key] = {kmer}

min_alphabet = []
for key, values in hashed_dict.items():
    if key < d*H:
        min_alphabet = min_alphabet + list(values)

print('Minimizers alphabet created with length:', len(min_alphabet))

def convert_to_min_space(read):
    res = []
    i = 0
    while i<len(read) - l + 1:
        minimizer = read[i:i+l]
        if minimizer in min_alphabet:
            res = res + [f'm{min_alphabet.index(minimizer)}']
            i+=l
        else:
            i+=1
    return res

# convert reads to minimizer space and get all k-min-mers
print('Generating k-min-mers...')
kminmers = {} 
index_rows = []
for read in samfile.fetch('NC_000913.3', start, end):
    read_sequence = read.query_sequence
    if read_sequence is not None:
        min_space_seq = convert_to_min_space(read_sequence)
        read_id = read.query_name
        index_rows = index_rows + [read_id]
        for i in range(len(min_space_seq) - k + 1):
            kminmer = ''.join(min_space_seq[i: i+k])
            if kminmer in kminmers:
                kminmers[kminmer].add(read_id)
            else:
                kminmers[kminmer] = {read_id}  

print('Total number of read sequence in range:', len(index_rows))
print('Total number of kminmer:', len(kminmers))

matrix = pd.DataFrame([],index = index_rows, columns= kminmers.keys())
print('Generating matrix...')
for key in kminmers:
    matrix.loc[list(kminmers[key]),[key]] = 1

# print('Sample and remove irrelevant kminmers...')
# # remove if the number of 1 is less than 20%
# matrix = matrix.dropna(thresh = len(index_rows)*20//100, axis = 1)

#fill empty positions with 0
matrix = matrix.fillna(0)
print('Exporting to csv...')
print('Dimension of matrix:', matrix.shape)
matrix.to_csv('data/kminmers_problem.csv')
