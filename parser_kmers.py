# -*- coding=utf-8 -*-

import pandas as pd
import numpy as np
import pysam as ps
from random import sample

samfile = ps.AlignmentFile('data/nanopore_sorted.bam','rb')

start = 0
end = start + 1000
k = 12
print('k =', k)

kmers = {} 
index_rows = []
metadata = {}
for read in samfile.fetch('NC_000913.3', start, end):
    read_sequence = read.query_sequence
    read_id = read.query_name
    index_rows = index_rows + [read_id]
    algn_start = read.query_alignment_start
    if read_sequence is not None:
        for i in range(len(read_sequence) - k + 1):
            kmer = read_sequence[i: i+k]
            current_pos = algn_start+i
            if kmer in metadata:
                # get the kmer's last appearance position
                latest_pos = metadata[kmer][-1]
                #update metadata
                #the smaller the distance between repeating kmers
                #the fewer informative positions we have
                if current_pos - latest_pos > 500:
                    # significant gap between duplicate kmer
                    # create new kmer (crude threshold)
                    kmers[f'{kmer}_{current_pos}'] = {read_id}
                    metadata[kmer] = np.append(metadata[kmer],[current_pos])
                else:
                    #get the nearest pos
                    idx = (np.abs(metadata[kmer]-current_pos)).argmin()
                    kmers[f'{kmer}_{metadata[kmer][idx]}'].add(read_id)
            else:
                #create a tuple of metadata and set of reads
                kmers[f'{kmer}_{current_pos}'] = {read_id}
                metadata[kmer] = np.array([current_pos])

print('Total number of read sequence in range:', len(index_rows))
print('Total number of kmer:', len(kmers))
print('Total number of kmer (without repeating):', len(metadata))

# look for suspicious position
print('Identify suspicious position...')
sus = [key for key, value in kmers.items() if len(value)>=(len(index_rows)/3)]
print('Number of informative position:', len(sus))

matrix = pd.DataFrame([],index = index_rows, columns= sus)
print('Generating matrix...')
for key in sus:
    matrix.loc[list(kmers[key]),[key]] = 1

#fill empty positions with 0
matrix = matrix.fillna(0)

print('Exporting to csv...')
print('Dimension of matrix:', matrix.shape)
matrix.to_csv('data/kmers_problem.csv')