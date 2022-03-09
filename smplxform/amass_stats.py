from __future__ import print_function
from __future__ import division
import argparse
import time
import configparser
import cPickle as pickle
import math
import numpy as np

OUTFILE_PATH='/media/data/amass/altogether/dataset.npz'
OUTFILE_PATH='/media/data/amass/altogether/dataset_compressed.npz'

parser = argparse.ArgumentParser()
parser.add_argument('--filelist', default = 'amass_files.txt', 
                    help = 'list of files to parse (str, default is amass_files.txt)')
opt = parser.parse_args()
filelist_fn=opt.filelist

with open(filelist_fn, 'r') as file:
    filelist=[line.rstrip() for line in file]

seq_counter={}
seq_counter['CMU']=0
seq_counter['KIT']=0
seq_counter['Bio']=0

frm_counter={}
frm_counter['CMU']=0
frm_counter['KIT']=0
frm_counter['Bio']=0


rows_to_combine=[]

abc=0
for fn in filelist:
    which=fn[18:21]
    seq_counter[which] = seq_counter[which]+1
    data=np.load(fn)
    betas=data['betas'][:10]
    handjoints=np.zeros((6,))
    for p in data['poses'][:,0:66]:
        thetabeta=np.concatenate((p,handjoints,betas))
        rows_to_combine.append(thetabeta)
        frm_counter[which]=frm_counter[which]+1

allofit=np.vstack(rows_to_combine)
np.savez_compressed(OUTFILE_PATH, poses=allofit)

print('Number of sequences:')
print(seq_counter)

print('Number of frames per sequence:')
print(frm_counter)
