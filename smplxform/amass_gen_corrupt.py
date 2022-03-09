from __future__ import print_function
from __future__ import division
import argparse
import time
from time import sleep
import configparser
import cPickle as pickle
import math
from tqdm import trange
import numpy as np
from smplxformutils import rand_vec3, rand_joint_xform

INFILE_PATH='/media/data/amass/altogether/dataset.npz'
OUTFILE_PATH='/media/data/amass/altogether/dataset_corrupted.npz'

print('Loading input file...')
indata=np.load(INFILE_PATH)

outlist=[]
the_poses=np.array(indata['poses'][::10])
total_items=the_poses.shape[0]
print('Loaded: ' + str(total_items) + ' items.')

tr = trange(total_items, desc='Bar desc', leave=True)
for i in tr:
    tr.set_description("SMPL Corruption")
    tr.refresh()
    #sleep(0.00001)
    itemin=the_poses[i]
    for repetition in range(10):
        changeit=np.array([])
        for j in range(24):
            rjx=rand_joint_xform()
            changeit=np.concatenate((changeit,rjx))
        thetaout=np.clip(itemin[0:72]+changeit, -math.pi/2.0, math.pi/2.0)
        betaout=np.clip(itemin[72:82]+np.random.uniform(-2,2,10), -5, 5)
        itemout=np.concatenate((thetaout,betaout))
        valout=np.array([[itemin, itemout]])
        outlist.append(valout)

stackedlist=np.vstack(outlist)
np.savez_compressed(OUTFILE_PATH, poses=stackedlist)


