import cPickle as pickle
import time
import numpy as np
import math
from smplxformutils import rand_vec3, rand_joint_xform

# Gaussian noise parameters (Unused; uniform distribution instead)
MEAN=0
STDDEV_BETA=1

def parse_sample(sample):
    rv={}
    rv['cam']=np.array(sample[0:3])
    rv['theta']=np.array(sample[3:75])
    rv['beta']=np.array(sample[75:85])
    return rv

# This file is just the first 100 samples
infile=open('./smallhmrdata.pkl','rb')
hmrdata=pickle.load(infile)
infile.close()

outds=[]
for sample in hmrdata:
    samplein=parse_sample(sample)
    sampleout={}

    changeit=np.array([])
    for i in range(int(len(samplein['theta'])/3)):
        rjx=rand_joint_xform()
        changeit=np.concatenate((changeit,rjx))
    sampleout['theta']=samplein['theta']+changeit

    sampleout['beta']=samplein['beta']+np.random.uniform(-2,2,10)
    sampleout['beta']=np.clip(sampleout['beta'],-5,5)
    outds.append((samplein['cam'], samplein['theta'],samplein['beta'],
                 sampleout['theta'],sampleout['beta']))

outfile=open('smplxformds.pkl','wb')
pickle.dump(outds, outfile)
outfile.close()
