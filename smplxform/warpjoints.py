import cPickle as pickle
import time
import numpy as np

# Entry format: cam, theta(orig), beta(orig), theta(corrupt), beta(corrupt)
infile=open('./smplxformds.pkl','rb')
data=pickle.load(infile)
infile.close()

# First, generate mesh

# Then, calculate each joint in good/corrupt mesh and vector between them


