import cPickle as pickle
import time

# This file is just the first 100 samples
infile=open('./smplxformds.pkl','rb')
data=pickle.load(infile)
infile.close()


