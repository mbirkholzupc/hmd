import cPickle as pickle
import time

def parse_sample(sample):
    rv={}
    rv['cam']=sample[0:3]
    rv['theta']=sample[3:75]
    rv['beta']=sample[75:85]
    return rv

# Training file, but it's huge and takes forever to load
#tick=time.time()
#
#infile=open('/media/data/btfm/pp/hmrparams_train.pkl','rb')
#hmrdata=pickle.load(infile)
#infile.close()
#
#tock=time.time()
#print('Elapsed time: ' + str(tock-tick))


# This file is just the first 100 samples
infile=open('./smallhmrdata.pkl','rb')
hmrdata=pickle.load(infile)
infile.close()


