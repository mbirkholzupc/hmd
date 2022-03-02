'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]


About the Script:
=================
This script demonstrates a few basic functions to help users get started with using 
the SMPL model. The code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Save the resulting body as a mesh in .OBJ format


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python hello_smpl.py

'''
from __future__ import print_function
import configparser
# parse configures
conf = configparser.ConfigParser()
conf.read(u'../conf.ini', encoding='utf8')
smpl_path = conf.get('SMPL', 'smpl_path')
smpl_m_model = conf.get('SMPL', 'smpl_m_model')
smpl_f_model = conf.get('SMPL', 'smpl_f_model')
smpl_n_model = conf.get('SMPL', 'smpl_n_model')

import sys
sys.path.append(smpl_path)

from smpl_webuser.serialization import load_model
import numpy as np

## Load SMPL model (here we load the female model)
## Make sure path is correct
m = load_model( smpl_n_model )

## Assign random pose and shape parameters
#m.pose[:] = np.random.rand(m.pose.size) * .2
#m.betas[:] = np.random.rand(m.betas.size) * .03
m.pose=[3.1228497, 0.02263209, -0.56491035, -0.03582681,
        0.15027371, 0.31324956, -0.2842782, -0.09210201,
        -0.43084067, 0.0430517, -0.01642321, 0.03031639,
        0.38809246, 0.05426005, -0.04263579, 0.44428903,
        0.15788978, -0.02794272, 0.03920969, -0.14065503,
        0.02583855, -0.01919581, 0.11426344, -0.12586972,
        0.18905681, -0.16054858, 0.49753594, 0.08314583,
        -0.07150073, 0.04953786, -0.31664142, 0.2882153,
        0.35766253, -0.3268497, 0.23936252, -0.7267184,
        -0.00872714, 0.4825574, -0.13775131, -0.12084066,
        -0.5697792, 0.28269175, -0.11872026, 0.27406922,
        -0.28088626, 0.1670548, 0.39706194, -0.05969993,
        -0.35136423, -1.072706, -0.89394045, -0.04146488,
        0.6313565, 1.1637819, -0.36107314, -1.3064588,
        0.2699906, -0.6149708, 1.9246329, -0.77766395,
        -1.0136985, -0.00708799, 0.327757, -0.824884,
        0.06962235, -0.2787365, 0.19544925, -0.04696121,
        -0.19775923, -0.16997726, 0.12657999, 0.2953849 ]

m.betas=[ 0.31528103, -0.16102043, 0.9034513, 3.6004677, 2.0294583,
          -0.14766431, -0.29047954, 0.19871268, 0.8781643, -0.24075282]

## Write to an .obj file
outmesh_path = './hello_smpl.obj'
with open( outmesh_path, 'w') as fp:
    for v in m.r:
        fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

    for f in m.f+1: # Faces are 1-based, not 0-based in obj files
        fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

## Print message
print('..Output mesh saved to: %s' % outmesh_path)
