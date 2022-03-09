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
- OpenCV [http://opencv.org/downloads.html] 
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the smpl model and rendering it using OpenDR 
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below). 

This code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
>	python render_smpl.py


'''
from __future__ import print_function
from __future__ import division
import argparse
import time
import configparser
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from smplxformutils import rand_vec3, rand_joint_xform, get_hmd_joints, make_trimesh, mag_mesh_diff

# parse configures
conf = configparser.ConfigParser()
conf.read(u'../conf.ini', encoding='utf8')
smpl_path = conf.get('SMPL', 'smpl_path')
smpl_m_model = conf.get('SMPL', 'smpl_m_model')
smpl_f_model = conf.get('SMPL', 'smpl_f_model')
smpl_n_model = conf.get('SMPL', 'smpl_n_model')

import sys
sys.path.append(smpl_path)

from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pc', type = int, default = 0,
                    help = 'which beta principal component to vary in range [0,9])')
opt = parser.parse_args()
beta_idx=opt.pc

## Load SMPL model (here we load the neutral model)
m = load_model(smpl_n_model)

# Create SMPL pose parameters (T-pose, standard shape)
thetas=np.zeros((72,))
thetas[0]=math.pi
initial_betas=np.zeros((10,))

# Which PC to vary is chosen by command-line options
# The beta_list are the values to which to set that component
beta_list=[-5, -2.5, 0, 2.5, 5]

#tick=time.time()
#tock=time.time()
#print('Elapsed time: '  + str(tock-tick))

## Create OpenDR renderer
rn = ColoredRenderer()
## Assign attributes to renderer
#w, h = (640, 480)
w, h = (240, 320)


renderlist=[]
for b in beta_list:
    betas=initial_betas.copy()
    betas[beta_idx]=b

    m.pose[:]=thetas
    m.betas[:]=betas

    #rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
    rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/1., c=np.array([w,h])/2., k=np.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

    ## Construct point light source
    rn.vc = LambertianPointLight(
        f=m.f,
        v=rn.v,
        num_verts=len(m),
        light_pos=np.array([-1000,-1000,-2000]),
        vc=np.ones_like(m)*.9,
        light_color=np.array([1., 1., 1.]))

    ## Show it using OpenCV
    #cv2.imshow('Varying beta ' + str(b), rn.r)
    renderlist.append(rn.r)

for img in renderlist:
    print(img.shape)

stacked=np.hstack(renderlist)
cv2.imshow('Varying beta ' + str(beta_idx), stacked)

print('..Print any key while on the display window')
cv2.waitKey(0)
cv2.destroyAllWindows()


## Could also use matplotlib to display
#import matplotlib.pyplot as plt
#plt.ion()
#plt.imshow(rn.r)
#plt.show()
#plt.waitforbuttonpress()
#import pdb; pdb.set_trace()
