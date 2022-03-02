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

import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model

## Load SMPL model (here we load the female model)
#m = load_model('../../models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
m = load_model(smpl_n_model)

## Assign random pose and shape parameters
#m.pose[:] = np.random.rand(m.pose.size) * .2
#m.betas[:] = np.random.rand(m.betas.size) * .03
#m.pose[0] = np.pi
#m.pose[:]=np.array([
#        3.1228497, 0.02263209, -0.56491035,
#        -0.03582681, 0.15027371, 0.31324956,
#        -0.2842782, -0.09210201, -0.43084067,
#        0.0430517, -0.01642321, 0.03031639,
#        0.38809246, 0.05426005, -0.04263579,
#        0.44428903, 0.15788978, -0.02794272,
#        0.03920969, -0.14065503, 0.02583855,
#        -0.01919581, 0.11426344, -0.12586972,
#        0.18905681, -0.16054858, 0.49753594,
#        0.08314583, -0.07150073, 0.04953786,
#        -0.31664142, 0.2882153, 0.35766253,
#        -0.3268497, 0.23936252, -0.7267184,
#        -0.00872714, 0.4825574, -0.13775131,
#        -0.12084066, -0.5697792, 0.28269175,
#        -0.11872026, 0.27406922, -0.28088626,
#        0.1670548, 0.39706194, -0.05969993,
#        -0.35136423, -1.072706, -0.89394045,
#        -0.04146488, 0.6313565, 1.1637819,
#        -0.36107314, -1.3064588, 0.2699906,
#        -0.6149708, 1.9246329, -0.77766395,
#        -1.0136985, -0.00708799, 0.327757,
#        -0.824884, 0.06962235, -0.2787365,
#        0.19544925, -0.04696121, -0.19775923,
#        -0.16997726, 0.12657999, 0.2953849 ])
#
#m.betas[:]=[ 0.31528103, -0.16102043, 0.9034513, 3.6004677, 2.0294583,
#          -0.14766431, -0.29047954, 0.19871268, 0.8781643, -0.24075282]

# Duncan
m.pose[:]=np.array([
    3.09209681e+00, 9.04115662e-02, -6.36945903e-01,
    -1.15702711e-02, 1.20201446e-01, 2.75974661e-01,
    -3.70851696e-01, -4.24526036e-02, -3.77204627e-01,
    7.20323846e-02, -3.58516350e-02, 2.47395616e-02,
    3.38428080e-01, 1.55544542e-02, -4.60999906e-02,
    5.13677120e-01, 1.77416027e-01, 1.07948203e-02,
    3.02835945e-02, -1.73599049e-01, 6.79066628e-02,
    -5.82361817e-02, 1.22312486e-01, -1.15890540e-01,
    8.19309279e-02, -1.34427339e-01, 4.36589986e-01,
    6.36300445e-02, -9.15359780e-02, 5.44258915e-02,
    -2.52774328e-01, 2.29515225e-01, 3.16194773e-01,
    -2.46402025e-01, 2.51304865e-01, -6.54805541e-01,
    -2.13666260e-03, 4.96386111e-01, -1.12899035e-01,
    -1.38813108e-01, -6.47870898e-01, 2.00670391e-01,
    -1.23274475e-01, 2.58787304e-01, -1.33183539e-01,
    2.40953982e-01, 3.29105914e-01, -7.76790977e-02,
    -3.66900772e-01, -1.03991544e+00, -6.88450456e-01,
    8.13802332e-02, 5.32597899e-01, 1.02590966e+00,
    -4.78861928e-01, -1.08845127e+00, 1.52976796e-01,
    -4.35340255e-01, 2.05429268e+00, -7.82835186e-01,
    -8.97597075e-01, -7.44033903e-02, 2.20523953e-01,
    -6.87785625e-01, 1.42936379e-01, -1.70832306e-01,
    1.94820806e-01, -4.99414653e-02, -2.10871667e-01,
    -1.31251782e-01, 1.51238412e-01, 2.65884608e-01
])

m.betas[:]=np.array([
    -1.74551755e-01, -2.99214721e-02, 6.83843195e-01, 3.47212029e+00,
    1.99634933e+00, -1.40190169e-01, -3.41550857e-01,  1.38338640e-01,
    8.74876499e-01, -2.26998240e-01
])

## Create OpenDR renderer
rn = ColoredRenderer()

## Assign attributes to renderer
w, h = (640, 480)

rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
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
import cv2
cv2.imshow('render_SMPL', rn.r)
print('..Print any key while on the display window')
cv2.waitKey(0)
cv2.destroyAllWindows()


## Could also use matplotlib to display
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
# import pdb; pdb.set_trace()
