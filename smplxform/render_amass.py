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
import cPickle as pickle
import math
from mesh_edit import fast_deform_dja
from mesh_edit import fast_deform_dsa
from smplxformutils import rand_vec3, rand_joint_xform, get_hmd_joints, make_trimesh, mag_mesh_diff
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

AMASS_NPZ_PATH='/media/data/amass/CMU/01/01_01_poses.npz'

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num', type = int, default = 0,
                    help = 'number of image to render')
opt = parser.parse_args()
npzidx=opt.num

## Load SMPL model (here we load the neutral model)
m = load_model(smpl_n_model)
m = load_model(smpl_m_model)

# Load NPZ
npzdata=np.load(AMASS_NPZ_PATH)
# SMPL model we're using can only handle 10 betas
#betas=npzdata['betas'][:10]
# The M/F models have 300 parameters
betas=np.zeros((300,))
betas[:16]=npzdata['betas']

# Read pose - only use first 22 joints of SMPL-H since the rest are hands
assert(npzidx<len(npzdata['poses']))
thetas=np.zeros((72,))
thetas[0:66]=npzdata['poses'][npzidx][0:66]
# Rotate body to face camera
thetas[0]=1.0*math.pi
thetas[1]=0
thetas[2]=0

# Translation params (don't need probably?)
translation=npzdata['trans'][npzidx]

# Original
m.pose[:]=thetas
m.betas[:]=betas

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
cv2.imshow('render_SMPL: AMASS', rn.r)

print('..Print any key while on the display window')
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()

m.pose[:]=orig_theta
m.betas[:]=orig_beta
mesh1=np.array(m.r)

m.pose[:]=corrupt_theta
m.betas[:]=corrupt_beta
mesh2=np.array(m.r)

meshdiff=mesh1-mesh2
print(meshdiff)

joints1=get_hmd_joints(mesh1)
joints2=get_hmd_joints(mesh2)
jointsdiff=joints1-joints2
print(jointsdiff)

# Do joint deform

# Make list of joints to modify
with open ('../predef/mesh_joint_list.pkl', 'rb') as fp:
    mesh_joint = pickle.load(fp)
point_list = mesh_joint["point_list"]
index_map = mesh_joint["index_map"]

new_joint_verts = []
for i, jdiff in enumerate(jointsdiff):
    for j in point_list[i]:
        new_joint_verts.append(mesh2[j] + jdiff)
new_joint_verts=np.array(new_joint_verts)

fd_ja = fast_deform_dja(weight = 10.0)
tick=time.time()
ja_mesh = fd_ja.deform(mesh2, new_joint_verts)
tock=time.time()
print('Elapsed time: '  + str(tock-tick))

# Then, do anchor deform
with open ('../predef/dsa_achr.pkl', 'rb') as fp:
    dic_achr = pickle.load(fp)
achr_id = dic_achr['achr_id']
achr_num = len(achr_id)

# make anchor posi
anchor_verts = np.zeros((200, 3))
for i in range(achr_num):
    anchor_verts[i, :] = ja_mesh[achr_id[i], :]

ja_trimesh = make_trimesh(ja_mesh, m.f, compute_vn = True)
vert_norms = ja_trimesh.vertex_normals()
print('vert_norms:')
print(vert_norms.shape)
mag_vert_norms=[np.linalg.norm(x) for x in vert_norms]
print('min: ' + str(min(mag_vert_norms)) + ' max: ' + str(max(mag_vert_norms)))

# 0.003 scale factor: where does it come from? Only matters for 2D projection?
# achr_para = achr_para * 0.003
achr_para=np.zeros((achr_num,))
new_av = []
ANCHOR_EXACT=True
if False:
    if ANCHOR_EXACT:
        # Method 1: Force anchors back to points in original mesh
        for j in range(achr_num):
            new_av.append(mesh1[achr_id[j]])
    else:
        # Method 2: Move anchors along normal as close as possible to original point
        for j in range(achr_num):
            # Calculate required adjustment along normal vector. Do this by
            # projecting desired adjustment onto normal vector and then adding to
            # initial points
            gt_achr_corr=mesh1[achr_id[j]]-ja_mesh[achr_id[j]]
            achr_para[j]=np.dot(gt_achr_corr, vert_norms[achr_id[j]])
            new_av.append(ja_mesh[achr_id[j]] + 
                          vert_norms[achr_id[j]] * achr_para[j])
else:
    # Method 2: Move anchors along normal as close as possible to original point
    for j in range(achr_num):
        # Calculate required adjustment along normal vector. Do this by
        # projecting desired adjustment onto normal vector and then adding to
        # initial points
        gt_achr_corr=mesh1[achr_id[j]]-mesh2[achr_id[j]]
        #achr_para[j]=mesh1[achr_id[j]]-mesh2[achr_id[j]]
        new_av.append(mesh2[achr_id[j]] + gt_achr_corr)
new_av = np.array(new_av)

fd_sa = fast_deform_dsa(weight=1.0)
sa_verts = fd_sa.deform(np.asarray(ja_mesh), 
                        new_av,
                       )
print('Sanity check (should be 0): ' + str(mag_mesh_diff(mesh1,mesh1)))
print('Diff corrupt and good: ' + str(mag_mesh_diff(mesh1,mesh2)))
print('Diff joint-adjusted and good: ' + str(mag_mesh_diff(mesh1,ja_mesh)))
print('Diff joint/anchor-adjusted and good: ' + str(mag_mesh_diff(mesh1,sa_verts)))

models=[('Orig', orig_theta, orig_beta), ('Corrupted', corrupt_theta, corrupt_beta)]

for mparams in models:
    # Original
    m.pose[:]=mparams[1]
    m.betas[:]=mparams[2]

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
    cv2.imshow('render_SMPL: ' + mparams[0], rn.r)

#####################################
# Now, show the joint-adjusted one
#####################################
## Create OpenDR renderer
rn = ColoredRenderer()

## Assign attributes to renderer
w, h = (640, 480)

rn.camera = ProjectPoints(v=ja_mesh, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=ja_mesh, f=m.f, bgcolor=np.zeros(3))

## Construct point light source
rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(ja_mesh),
    light_pos=np.array([-1000,-1000,-2000]),
    vc=np.ones_like(ja_mesh)*.9,
    light_color=np.array([1., 1., 1.]))

## Show it using OpenCV
import cv2
cv2.imshow('render_SMPL: ' + 'Joint-Adjusted', rn.r)

#####################################
# Now, show the final anchor-adjusted one
#####################################
## Create OpenDR renderer
rn = ColoredRenderer()

## Assign attributes to renderer
w, h = (640, 480)

rn.camera = ProjectPoints(v=sa_verts, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=sa_verts, f=m.f, bgcolor=np.zeros(3))

## Construct point light source
rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(sa_verts),
    light_pos=np.array([-1000,-1000,-2000]),
    vc=np.ones_like(sa_verts)*.9,
    light_color=np.array([1., 1., 1.]))


## Show it using OpenCV
import cv2
cv2.imshow('render_SMPL: ' + 'Anchor-Adjusted', rn.r)

print('..Print any key while on the display window')
cv2.waitKey(0)
cv2.destroyAllWindows()


## Could also use matplotlib to display
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
# import pdb; pdb.set_trace()
