from __future__ import print_function
import numpy as np
import PIL.Image
import torch
import cv2
import openmesh as om
import argparse
import pickle
import configparser
import sys
import os
sys.path.append("../src/")
import renderer as rd
from predictor import joint_predictor
from predictor import anchor_predictor
from network import shading_net
from data_loader import dataloader_demo
from mesh_edit import fast_deform_dja
from mesh_edit import fast_deform_dsa
from utility import str2bool
from utility import center_crop
from utility import get_joint_posi
from utility import get_anchor_posi
from utility import subdiv_mesh_x4
from utility import CamPara
from utility import make_trimesh
from utility import smpl_detoe
from utility import flatten_naval

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--img', required = True, 
                    help = 'path to results dir where HMR results are')
parser.add_argument('--outf', default = '', 
                    help = 'output folder (str, default is image name)')
parser.add_argument('--step', type = str2bool, default = True, 
                    help = 'save step results or not (True/False, default: True)')
parser.add_argument('--mesh', type = str2bool, default = False, 
                    help = 'save mesh or not (True/False, default: False)')
parser.add_argument('--gif', type = str2bool, default = False, 
                    help = 'make gif or not (True/False, default: False)')
parser.add_argument('--gpu', type = str2bool, default = True, 
                    help = 'enable gpu or not (True/False, default: True)')
opt = parser.parse_args()


if opt.gpu is False:
    print("cpu mode is slow, use '--gpu True' to enable gpu mode if conditions permit.")


# parse configures
conf = configparser.ConfigParser()
conf.read(u'../conf.ini', encoding='utf8')
dataset_path = conf.get('DEMOBTFM', 'dataset_path')
joint_model = conf.get('DEMOBTFM', 'joint_model')
anchor_model = conf.get('DEMOBTFM', 'anchor_model')

# Convert opt.img into better format
opt.img = opt.img.split("/")[-1] + '.png'

if opt.outf == '':
    opt.outf = "./result/" + opt.img.split("/")[-1][:-4] + "/"
print(opt)

# ==============================initialize==============================
print("initialize......", end='')

# initialize renderer
my_renderer = rd.SMPLRenderer()

# initialize joint and anchor predictor
pdt_j = joint_predictor(joint_model, gpu = opt.gpu)
pdt_a = anchor_predictor(anchor_model, gpu = opt.gpu)

# 
dataset = dataloader_demo(dataset_path)

# make verts for joint deform
with open ('../predef/mesh_joint_list.pkl', 'rb') as fp:
    item_dic = pickle.load(fp)
point_list = item_dic["point_list"]
index_map = item_dic["index_map"]

# make verts for anchor deform
with open ('../predef/dsa_achr.pkl', 'rb') as fp:
    dic_achr = pickle.load(fp)
achr_id = dic_achr['achr_id']
achr_num = len(achr_id)

cam_para = CamPara(K = np.array([[1000, 0, 224],
                                 [0, 1000, 224],
                                 [0, 0, 1]]))
with open ('../predef/exempt_vert_list.pkl', 'rb') as fp:
    exempt_vert_list = pickle.load(fp)

# get image and parameters of HMR prediction
test_num = 0
src_img = np.array(PIL.Image.open(opt.outf + "std_img.jpg"))

hmr_mesh = om.read_trimesh(opt.outf + "hmr_mesh.obj")
hmr_mesh.request_vertex_normals()
hmr_mesh.update_normals()
verts = hmr_mesh.points()
vert_norms = hmr_mesh.vertex_normals()

# make input tensor for joint net
joint_posi = get_joint_posi(verts)
proj_sil = my_renderer.silhouette(verts = verts)
src_sil = np.expand_dims(proj_sil, 2)
src_j = np.zeros((10, 4, 64, 64))
for i in range(len(joint_posi)):
    crop_sil = center_crop(src_sil, joint_posi[i], 64)
    crop_img = center_crop(src_img, joint_posi[i], 64)
    crop_img = crop_img.astype(np.float)
    crop_img = crop_img - crop_img[31, 31, :]
    crop_img = np.absolute(crop_img)
    crop_img = crop_img/255.0
    src_j[i,0,:,:] = np.rollaxis(crop_sil, 2, 0)
    src_j[i,1:4,:,:] = np.rollaxis(crop_img, 2, 0)    
    
print("done")

# ==============================predict joint==============================
print("joint deform......", end='')
joint_tsr = pdt_j.predict_batch(src_j)
joint_para = np.array(joint_tsr.data.cpu())
joint_para = np.concatenate((joint_para, np.zeros((10,1))),axis = 1)

# apply scale
joint_para = joint_para * 0.007# 0.007

flat_point_list = [item for sublist in point_list for item in sublist]

num_mj = len(point_list)
j_list = []
for i in range(num_mj):
    j_p_list = []
    for j in range(len(point_list[i])):
        j_p_list.append(verts[point_list[i][j]])
    j_list.append(sum(j_p_list)/len(j_p_list))

new_jv = []
ori_jv = []
for i in range(len(j_list)):
    # make new joint verts
    for j in point_list[i]:
        new_jv.append(verts[j] + joint_para[i])
        ori_jv.append(verts[j])
new_jv = np.array(new_jv)
ori_jv = np.array(ori_jv)

# joint deform
fd_ja = fast_deform_dja(weight = 10.0)
ja_verts = fd_ja.deform(np.asarray(verts), new_jv)

print("done")

# ==============================predict anchor==============================
print("anchor deform......", end='')

# make src_a
proj_sil_j = my_renderer.silhouette(verts = ja_verts)
src_sil_j = np.zeros((224, 224, 2))
src_a = np.zeros((200, 4, 32, 32))

# make anchor posi
anchor_verts = np.zeros((200, 3))
for i in range(achr_num):
    anchor_verts[i, :] = ja_verts[achr_id[i], :]
achr_posi = get_anchor_posi(anchor_verts)

for i in range(len(achr_posi)):
    crop_sil = center_crop(proj_sil_j, achr_posi[i], 32)
    crop_img = center_crop(src_img, achr_posi[i], 32)
    crop_img = crop_img.astype(np.int)
    crop_img = crop_img - crop_img[15, 15, :]
    crop_img = np.absolute(crop_img)
    crop_img = crop_img.astype(np.float)/255.0
    src_a[i,0,:,:] = crop_sil
    src_a[i,1:4,:,:] = np.rollaxis(crop_img, 2, 0)

# predict anchor
achr_tsr = pdt_a.predict_batch(src_a)
achr_para = np.array(achr_tsr.data.cpu())
achr_para = achr_para * 0.003

ori_av = []
new_av = []
for j in range(achr_num):
    ori_av.append(ja_verts[achr_id[j]])
    new_av.append(ja_verts[achr_id[j]] + 
                  vert_norms[achr_id[j]] * achr_para[j])
ori_av = np.array(ori_av)
new_av = np.array(new_av)

# anchor deform
fd_sa = fast_deform_dsa(weight=1.0)
sa_verts = fd_sa.deform(np.asarray(ja_verts), 
                        new_av,
                       )

print("done")

# ==============================render and save==============================
print("render and save......", end = '')

# render hmr
ori_proj_img = my_renderer(verts = verts)
ori_sil_img = my_renderer.silhouette(verts = verts)
ori_sil_img = np.stack((ori_sil_img,)*3).transpose((1,2,0))
ori_proj_img[ori_sil_img==0] = src_img[ori_sil_img==0]

# render ours_ja
ja_proj_img = my_renderer(verts = ja_verts)
ja_sil_img = my_renderer.silhouette(verts = ja_verts)
ja_sil_img = np.stack((ja_sil_img,)*3).transpose((1,2,0))
ja_proj_img[ja_sil_img==0] = src_img[ja_sil_img==0]

# render ours_sa
sa_proj_img = my_renderer(verts = sa_verts)
sa_sil_img = my_renderer.silhouette(verts = sa_verts)
sa_sil_img = np.stack((sa_sil_img,)*3).transpose((1,2,0))
sa_proj_img[sa_sil_img==0] = src_img[sa_sil_img==0]

# build output folder if not exist
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

if opt.step:
    PIL.Image.fromarray(src_img).save(opt.outf+"%04d_src.png" % test_num)
    PIL.Image.fromarray(ja_proj_img).save(opt.outf+"%04d_ours_j.png" % test_num)
    PIL.Image.fromarray(sa_proj_img).save(opt.outf+"%04d_ours_a.png" % test_num)
    PIL.Image.fromarray(ori_proj_img).save(opt.outf+"%04d_hmr.png" % test_num)

if opt.mesh:
    ja_mesh = make_trimesh(ja_verts, faces_smpl)
    om.write_mesh(opt.outf + "%04d_mesh_j.obj" % test_num, ja_mesh)
    sa_mesh = make_trimesh(sa_verts, faces_smpl)
    om.write_mesh(opt.outf + "%04d_mesh_a.obj" % test_num, sa_mesh)
    om.write_mesh(opt.outf + "%04d_mesh_v.obj" % test_num, deformed_mesh)
    print("mesh saved to [%s]" + opt.outf + "%04d_mesh.obj" % test_num)

if opt.gif:
    gb_src_img = src_img.copy()# green bounding source image
    gb_src_img[:,:3] = gb_src_img[:,:3]/2 + np.array([0, 128, 0])
    gb_src_img[:3,:] = gb_src_img[:3,:]/2 + np.array([0, 128, 0])
    gb_src_img[-3:,:] = gb_src_img[-3:,:]/2 + np.array([0, 128, 0])
    gb_src_img[:,-3:] = gb_src_img[:,-3:]/2 + np.array([0, 128, 0])
    PIL.Image.fromarray(gb_src_img).save(opt.outf+"%04d_src_gb.png" % test_num)
    
    gif_name = opt.outf + "%04d.gif" % test_num
    file_list = [opt.outf + "%04d_src_gb.png" % test_num,
                 opt.outf + "%04d_hmr.png" % test_num,
                 opt.outf + "%04d_ours_j.png" % test_num,
                 opt.outf + "%04d_ours_a.png" % test_num,
                 ]
    with open(opt.outf + "tmp_image_list.txt", 'w') as file:
        for item in file_list:
            file.write("%s\n" % item)
    
    # this require ImageMagick on Ubuntu (installed with Ubuntu 16 by default).
    # on windows use 'magick' instead of 'convert'
    os.system("convert -delay 100  @" + opt.outf + "tmp_image_list.txt %s" % gif_name)
    os.system("rm " + opt.outf + "tmp_image_list.txt")
    os.system("rm " + opt.outf + "%04d_src_gb.png" % test_num)

print("%s - finished, results are saved to [%s]" % (opt.img, opt.outf))
print("hmd done")
