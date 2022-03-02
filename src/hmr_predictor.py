from __future__ import print_function 
import numpy as np
import sys
import tensorflow as tf
from utility import scale_and_crop
from utility import shift_verts
from utility import resize_img
from utility import get_sil_bbox
from utility import get_joint_bbox

# cofigure hmr path
import configparser
conf = configparser.ConfigParser()
conf.read(u'../conf.ini', encoding='utf8')
hmr_path = conf.get('HMR', 'hmr_path')
sys.path.append(hmr_path)
from absl import flags
import src.config
from src.RunModel import RunModel

def proc_sil(src_sil, proc_para):
    scale = proc_para["scale"],
    start_pt = proc_para["start_pt"]
    end_pt = proc_para["end_pt"]
    img_size = proc_para["img_size"]
    
    sil_scaled, _ = resize_img(src_sil, scale)
    
    margin = int(img_size / 2)
    sil_pad = np.pad(sil_scaled.tolist(), 
                     ((margin, ), (margin, )), 
                     mode='constant') # use 0 to fill the padding area of sil
    sil_pad = np.asarray(sil_pad).astype(np.uint8)
    std_sil = sil_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]
    return std_sil


class hmr_predictor():
    def __init__(self):
        self.config = flags.FLAGS
        self.config([1, "main"])
        self.config.load_path = src.config.PRETRAINED_MODEL
        self.config.batch_size = 1
        self.sess = tf.Session()
        self.model = RunModel(self.config, sess=self.sess)
    
    def predict(self, img, sil_bbox = False, sil = None, normalize = True, use_j_bbox=False, j_bbox=None):
        if sil_bbox is True:
            std_img, proc_para = preproc_img(img, True, sil, normalize = normalize)
        elif use_j_bbox is True:
            std_img, proc_para = preproc_img(img, normalize = normalize, use_j_bbox=use_j_bbox, j_bbox=j_bbox, margin=30)
        else:
            std_img, proc_para = preproc_img(img, normalize = normalize)
        std_img = np.expand_dims(std_img, 0)# Add batch dimension

        _, verts, ori_cam, joints, theta = self.model.predict(std_img)
        #_, verts, ori_cam, joints, theta = self.model.predict(std_img, get_theta=True)
        #print('Theta: ')
        #print(theta)
        #print('Joints: ')
        #print(joints)
        shf_vert = shift_verts(proc_para, verts[0], ori_cam[0])
        cam = np.array([500, 112.0, 112.0])
        return shf_vert, cam, proc_para, std_img[0]

    # Used for multiprocessing. Pre-processing already done in other process.
    def predict_nopp(self, std_img, proc_para):
        std_img = np.expand_dims(std_img, 0)# Add batch dimension
        _, verts, ori_cam, joints = self.model.predict(std_img)
        shf_vert = shift_verts(proc_para, verts[0], ori_cam[0])
        cam = np.array([500, 112.0, 112.0])
        return shf_vert, cam
    
    def close(self):
        self.sess.close()

# Added for multiprocessing
def predict_pponly(img, sil_bbox = False, sil = None, normalize = True, use_j_bbox=False, j_bbox=None):
    if sil_bbox is True:
        std_img, proc_para = preproc_img(img, True, sil, normalize = normalize)
    elif use_j_bbox is True:
        std_img, proc_para = preproc_img(img, normalize = normalize, use_j_bbox=use_j_bbox, j_bbox=j_bbox, margin=30)
    else:
        std_img, proc_para = preproc_img(img, normalize = normalize)
    return std_img, proc_para


# pre-processing original image to standard image for network
def preproc_img(img, sil_bbox = False, sil = None, img_size = 224, margin = 15, normalize = False, use_j_bbox=False, j_bbox=None):
    # if grey, change to rgb
    if len(img.shape) == 2:
        img = np.stack((img,)*3, -1)
    # if 4-channel, change to 3-channel
    if img.shape[2] == 4:
        img = img[:, :, :3]
    # get bbox of sil
    if sil_bbox is True:
        if sil is None:
            print("ERROR: sil not found in preproc_img().")
            return False
        # compute scale according to max of bounding box size
        bbox = get_sil_bbox(sil, margin = margin)
        bbox_size = [bbox[1]-bbox[0], bbox[3]-bbox[2]]
        if np.max(bbox_size) != img_size:
            scale = (float(img_size) / np.max(bbox_size))
        else:
            scale = 1.
        # compute center
        center = np.array([bbox[1]+bbox[0], bbox[3]+bbox[2]]).astype(float)
        center = np.round(center/2).astype(int)
        center = center[::-1]
        #center = center + np.array([img_size/2, img_size/2])
    elif use_j_bbox is True:
        bbox = get_joint_bbox(j_bbox, margin = margin)
        bbox_size = [bbox[1]-bbox[0], bbox[3]-bbox[2]]
        if np.max(bbox_size) != img_size:
            scale = (float(img_size) / np.max(bbox_size))
        else:
            scale = 1.
        # compute center
        center = np.array([bbox[1]+bbox[0], bbox[3]+bbox[2]]).astype(float)
        center = np.round(center/2).astype(int)
        center = center[::-1]
        #center = center + np.array([img_size/2, img_size/2])
    else:
        # compute scale according to max of width and height
        if np.max(img.shape[:2]) != img_size:
            scale = (float(img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        # get center
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        center = center[::-1]
    # apply scale and crop
    std_img, proc_para = scale_and_crop(img, scale, center, img_size)
    # normalize image to [-1, 1]
    if normalize is True:
        std_img = 2 * ((std_img / 255.) - 0.5)
    return std_img, proc_para

