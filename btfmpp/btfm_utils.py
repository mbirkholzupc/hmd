from __future__ import print_function 
from __future__ import division 

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

def get_joints_lspformat(entry):
    """
    Reads a JSON entry from the BTFM dataset and extracts joints
    in LSP format.
    """
    # Detect input joint format
    if entry['set'] == 'LSP':
        lsp_joints=read_lsp_joints(entry)
    elif entry['set'] == 'LSPET':
        lspet_joints=read_lspet_joints(entry)
        # No conversion needed
        lsp_joints=lspet_joints
    elif entry['set'] == 'MPII':
        mpii_joints=read_mpii_joints(entry)
        lsp_joints=mpii_to_lsp_joints(mpii_joints)
    elif entry['set'] == 'COCO':
        coco_joints=read_coco_joints(entry)
        lsp_joints=coco_to_lsp_joints(coco_joints)
    elif entry['set'] == '3DPW':
        tdpw_joints=read_3dpw_joints(entry)
        lsp_joints=tdpw_to_lsp_joints(tdpw_joints)
    elif entry['set'] == 'MI3':
        mi3_joints=read_mi3_joints(entry)
        lsp_joints=mi3_to_lsp_joints(mi3_joints)

    return lsp_joints

def read_lsp_joints(entry):
    return read_raw_joints(entry, 14)

def read_lspet_joints(entry):
    return read_raw_joints(entry, 14)

def read_mpii_joints(entry):
    return read_raw_joints(entry, 16)

def read_coco_joints(entry):
    return read_raw_joints(entry, 17)

def read_3dpw_joints(entry):
    return read_raw_joints(entry, 18)

def read_mi3_joints(entry):
    # Set to 3-wide, but note that v doesn't exist
    joints=np.zeros((3,28))
    for i, xy in enumerate(['x', 'y']):
        for j in range(28):
            key=str(xy)+str(j)
            joints[i,j]=entry[key]
    return joints

def read_raw_joints(entry, number):
    joints=np.zeros((3,number))
    for i, xyv in enumerate(['x', 'y', 'v']):
        for j in range(number):
            key=str(xyv)+str(j)
            joints[i,j]=entry[key]
    return joints

def mpii_to_lsp_joints(mpii_j):
    # Note: Based on HMD transform_mpii_joints but converts a single joint
    lsp_j = np.zeros((3, 14))
    lsp_j[:,0:6] = mpii_j[:,0:6] # lower limbs
    lsp_j[:,6:12] = mpii_j[:,10:16] # upper limbs
    lsp_j[:,12] = mpii_j[:,8] # neck
    lsp_j[:,13] = mpii_j[:,9] # head
    
    
    # head compensation - adjust top of head down towards neck
    # TODO: Did they really mean to adjust neck? (LSP: neck, MPII: upper neck)
    lsp_j[:2,13] = lsp_j[:2,13]*0.8 + lsp_j[:2,12]*0.2
    
    # ankle compensation - make ankle move slightly towards knee
    lsp_j[:2,5] = lsp_j[:2,5]*0.95 + lsp_j[:2,4]*0.05
    lsp_j[:2,0] = lsp_j[:2,0]*0.95 + lsp_j[:2,1]*0.05

    return lsp_j

def coco_to_lsp_joints(coco_j):
    # Note: Based on HMD transform_coco_joints
    lsp_j = np.zeros((3, 14))
    lsp_j[:,0] = coco_j[:,16]  # Right ankle
    lsp_j[:,1] = coco_j[:,14]  # Right knee
    lsp_j[:,2] = coco_j[:,12]  # Right hip
    lsp_j[:,3] = coco_j[:,11]  # Left hip
    lsp_j[:,4] = coco_j[:,13]  # Left knee
    lsp_j[:,5] = coco_j[:,15]  # Left ankle
    lsp_j[:,6] = coco_j[:,10]  # Right wrist
    lsp_j[:,7] = coco_j[:,8]  # Right elbow
    lsp_j[:,8] = coco_j[:,6]  # Right shoulder
    lsp_j[:,9] = coco_j[:,5]  # Left shoulder
    lsp_j[:,10] = coco_j[:,7]  # Left elbow
    lsp_j[:,11] = coco_j[:,9]  # Left wrist
    lsp_j[:,12] = np.array([-1, -1, 0])  # Neck
    lsp_j[:,13] = np.array([-1, -1, 0])  # Head top
    
    return lsp_j

def tdpw_to_lsp_joints(tdpw_j):
    ## Says it's COCO format (according to README), but joints don't match up
    #py3dpw_joints2d = [ 'nose', 'neck',  1
    #                    'right_shoulder', 'right_elbow', 'right_wrist', 4
    #                    'left_shoulder', 'left_elbow', 'left_wrist', 7
    #                    'right_hip', 'right_knee', 'right_ankle', 10
    #                    'left_hip', 'left_knee', 'left_ankle', 13
    #                    'right_eye', 'left_eye', 15
    #                    'right_ear', 'left_ear' ]  17
    #
    # 14 joints: Right ankle, Right knee, Right hip, Left hip
    # Left knee, Left ankle, Right wrist, Right elbow,
    # Right shoulder, Left shoulder, Left elbow, Left wrist,
    # Neck, Head top
    lsp_j = np.zeros((3, 14))
    lsp_j[:,0:3]=tdpw_j[:,10:7:-1]
    lsp_j[:,3:6]=tdpw_j[:,11:14]
    lsp_j[:,6:9]=tdpw_j[:,4:1:-1]
    lsp_j[:,9:12]=tdpw_j[:,5:8]
    lsp_j[:,12]=tdpw_j[:,1]
    lsp_j[:,13] = np.array([-1, -1, 0])  # Head top
    #0:10 1:9 2:8
    #3:11 4:12 5:13
    #6:4  7:3  8:2
    #9:5 10:6 11:7
    #12:1 13:-1
    return lsp_j

def mi3_to_lsp_joints(mi3_j):
    #3DHP
    #all_joint_names = {'spine3', 'spine4', 'spine2', 'spine', 'pelvis', ...     %5
    #        'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', ... %11
    #       'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', ... %17
    #       'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', ...        %23
    #       'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe'};
    #
    # 14 joints: Right ankle, Right knee, Right hip, Left hip
    # Left knee, Left ankle, Right wrist, Right elbow,
    # Right shoulder, Left shoulder, Left elbow, Left wrist,
    # Neck, Head top

    lsp_j = np.zeros((3, 14))
    lsp_j[:,0:3]=mi3_j[:,25:22:-1]
    lsp_j[:,3:6]=mi3_j[:,18:21]
    lsp_j[:,6:9]=mi3_j[:,16:13:-1]
    lsp_j[:,9:12]=mi3_j[:,9:12]
    lsp_j[:,12]=mi3_j[:,5]
    lsp_j[:,13]=mi3_j[:,7]
    #0:25 1:24 2:23
    #3:18 4:19 5:20
    #6:16  7:15  8:14
    #9:9 10:10 11:11
    #12:5 13:7

    # The shoulder joints are a bit too high, so let's try to adjust them down a bit
    # We'll draw a line from midpoints of shoulders to midpoints of hips and move
    # each joint in that direction about 10% of the line length
    #mid_shoulders=(lsp_j[:2,8]+lsp_j[:2,9])/2
    #mid_hips=(lsp_j[:2,2]+lsp_j[:2,3])/2
    #offset=(mid_hips-mid_shoulders)*0.05
    #lsp_j[:2,8]=lsp_j[:2,8]+offset
    #lsp_j[:2,9]=lsp_j[:2,9]+offset

    # Alternate method: make each shoulder move toward its hip
    # Both results seem nearly identical, so let's keep this since it's simpler
    lsp_j[:2,8] = lsp_j[:2,8]*0.95 + lsp_j[:2,2]*0.05
    lsp_j[:2,9] = lsp_j[:2,9]*0.95 + lsp_j[:2,3]*0.05

    return lsp_j


def plot_lsp_joints(img, joints, proc_para=None, color=[1,0,0]):
    newimg=img/255

    if proc_para:
        #print(joints)
        scale=proc_para['scale']
        img_size = proc_para['img_size']
        bias = np.array([img_size/2, img_size/2]) - proc_para["start_pt"]    
        #print(bias)
        #print(scale)
        scaledj=joints*scale
        #print(scaledj)
        scaledj[0,:]+=bias[0]
        scaledj[1,:]+=bias[1]
        #print(scaledj)
        adj_joints=scaledj
    else:
        adj_joints=joints

    joint_img=draw_lsp_joints(img.shape, adj_joints, color)
    outimg = alpha_blend(newimg, joint_img)

    return outimg

def draw_lsp_joints(dims, joints, color):
    # Return array of RGBA for blending with original image
    outimg=np.zeros((dims[0],dims[1],4))

    # TODO: Only plot valid joints. Right now, don't care as long as we clip.

    jointsx=np.round(joints[0,:])
    np.clip(jointsx, 0, dims[1]-1, out=jointsx)
    jointsx=jointsx.astype(int)
    jointsy=np.round(joints[1,:])
    np.clip(jointsy, 0, dims[0]-1, out=jointsy)
    jointsy=jointsy.astype(int)
    #jointsv=joints[2,:]

    # TODO: If too small, replace with wider circle, but will need to clip possibly
    for j in zip(jointsx, jointsy):
        outimg[j[1],j[0],:3] = color
        outimg[j[1],j[0],3] = 1

    return outimg

def alpha_blend(img1, img2):
    # img1 is a source 3-channel image (alpha=1.0)
    # img2 is 4-channel image to be alpha-blended on top of first image
    assert(img1.shape[2]==3)
    assert(img2.shape[2]==4)

    # Separate img2 into RGB and alpha
    img2_rgb=img2[:,:,:3]
    alpha=img2[:,:,3]

    alpha3=np.dstack((alpha,alpha,alpha))

    blended=img1*(1-alpha3) + img2_rgb*(alpha3)

    return blended
