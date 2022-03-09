from __future__ import division
import cPickle as pickle
from math import pi, sqrt, sin, cos
import random
import numpy as np
import openmesh

def rand_vec3():
    theta=random.uniform(0, 2*pi)
    z=random.uniform(-1,1)
    x=sqrt(1-z*z)*cos(theta)
    y=sqrt(1-z*z)*sin(theta)

    return np.array([x,y,z])

def rand_joint_xform():
    # Value: up to +/-15 degrees, uniform distribution
    #        split among 3 components is random
    rvec=rand_vec3() # Random unit vector
    degrees=random.uniform(-15,15)
    radians=(degrees*pi)/180.0
    return rvec*radians

def get_hmd_joints(mesh):
    with open ('../predef/mesh_joint_list.pkl', 'rb') as fp:
        item_dic = pickle.load(fp)
    point_list = item_dic["point_list"]
    index_map = item_dic["index_map"]

    num_mj = len(point_list)
    j_list_1 = []
    for i in range(num_mj):
        j_p_list = []
        for j in range(len(point_list[i])):
            j_p_list.append(mesh[point_list[i][j]])
        j_list_1.append(sum(j_p_list)/len(j_p_list))

    j_list_1=np.array(j_list_1)
    return j_list_1

# Compose verts and faces to openmesh TriMesh
def make_trimesh(verts, faces, compute_vn = True):
    # if vertex index starts with 1, make it start with 0
    if np.min(faces) == 1:
        faces = np.array(faces)
        faces = faces - 1
    
    # make a mesh
    mesh = openmesh.TriMesh()

    # transfer verts and faces
    for i in range(len(verts)):
        mesh.add_vertex(verts[i])
    for i in range(len(faces)):
        a = mesh.vertex_handle(faces[i][0])
        b = mesh.vertex_handle(faces[i][1])
        c = mesh.vertex_handle(faces[i][2])
        mesh.add_face(a,b,c)

    # compute vert_norms
    if compute_vn is True:
        mesh.request_vertex_normals()
        mesh.update_normals()

    return mesh

def mag_mesh_diff(mesh1, mesh2):
    ptdiffs=np.linalg.norm(mesh1-mesh2, axis=1)
    return np.sum(ptdiffs)
