# Author: 	Qiaojun Feng
# Date:		02/11/2018

import numpy as np
import pickle
import os
import random
import scipy
import scipy.spatial
import heapq
import pandas as pd
from pyntcloud import PyntCloud
from Q2_func import *

# change this to change the read object and a series of corresponding files
obj = 'teapot'
# obj = 'violin_case'

# read the vertices and faces data
# have been stored in the .pkl form
# calculate the area of all the triangles in the mesh
vertices = pickle.load(open(obj+'_vertices.pkl', 'rb'))
faces = pickle.load(open(obj+'_faces.pkl', 'rb'))
if os.path.isfile(obj+'_area.pkl'):
    area = pickle.load(open(obj+'_area.pkl', 'rb'))
else:
    area = triangle_faces_area(vertices,faces)


# transform to weight and assign number of sampling point on each faces
# N is total sampling points' number: point set P
if os.path.isfile('P_'+obj+'.pkl'):
	P = pickle.load(open('P_'+obj+'.pkl', 'rb'))
else:
	N = 11000
	sample_number = N*np.round(area/np.sum(area),np.log10(N).astype(int))
	sample_number = sample_number.astype(int)
	N_sample = np.sum(sample_number).astype(int)

	P = np.zeros([N_sample,3])
	index_sample = 0
	for index_face in range(faces.shape[0]):
	    vertice_coordinate = vertices[faces[index_face]]
	    for index_sample_mesh in range(sample_number[index_face]):
	        r1 = np.random.uniform()
	        r2 = np.random.uniform()
	        P[index_sample] = (1-np.sqrt(r1))*vertice_coordinate[0] + \
	                                    np.sqrt(r1)*(1-r2)*vertice_coordinate[1] + \
	                                    np.sqrt(r1)*r2*vertice_coordinate[2]
	        index_sample = index_sample + 1

# Farthest Point Sampling
# Given a point set and a distance matrix
# Return a sampled point set
# dist_file = dist_'+obj'_all.pkl
# d = pickle.load(open(dist_file, 'rb'))
d = scipy.spatial.distance.cdist(P,P)
n = 1000
N = P.shape[0]
S_index = random.sample(range(N),1)
for iters in range(n-1):
    min_distance = np.min(d[S_index,],0)
    index_max = np.argmax(min_distance)
    S_index.extend([index_max])
S = P[S_index]

# for display visualization
# only in jupyter notebook
points = pd.DataFrame(S, columns=['x', 'y', 'z'])
cloud = PyntCloud(points)
cloud.plot(lines=[], line_color=[])