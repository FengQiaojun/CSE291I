import numpy as np
import pickle
import os
import scipy
import heapq
import pandas as pd
from pyntcloud import PyntCloud

# Given a point cloud with n points, using k nearest-neighbors algorithm to build a graph on it
# return the distance matrix(n*n) of the graph. Inf means not edge between 2 points
# 1. make sure each edge is bilateral
# 2. make sure the full connectivity, i.e. any point can reach any point
# Because of the request above, degree of each point may vary
# if return nan, means that some points are not connected to the others
# maybe you may want to increase k

def knn_distance_init(points,k):
    k = 5
    N = points.shape[0]
    # calculate the distance matrix between each pair of points
    Dist = scipy.spatial.distance.cdist(points,points)
    # initialize a distance matrix and set all the elements to zero
    D = np.zeros(Dist.shape)
    D[:] = np.inf
    # for each point, find its (k+1) values that are the minimal(because including 0)
    # assign them to matrix D
    for index_point in range(N):
        k_smallest_index = np.argpartition(Dist[index_point], k+1)[:k+1]
        D[index_point][k_smallest_index] = Dist[index_point][k_smallest_index]
    # 1. make sure each edge is bilateral
    # by comparing D[i][j] and D[j][i]
    # let D[i][j]=D[j][i] = min(D[i][j],D[j][i])
    for index_i in range(N):
        for index_j in range(index_i+1,N):
            if D[index_i][index_j] != D[index_j][index_i]:
                D_min = np.min([D[index_i][index_j],D[index_j][index_i]])
                D[index_i][index_j] = D_min
                D[index_j][index_i] = D_min
    # 2. make sure the full connectivity
    # by expanding from one point till the end
    check_connection = np.zeros(N)
    check_connection[0] = 1
    open_list = [0]
    for iters in range(N-1):
        if len(open_list) == 0:
            return np.array(np.nan)
        else:
            open_node = open_list[0]
        open_list.remove(open_node)
        neighbor_nodes = list(np.where(np.isfinite(D[open_node]))[0])
        neighbor_nodes.remove(open_node)
        # only keep the neighbor that hasn't been visited before
        neighbor_keep = []
        for index_neighbor in range(len(neighbor_nodes)):
            if check_connection[neighbor_nodes[index_neighbor]] != 1:
                neighbor_keep.extend([index_neighbor])
        neighbor_nodes_keep = [neighbor_nodes[i] for i in neighbor_keep]
        open_list.extend(neighbor_nodes_keep)
        check_connection[neighbor_nodes_keep] = 1
        if np.sum(check_connection) == N:
            return D
    if np.sum(check_connection) == N:
        return D
    else:
        return np.array(np.nan)

P = pickle.load(open('P_teapot.pkl', 'rb'))

D = knn_distance_init(P,5)

# Try Floyd algorithm
# def Floyd_distance(D):
N = D.shape[0]
d = np.copy(D)
for index_middle in range(N):
    for index_i in range(N):
        d[index_i] = np.min(np.vstack([d[index_i],d[index_i][index_middle]+d[index_middle]]),0)

output = open('dist_teapot.pkl', 'wb')
pickle.dump(d, output)
output.close()

print(np.sum(np.isinf(d)))