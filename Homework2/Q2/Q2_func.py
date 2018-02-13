# function to compute the area of an triangle
# input is a 3*3 matrix, with each row a vertices in 3d space
def triangles_area(vertices):
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    cross = np.cross(v1,v2)
    area = np.linalg.norm(cross)/2
    return area

# function to compute the area of an triangle
# input is a 3*3 matrix, with each row a vertices in 3d space
# using Heron's formula 
def triangles_area_Heron(vertices):
    a = np.linalg.norm(vertices[2]-vertices[0])
    b=np.linalg.norm(vertices[2]-vertices[1])
    c=np.linalg.norm(vertices[1]-vertices[0])
    p = (a+b+c)/2
    area = np.sqrt(p*(p-a)*(p-b)*(p-c))
    return area

# function that return all the triangle faces' area
def triangle_faces_area(vertices,faces):
    n_faces = faces.shape[0]
    area = np.zeros(n_faces)
    for index_face in range(n_faces):
        area[index_face] = triangles_area_Heron(vertices[faces[index_face]])
    return area

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

# Given a initial distance matrix D with inf (meaning no connection)
# Return a new distance matrix d by Dijkstra algorithm
# Compute the distance by adding the edge length given in D
# Just too slow

def Dijkstra_distance(D):
    N = D.shape[0]
    d = np.copy(D)
    # for each point i explore the distance
    for index_point in range(N):
        print(index_point)
        close_list = np.zeros(N)
        close_list[index_point] = 1
        open_list = list(np.where(np.isfinite(d[index_point]))[0])
        open_list_distance = list(d[index_point][open_list])
        for iters in range(N-1):
            # find the node with smallest distance
            # name it open_node
            # remove from open_list, add to close_list
            # print(open_list)
            nearest_index = np.argmin(np.array(open_list_distance))
            open_node = open_list[nearest_index]
            open_list.remove(open_node)
            open_list_distance.remove(open_list_distance[nearest_index])
            close_list[open_node] = 1
            # find the neighbors of open_list
            neighbor_nodes = list(np.where(np.isfinite(d[open_node]))[0])
            neighbor_nodes.remove(open_node)
            # only keep the neighbors that are not in close_list
            # actually we should also discard those in open_list
            neighbor_keep = []
            for index_neighbor in range(len(neighbor_nodes)):
                if close_list[neighbor_nodes[index_neighbor]] != 1:
                    neighbor_keep.extend([index_neighbor])
            neighbor_nodes_keep = [neighbor_nodes[i] for i in neighbor_keep]
            for index_neighbor in neighbor_nodes_keep:
                if d[index_point][open_node]+d[open_node][index_neighbor] < d[index_point][index_neighbor]:
                    d[index_point][index_neighbor] = d[index_point][open_node]+d[open_node][index_neighbor]
                    open_list.extend([index_neighbor])
                    open_list_distance.extend([d[index_point][index_neighbor]])
            if np.sum(close_list) == N:
                continue
        # connect point[index_point] with all the other points
        d[:,index_point] = d[index_point,:]

    return d

# Floyd algorithm to build distance matrix
def Floyd_distance(D):
    N = D.shape[0]
    d = np.copy(D)
    for index_middle in range(N):
        for index_i in range(N):
            d[index_i] = np.min(np.vstack([d[index_i],d[index_i][index_middle]+d[index_middle]]),0)
    return d