import tf_emddistance
import numpy as np
import tensorflow as tf
import random
import pandas as pd
from pyntcloud import PyntCloud

# generate a series of point clouds sampled from a series of circle
def generate_circle_points(n_clouds,n_points,r_min,r_max):
    S = np.zeros([n_clouds,n_points,3])
    np.random.seed(291)
    for index_cloud in range(n_clouds):       
        r = np.random.uniform(r_min,r_max)
        for index_point in range(n_points):
            theta = np.random.uniform(0,2*np.pi)
            S[index_cloud][index_point][0:2] = [r*np.cos(theta),r*np.sin(theta)]
    return S

# define the parameter for generating circle point set
n_clouds = 100
n_points = 500
r_min = 1
r_max = 10

# define the optimization problem
S = tf.placeholder(tf.float32,[None,n_points,3])
x = tf.Variable(tf.truncated_normal([1,n_points,3],mean=0.0,stddev=0.1), dtype = tf.float32)
X = tf.tile(x,[tf.shape(S)[0],1,1])
dist,_,_ = tf_emddistance.emd_distance(S,X)
model_loss = tf.reduce_mean(dist)*10000
avg_r = tf.reduce_mean(tf.norm(x,axis = 2))
train = tf.train.GradientDescentOptimizer(1e-3).minimize(model_loss)

# start to optimize
iter_times = 300
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    S_circle = generate_circle_points(n_clouds,n_points,r_min,r_max)   
    for iters in range(iter_times):
        sess.run(train, {S: S_circle})
        loss,r = sess.run([model_loss,avg_r],{S: S_circle})
        if iters%10 == 0:
            print('iter: ', iters, '\tloss: ', loss, '\tavg_r: ', r)
        
    x_final = sess.run(x,{S: S_circle})
    x_final = np.reshape(x_final,[n_points,3])


# for display visualization
# only in jupyter notebook
red_color = np.tile(np.array([255,0,0]),[n_points,1])
grey_color = np.tile(np.array([255,255,255]),[n_clouds*n_points,1])
colors = (np.vstack([red_color,grey_color])).astype(np.uint8)
P = np.vstack([x_final,np.reshape(S_circle,[n_clouds*n_points,3])])
points = pd.DataFrame(P, columns=['x', 'y', 'z'])
points[['red', 'blue', 'green']] = pd.DataFrame(colors, index=points.index)
cloud = PyntCloud(points)
cloud.plot(point_size = 0.01,lines=[], line_color=[])