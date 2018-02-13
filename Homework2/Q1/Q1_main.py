import numpy as np
import random
import scipy
import pickle
import os
import matplotlib.pyplot as plt
from sklearn import manifold, datasets, decomposition, ensemble, random_projection
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def plot_embedding_2d(X, Y, title=None):
    color_space = ['k','grey','r','y','g','c','b','m','orange','navy'] 
    fig = plt.figure(figsize=[8,8])
    ax = fig.add_subplot(1, 1, 1)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1],str(Y[i]),
                 color=color_space[Y[i]],
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)    
    x_range = np.max(np.abs(X))
    plt.xlim((-x_range, x_range))
    plt.ylim((-x_range, x_range))
    plt.savefig(title+".png")
    plt.show()

# generate the data
if os.path.isfile('mnist_X.pickle') and os.path.isfile('mnist_Y.pickle'):
    mnist_X = pickle.load(open('mnist_X.pickle', 'rb'))
    mnist_Y = pickle.load(open('mnist_Y.pickle', 'rb'))
else:
    mnist = input_data.read_data_sets("../MNIST_data/", reshape=True)
    # notice that here we don't shuffle the data so the samples with same label are together
    sample_per_label = 1000
    random.seed(291)
    index_sampling = []
    for target_label in range(10):
        target_label_position = np.where(mnist.train.labels==target_label)[0]
        target_label_sampling = random.sample(list(target_label_position), sample_per_label)
        index_sampling.extend(target_label_sampling)
    mnist_X = mnist.train.images[index_sampling,:]
    mnist_Y = mnist.train.labels[index_sampling]
if os.path.isfile('mnist_Dist.pickle'):
    mnist_Dist = pickle.load(open('mnist_Dist.pickle', 'rb'))
else:
    mnist_Dist = scipy.spatial.distance.cdist(mnist_X,mnist_X)

# build the graph
# generate the embedding space point automatically and
# calculate the difference
N = 10000
D = tf.placeholder(tf.float32,[N,N])
D_vec = tf.reshape(D,[-1,1])
x = tf.Variable(tf.truncated_normal([N, 2],mean=0.0,stddev=0.1), dtype = tf.float32)
xsq = tf.reduce_sum(x*x, 1)
xsq_vec = tf.reshape(xsq,[-1,1])
dsq = xsq_vec - 2*tf.matmul(x,tf.transpose(x)) + tf.transpose(xsq_vec)
d_vec = tf.sqrt(tf.maximum(dsq,1e-20))
d_vec = tf.reshape(d_vec,[-1,1])
model_loss = tf.reduce_sum(tf.square(D_vec-d_vec))/2
avg_dist = model_loss/(N*(N-1)/2)

train = tf.train.GradientDescentOptimizer(1e-5).minimize(model_loss)
# train = tf.train.AdamOptimizer(1e-3).minimize(model_loss)

# begin the training
iter_times = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x_init,d_init,model_loss_init = sess.run([x,d_vec,model_loss],{D: mnist_Dist[:N,:N]})
    for iters in range(iter_times):
        sess.run(train, {D: mnist_Dist[:N,:N]})
        loss,dist = sess.run([model_loss,avg_dist],{D: mnist_Dist[:N,:N]})
        if iters%10 == 0:
            print('iter: ', iters, ' loss: ', loss, " avg_distance: ", dist)        
    x_final = sess.run(x,{D: mnist_Dist[:N,:N]})

# visualization
plot_embedding_2d(x_init, mnist_Y, 'initial')
plot_embedding_2d(x_final, mnist_Y, 'final')