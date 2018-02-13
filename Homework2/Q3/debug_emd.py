import tf_emddistance
import numpy as np
import tensorflow as tf

'''
Toy example

Basically pass two tensors of (BATCH_SIZE, CLOUD_SIZE, 3) into the loss function


Expected output:
loss should be equal to loss3

loss2 should be 0
'''


pt_gt=np.ndarray((3,64,3))
pt_gt[0,:]=np.reshape(np.loadtxt('latent_0.txt'), (64,3))
pt_gt[1,:]=np.reshape(np.loadtxt('latent_0.txt'), (64,3))
pt_gt[2,:]=np.reshape(np.loadtxt('latent_0.txt'), (64,3))
pt_gt=tf.convert_to_tensor(pt_gt, dtype=tf.float32)
x_res=np.ndarray((3,64,3))
x_res[0,:]=np.reshape(np.loadtxt('latent_1000.txt'), (64,3))
x_res[1,:]=np.reshape(np.loadtxt('latent_0.txt'), (64,3))
x_res[2,:]=np.reshape(np.loadtxt('latent_1000.txt'), (64,3))
x_res=tf.convert_to_tensor(x_res, dtype=tf.float32)
dist,idx1,idx2=tf_emddistance.emd_distance(pt_gt,x_res)
loss=tf.reduce_mean(dist[0,:])*10000
loss2=tf.reduce_mean(dist[1,:])*10000
loss3=tf.reduce_mean(dist[2,:])*10000
#loss2=0
sess=tf.Session()
print(sess.run([loss, loss2, loss3]))
