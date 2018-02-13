import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# read the data
# using 60000 images for training
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data', validation_size=0)

# the network architecture
inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

### Encoder
## Encoder has 3 convolution+maxpooling layer combination
# 28x28x1
conv1 = tf.layers.conv2d(inputs_, 32, (3,3), padding='same', activation=tf.nn.relu)
maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
# 14x14x32
conv2 = tf.layers.conv2d(maxpool1, 32, (3,3), padding='same', activation=tf.nn.relu)
maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
# 7x7x32
conv3 = tf.layers.conv2d(maxpool2, 16, (3,3), padding='same', activation=tf.nn.relu)
encoded = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
# 4x4x16

### Decoder
## Decoder has 3 unpooling+convolution layer combination
## Implement unpooling by nearest-neighbor resize
# 4x4x16
upsample1 = tf.image.resize_nearest_neighbor(encoded, (7,7))
conv4 = tf.layers.conv2d(upsample1, 16, (3,3), padding='same', activation=tf.nn.relu)
# 7x7x16
upsample2 = tf.image.resize_nearest_neighbor(conv4, (14,14))
conv5 = tf.layers.conv2d(upsample2, 32, (3,3), padding='same', activation=tf.nn.relu)
# 14x14x32
upsample3 = tf.image.resize_nearest_neighbor(conv5, (28,28))
conv6 = tf.layers.conv2d(upsample3, 32, (3,3), padding='same', activation=tf.nn.relu)
# 28x28x32
logits = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)
# 28x28x1

# finally include a sigmoid for display but not for calculating loss
# to keep the symmetric of encoder & decoder
decoded = tf.nn.sigmoid(logits, name='decoded')

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(0.001).minimize(cost)

# start training
epochs = 10
batch_size = 200
# noise_factor
noise_factor = 0.5
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for e in range(epochs):
	for ii in range(mnist.train.num_examples//batch_size):
		batch = mnist.train.next_batch(batch_size)
		imgs = batch[0].reshape((-1, 28, 28, 1))
		# make sure the noisy_image is still in [0,1]
		noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
		noisy_imgs = np.clip(noisy_imgs, 0., 1.)
		batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs,targets_: imgs})
	print("Epoch: {}/{}...".format(e+1, epochs),"Training loss: {:.4f}".format(batch_cost))
# don't closed the session after training
# because we need to test

fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(20,4))
test_index = [random.sample(list(np.where(mnist.test.labels==i)[0]),1)[0] for i in range(10)]
in_imgs = mnist.test.images[test_index]
noisy_imgs = in_imgs + 1*noise_factor * np.random.randn(*in_imgs.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)

reconstructed = sess.run(decoded, feed_dict={inputs_: noisy_imgs.reshape((10, 28, 28, 1))})

for images, row in zip([in_imgs, noisy_imgs, reconstructed], axes):
	for img, ax in zip(images, row):
		ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)
plt.savefig('denoising.png')
plt.show()