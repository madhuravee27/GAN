import tensorflow as tf
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


def discriminator(image, reuse = False):
	with tf.variable_scope('discriminator') as scope:
		if(reuse):
			tf.get_variable_scope().reuse_variables()
		
		d_w1 = tf.get_variable('d_w1', shape = [5,5,1,32], initializer = tf.contrib.layers.xavier_initializer())
		d_b1 = tf.get_variable('d_b1', shape = [32], initializer = tf.constant_initializer(0))
		d1 = tf.nn.conv2d(input = image, filter = d_w1, strides = [1,1,1,1], padding = 'SAME') + d_b1

		d1 = tf.nn.leaky_relu(features = d1, alpha = 0.2)

		d1 = tf.nn.max_pool(d1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

		d_w2 = tf.get_variable('d_w2', shape = [5,5,32,64], initializer = tf.contrib.layers.xavier_initializer())
		d_b2 = tf.get_variable('d_b2', shape = [64], initializer = tf.constant_initializer(0))
		d2 = tf.nn.conv2d(input = d1, filter = d_w2, strides = [1,1,1,1], padding = 'SAME') + d_b2

		d2 = tf.nn.leaky_relu(features = d2, alpha = 0.2)

		d2 = tf.nn.max_pool(d2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

		d2 = tf.reshape(d2, shape = [-1, 7*7*64])

		d_w3 = tf.get_variable('d_w3', shape = [7*7*64,1024], initializer = tf.contrib.layers.xavier_initializer())
		d_b3 = tf.get_variable('d_b3', shape = [1024], initializer = tf.constant_initializer(0))

		d3 = tf.matmul(d2,d_w3) + d_b3
		d3 = tf.nn.leaky_relu(features = d3, alpha = 0.2)

		d_w4 = tf.get_variable('d_w4', shape = [1024,1], initializer = tf.contrib.layers.xavier_initializer())
		d_b4 = tf.get_variable('d_b4', shape = [1], initializer = tf.constant_initializer(0))

		d4 = tf.matmul(d3,d_w4) + d_b4
		#d4 = tf.nn.sigmoid(d4)
		return d4
		'''
		d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.leaky_relu(d1, alpha = 0.2)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.leaky_relu(d2, alpha = 0.2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.contrib.layers.xavier_initializer())
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.contrib.layers.xavier_initializer())
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4
        '''

def generator(z, batch_size):
	with tf.variable_scope('generator') as scope:
		
		input = z
		g_w1 = tf.get_variable('g_w1', shape = [100,1024], initializer=tf.contrib.layers.xavier_initializer())
		g_b1 = tf.get_variable('g_b1', shape = [1024], initializer = tf.constant_initializer(0))
		g1 = tf.matmul(input, g_w1) + g_b1
		g1 = tf.contrib.layers.batch_norm(inputs = g1, decay = 0.9, center=True, scale=True, is_training=True, scope = 'g_bn1')
		g1 = tf.nn.leaky_relu(features = g1, alpha = 0.2)

		g_w2 = tf.get_variable('g_w2', shape = [1024,7*7*64], initializer=tf.contrib.layers.xavier_initializer())
		g_b2 = tf.get_variable('g_b2', shape = [7*7*64], initializer = tf.constant_initializer(0))
		g2 = tf.matmul(g1, g_w2) + g_b2
		g2 = tf.contrib.layers.batch_norm(inputs = g2, decay = 0.9, center=True, scale=True, is_training=True, scope = 'g_bn2')
		g2 = tf.nn.leaky_relu(features = g2, alpha = 0.2)
		g2 = tf.reshape(g2, [batch_size,7,7,64])

		g_w3 = tf.get_variable('g_w3', shape = [5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
		g_b3 = tf.get_variable('g_b3', shape = [32], initializer = tf.constant_initializer(0))
		g3 = tf.nn.conv2d_transpose(value = g2, filter = g_w3, output_shape = [batch_size, 14, 14, 32], strides = [1,2,2,1], padding = 'SAME') + g_b3
		g3 = tf.contrib.layers.batch_norm(inputs = g3, decay = 0.9, center=True, scale=True, is_training=True, scope = 'g_bn3')
		g3 = tf.nn.leaky_relu(features = g3, alpha = 0.2)

		g_w4 = tf.get_variable('g_w4', shape = [5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
		g_b4 = tf.get_variable('g_b4', shape = [1], initializer = tf.constant_initializer(0))
		g4 = tf.nn.conv2d_transpose(value = g3, filter = g_w4, output_shape = [batch_size, 28, 28, 1], strides = [1,2,2,1], padding = 'SAME') + g_b4
		#generator_conv4_output = tf.contrib.layers.batch_norm(generator_conv4_output, decay = 0.9)
		g4 = tf.nn.sigmoid(g4)

		return g4

		'''
		g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g1 = tf.matmul(z, g_w1) + g_b1
		g1 = tf.reshape(g1, [-1, 56, 56, 1])
		g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
		g1 = tf.nn.relu(g1)

	    # Generate 50 features
		g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
		g2 = g2 + g_b2
		g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')
		g2 = tf.nn.relu(g2)
		g2 = tf.image.resize_images(g2, [56, 56])

	    # Generate 25 features
		g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
		g3 = g3 + g_b3
		g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')
		g3 = tf.nn.relu(g3)
		g3 = tf.image.resize_images(g3, [56, 56])

	    # Final convolution with one output channel
		g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
		g4 = g4 + g_b4
		g4 = tf.sigmoid(g4)

	    # Dimensions of g4: batch_size x 28 x 28 x 1
		return g4
		'''



from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

tf.reset_default_graph() 

noise_size = 100
batch_size = 50
ip_g = tf.placeholder(dtype = tf.float32, shape = [None, noise_size], name = 'ip_g')
ip_d = tf.placeholder(dtype = tf.float32, shape = [None, 28,28,1], name = 'ip_d')
dReal = discriminator(ip_d)
gFake = generator(ip_g, batch_size)
dFake = discriminator(gFake, reuse = True)

d_Real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dReal, labels = tf.ones_like(dReal)))
d_Fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dFake, labels = tf.zeros_like(dFake)))
d_Total_loss = d_Real_loss + d_Fake_loss

g_Fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dFake, labels = tf.ones_like(dFake)))

trainable_var = tf.trainable_variables()

d_var = []
g_var = []

for var in trainable_var:
	if('g_' in var.name):
		g_var.append(var)
	if('d_' in var.name):
		d_var.append(var)

#d_real_Train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss = d_Real_loss, var_list = d_var)
#d_fake_Train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss = d_Fake_loss, var_list = d_Fake_loss)
d_Train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss = d_Total_loss, var_list = d_var)
g_Train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss = g_Fake_loss, var_list = g_var)

tf.get_variable_scope().reuse_variables()
sess = tf.Session()

tf.summary.scalar(name = 'Generator loss', tensor = g_Fake_loss)
tf.summary.scalar(name = 'Discriminator real loss', tensor = d_Real_loss)
tf.summary.scalar(name = 'Discriminator fake loss', tensor = d_Fake_loss)
tensorboard_generated_images = generator(ip_g, batch_size)
tf.summary.image(name = 'Generated images', tensor = tensorboard_generated_images)
merged_data = tf.summary.merge_all()
logdir = 'tensorboard/'+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+'/'
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(300):
	ip_fake = np.random.normal(0,1,size = [batch_size,noise_size])
	ip_real = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
	#_, __, discriminator_real_loss, discriminator_fake_loss = sess.run([d_real_Train,d_fake_Train,d_Real_loss, d_Fake_loss], feed_dict = {ip_d:ip_real, ip_g:ip_fake})
	_, dRealLoss, dFakeLoss = sess.run([d_Train,d_Real_loss, d_Fake_loss], feed_dict = {ip_d:ip_real, ip_g:ip_fake})
	if(i%100 == 0):
		print(f"Discriminator real loss:{dRealLoss} \t Discriminator fake loss:{dFakeLoss}")


for i in range(100000):
	ip_fake = np.random.normal(0,1,size = [batch_size,noise_size])
	ip_real = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
	#_, __, discriminator_real_loss, discriminator_fake_loss = sess.run([d_real_Train,d_fake_Train,d_Real_loss, d_Fake_loss], feed_dict = {ip_d:ip_real, ip_g:ip_fake})
	_, dRealLoss, dFakeLoss = sess.run([d_Train,d_Real_loss, d_Fake_loss], feed_dict = {ip_d:ip_real, ip_g:ip_fake})

	ip_fake = np.random.normal(0,1,size = [batch_size,noise_size])
	_ = sess.run(g_Train, feed_dict = {ip_g:ip_fake})
	
	if(i%10 == 0):
		ip_fake = np.random.normal(0,1,size = [batch_size,noise_size])
		summary = sess.run(merged_data, feed_dict = {ip_d:ip_real, ip_g:ip_fake})
		writer.add_summary(summary = summary,global_step = i)
'''

z_dimensions = 100
batch_size = 50

z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
# z_placeholder is for feeding input noise to the generator

x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator

Gz = generator(z_placeholder, batch_size, z_dimensions)
# Gz holds the generated images

Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz, reuse=True)
# Dg will hold discriminator prediction probabilities for generated images

# Define losses
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

# Define variable lists
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Define the optimizers
# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0002, beta1 = 0.5).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0002, beta1 = 0.5).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0002, beta1 = 0.5).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

sess = tf.Session()

# Send summary statistics to TensorBoard
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())

# Pre-train discriminator
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

# Train generator and discriminator together
for i in range(100000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        writer.add_summary(summary, i)
'''