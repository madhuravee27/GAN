import tensorflow as tf
import tensorflow as tf
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

def generator(noise, batch_size, reuse = False):
	with tf.variable_scope('generator') as scope:
		if(reuse):
			tf.get_variable_scope().reuse_variables()
		
		input_vector = noise
		generator_weight1 = tf.get_variable('generator_weight1', shape = [100,2048], initializer=tf.contrib.layers.xavier_initializer())
		generator_bias1 = tf.get_variable('generator_bias1', shape = [2048], initializer = tf.constant_initializer(0))
		fc1_output = tf.matmul(input_vector, generator_weight1) + generator_bias1
		fc1_output_bn = tf.contrib.layers.batch_norm(inputs = fc1_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'generator_bn1')
		fc2_input = tf.nn.leaky_relu(features = fc1_output_bn, alpha = 0.2)

		generator_weight2 = tf.get_variable('generator_weight2', shape = [2048,2*2*512], initializer=tf.contrib.layers.xavier_initializer())
		generator_bias2 = tf.get_variable('generator_bias2', shape = [2*2*512], initializer = tf.constant_initializer(0))
		fc2_output = tf.matmul(fc2_input, generator_weight2) + generator_bias2
		fc2_output_bn = tf.contrib.layers.batch_norm(inputs = fc2_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'generator_bn2')
		deconv1_input = tf.nn.leaky_relu(features = fc2_output_bn, alpha = 0.2)
		deconv1_input_reshape = tf.reshape(deconv1_input, [batch_size,2,2,512])

		generator_weight3 = tf.get_variable('generator_weight3', shape = [5, 5, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
		generator_bias3 = tf.get_variable('generator_bias3', shape = [256], initializer = tf.constant_initializer(0))
		deconv1_output = tf.nn.conv2d_transpose(value = deconv1_input_reshape, filter = generator_weight3, output_shape = [batch_size, 4, 4, 256], strides = [1,2,2,1], padding = 'SAME') + generator_bias3
		deconv1_output_bn = tf.contrib.layers.batch_norm(inputs = deconv1_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'generator_bn3')
		deconv2_input = tf.nn.leaky_relu(features = deconv1_output_bn, alpha = 0.2)

		generator_weight4 = tf.get_variable('generator_weight4', shape = [5, 5, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
		generator_bias4 = tf.get_variable('generator_bias4', shape = [128], initializer = tf.constant_initializer(0))
		deconv2_output = tf.nn.conv2d_transpose(value = deconv2_input, filter = generator_weight4, output_shape = [batch_size, 8, 8, 128], strides = [1,2,2,1], padding = 'SAME') + generator_bias4
		deconv2_output_bn = tf.contrib.layers.batch_norm(deconv2_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'generator_bn4')
		deconv3_input = tf.nn.leaky_relu(features = deconv2_output_bn, alpha = 0.2)

		generator_weight5 = tf.get_variable('generator_weight5', shape = [5, 5, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
		generator_bias5 = tf.get_variable('generator_bias5', shape = [64], initializer = tf.constant_initializer(0))
		deconv3_output = tf.nn.conv2d_transpose(value = deconv3_input, filter = generator_weight5, output_shape = [batch_size, 16, 16, 64], strides = [1,2,2,1], padding = 'SAME') + generator_bias5
		deconv3_output_bn = tf.contrib.layers.batch_norm(inputs = deconv3_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'generator_bn5')
		deconv4_input = tf.nn.leaky_relu(features = deconv3_output_bn, alpha = 0.2)

		generator_weight6 = tf.get_variable('generator_weight6', shape = [5, 5, 3, 64], initializer=tf.contrib.layers.xavier_initializer())
		generator_bias6 = tf.get_variable('generator_bias6', shape = [3], initializer = tf.constant_initializer(0))
		deconv4_output = tf.nn.conv2d_transpose(value = deconv4_input, filter = generator_weight6, output_shape = [batch_size, 32, 32, 3], strides = [1,2,2,1], padding = 'SAME') + generator_bias6
		#generator_conv4_output = tf.contrib.layers.batch_norm(generator_conv4_output, decay = 0.9)
		generated_image = tf.nn.tanh(deconv4_output)

		return generated_image

'''
sess = tf.Session()
noise_dimension = 100
noise_value = tf.placeholder(tf.float32, [None, noise_dimension])
generator_output = generator(noise_value,1)
test_input = np.random.normal(-1, 1, [1,noise_dimension])
sess.run(tf.global_variables_initializer())
test = (sess.run(generator_output, feed_dict = {noise_value: test_input}))
#print(test)
plt.imshow(test.squeeze(), cmap = 'gray_r')
plt.show()
'''


