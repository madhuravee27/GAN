import tensorflow as tf
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt


def discriminator(image, reuse = False):
	with tf.variable_scope('discriminator') as scope:
		if(reuse):
			tf.get_variable_scope().reuse_variables()
		
		input_image = image
		discriminator_weight1 = tf.get_variable('discriminator_weight1', shape = [5,5,3,64], initializer = tf.contrib.layers.xavier_initializer())
		discriminator_bias1 = tf.get_variable('discriminator_bias1', shape = [64], initializer = tf.constant_initializer(0))
		conv1_output = tf.nn.conv2d(input = input_image, filter = discriminator_weight1, strides = [1,1,1,1], padding = 'SAME') + discriminator_bias1

		maxpool1_input = tf.nn.leaky_relu(features = conv1_output, alpha = 0.2)

		conv2_input = tf.nn.max_pool(maxpool1_input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

		discriminator_weight2 = tf.get_variable('discriminator_weight2', shape = [5,5,64,128], initializer = tf.contrib.layers.xavier_initializer())
		discriminator_bias2 = tf.get_variable('discriminator_bias2', shape = [128], initializer = tf.constant_initializer(0))
		conv2_output = tf.nn.conv2d(input = conv2_input, filter = discriminator_weight2, strides = [1,1,1,1], padding = 'SAME') + discriminator_bias2
		conv2_output = tf.contrib.layers.batch_norm(inputs = conv2_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'discriminator_bn1')

		maxpool2_input = tf.nn.leaky_relu(features = conv2_output, alpha = 0.2)

		conv3_input = tf.nn.max_pool(maxpool2_input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

		discriminator_weight3 = tf.get_variable('discriminator_weight3', shape = [5,5,128,256], initializer = tf.contrib.layers.xavier_initializer())
		discriminator_bias3 = tf.get_variable('discriminator_bias3', shape = [256], initializer = tf.constant_initializer(0))
		conv3_output = tf.nn.conv2d(input = conv3_input, filter = discriminator_weight3, strides = [1,1,1,1], padding = 'SAME') + discriminator_bias3
		conv3_output = tf.contrib.layers.batch_norm(inputs = conv3_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'discriminator_bn2')

		maxpool3_input = tf.nn.leaky_relu(features = conv3_output, alpha = 0.2)

		conv4_input = tf.nn.max_pool(maxpool3_input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

		discriminator_weight4 = tf.get_variable('discriminator_weight4', shape = [5,5,256, 512], initializer = tf.contrib.layers.xavier_initializer())
		discriminator_bias4 = tf.get_variable('discriminator_bias4', shape = [512], initializer = tf.constant_initializer(0))
		conv4_output = tf.nn.conv2d(input = conv4_input, filter = discriminator_weight4, strides = [1,1,1,1], padding = 'SAME') + discriminator_bias4
		conv4_output = tf.contrib.layers.batch_norm(inputs = conv4_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'discriminator_bn3')

		maxpool4_input = tf.nn.leaky_relu(features = conv2_output, alpha = 0.2)

		fc1_input = tf.nn.max_pool(maxpool4_input, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

		fc1_input_reshape = tf.reshape(fc1_input, shape = [-1, 2*2*512])

		discriminator_weight5 = tf.get_variable('discriminator_weight5', shape = [2*2*512,2048], initializer = tf.contrib.layers.xavier_initializer())
		discriminator_bias5 = tf.get_variable('discriminator_bias5', shape = [2048], initializer = tf.constant_initializer(0))
		fc1_output = tf.matmul(fc1_input_reshape,discriminator_weight5) + discriminator_bias5
		fc1_output = tf.contrib.layers.batch_norm(inputs = fc1_output, decay = 0.9, center=True, scale=True, is_training=True, scope = 'discriminator_bn4')

		fc2_input = tf.nn.leaky_relu(features = fc1_output, alpha = 0.2)

		discriminator_weight6 = tf.get_variable('discriminator_weight6', shape = [2048,1], initializer = tf.contrib.layers.xavier_initializer())
		discriminator_bias6 = tf.get_variable('discriminator_bias6', shape = [1], initializer = tf.constant_initializer(0))

		fc2_output = tf.matmul(fc2_input,discriminator_weight6) + discriminator_bias4
		fc2_output = tf.nn.sigmoid(fc2_output)
		return fc2_output

