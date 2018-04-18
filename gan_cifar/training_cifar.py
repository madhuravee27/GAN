import tensorflow as tf
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from generator_cifar import generator
from discriminator_cifar import discriminator
import datetime

from sklearn.utils import shuffle

from keras.datasets import cifar10
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()

cifar_data = np.concatenate((train_features, test_features))

cifar_data = cifar_data.astype('float32') / 255

#print(cifar_data.shape)

cifar = shuffle(cifar_data)
#print (cifar.shape)

#cifar.reshape([None, ])

#print(cifar)

tf.reset_default_graph() 

noise_size = 100
batch_size = 100
training_epoch = 500
input_generator = tf.placeholder(dtype = tf.float32, shape = [None, noise_size], name = 'input_generator')
input_discriminator = tf.placeholder(dtype = tf.float32, shape = [None, 32,32,3], name = 'input_discriminator')
discriminatorReal = discriminator(input_discriminator, reuse = False)
generatorFake = generator(input_generator, batch_size, reuse = False)
discriminatorFake = discriminator(generatorFake, reuse = True)

discriminator_Real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminatorReal, labels = tf.ones_like(discriminatorReal)))
discriminator_Fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminatorFake, labels = tf.zeros_like(discriminatorFake)))
discriminator_Total_loss = discriminator_Real_loss + discriminator_Fake_loss

generator_Fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminatorFake, labels = tf.ones_like(discriminatorFake)))

trainable_variables = tf.trainable_variables()

discriminator_variables = []
generator_variables = []

for var in trainable_variables:
	if('generator_' in var.name):
		generator_variables.append(var)
	if('discriminator_' in var.name):
		discriminator_variables.append(var)

#d_real_Train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss = d_Real_loss, var_list = d_var)
#d_fake_Train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss = d_Fake_loss, var_list = d_Fake_loss)
discriminator_Train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss = discriminator_Total_loss, var_list = discriminator_variables)
generator_Train = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss = generator_Fake_loss, var_list = generator_variables)

#tf.get_variable_scope().reuse_variables()
sess = tf.Session()

tf.summary.scalar(name = 'Generator loss', tensor = generator_Fake_loss)
tf.summary.scalar(name = 'Discriminator real loss', tensor = discriminator_Real_loss)
tf.summary.scalar(name = 'Discriminator fake loss', tensor = discriminator_Fake_loss)
tensorboard_generated_images = generator(input_generator, batch_size, reuse = True)
tf.summary.image(name = 'Generated images', tensor = tensorboard_generated_images, max_outputs = 5)
merged_data = tf.summary.merge_all()
logdir = 'tensorboard_cifar/'+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+'/'
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())

for epoch in range(3):
	for i in range(cifar.shape[0]//batch_size):
		input_fake = np.random.normal(0,1,size = [batch_size,noise_size])
		input_real = cifar[i*batch_size : (i+1)*batch_size]
		discriminatorTrain, discriminatorRealLoss, discriminatorFakeLoss = sess.run([discriminator_Train,discriminator_Real_loss, discriminator_Fake_loss], feed_dict = {input_discriminator:input_real, input_generator:input_fake})


count = 1
cifar = shuffle(cifar)
for epoch in range(training_epoch):
	for i in range(cifar.shape[0]//batch_size):
		input_fake = np.random.normal(0,1,size = [batch_size,noise_size])
		input_real = cifar[i*batch_size : (i+1)*batch_size]
		discriminatorTrain, discriminatorRealLoss, discriminatorFakeLoss = sess.run([discriminator_Train,discriminator_Real_loss, discriminator_Fake_loss], feed_dict = {input_discriminator:input_real, input_generator:input_fake})
		
		input_fake = np.random.normal(0,1,size = [batch_size,noise_size])
		generatorTrain = sess.run(generator_Train, feed_dict = {input_generator:input_fake})
		
		input_fake = np.random.normal(0,1,size = [batch_size,noise_size])
	summary = sess.run(merged_data, feed_dict = {input_discriminator:input_real, input_generator:input_fake})
	writer.add_summary(summary = summary,global_step = epoch)
	cifar = shuffle(cifar)