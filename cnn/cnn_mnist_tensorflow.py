'''
softmax classifier for mnist  

created on 2019.9.28
author: vince
'''
import math
import logging
import numpy  
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn.metrics import accuracy_score

def weight_bais_variable(shape):
	init = tf.random.truncated_normal(shape = shape, stddev = 0.01);
	return tf.Variable(init);

def bais_variable(shape):
	init = tf.constant(0.1, shape=shape);
	return tf.Variable(init);

def conv2d(x, w):
	return tf.nn.conv2d(x, w, [1, 1, 1, 1], padding = "SAME");

def max_pool_2x2(x):
	return tf.nn.max_pool2d(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME");

def cnn(x, rate):
	with tf.name_scope('reshape'):
		x_image = tf.reshape(x, [-1, 28, 28, 1]);
	
	#first layer, conv & pool 
	with tf.name_scope('conv1'):
		w_conv1 = weight_bais_variable([5, 5, 1, 32]);
		b_conv1 = bais_variable([32]);
		h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1); #28 * 28 * 32
	with tf.name_scope('pool1'):
		h_pool1 = max_pool_2x2(h_conv1); #14 * 14 * 32
	
	#second layer, conv & pool 
	with tf.name_scope('conv2'):
		w_conv2 = weight_bais_variable([5, 5, 32, 64]);
		b_conv2 = bais_variable([64]);
		h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2); #14 * 14 * 64 
	with tf.name_scope('pool2'):
		h_pool2 = max_pool_2x2(h_conv2);  #7 * 7 * 64 

	#first full connect layer, feature graph -> feature vector 
	with tf.name_scope('fc1'):
		w_fc1 = weight_bais_variable([7 * 7 * 64, 1024]);
		b_fc1 = bais_variable([1024]);
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64]);
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1);
	with tf.name_scope("dropout1"):
		h_fc1_drop = tf.nn.dropout(h_fc1, rate);

	#second full connect layer, 
	with tf.name_scope('fc2'):
		w_fc2 = weight_bais_variable([1024, 10]);
		b_fc2 = bais_variable([10]);
		#h_fc2 = tf.matmul(h_fc1_drop, w_fc2) + b_fc2;
		h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2;
	return h_fc2;


def main(): 
	logging.basicConfig(level = logging.INFO,
			format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
			datefmt = '%a, %d %b %Y %H:%M:%S');

	mnist = read_data_sets('../data/MNIST',one_hot=True)    # MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签

	x = tf.placeholder(tf.float32, [None, 784]);
	y_real = tf.placeholder(tf.float32, [None, 10]);
	rate = tf.placeholder(tf.float32);

	y_pre = cnn(x, rate);

	sess = tf.InteractiveSession();
	sess.run(tf.global_variables_initializer());

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_pre, labels = y_real));
	train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss);

	correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_real, 1));
	prediction_op= tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
	for _ in range(300):
		batch_xs, batch_ys = mnist.train.next_batch(128);
		sess.run(train_op, feed_dict = {x : batch_xs, y_real : batch_ys, rate: 0.5});
		if _ % 10 == 0: 
			accuracy = sess.run(prediction_op, feed_dict = {x : mnist.test.images, y_real : mnist.test.labels, rate: 0.0 });
			logging.info("%s : %s" % (_, accuracy));

if __name__ == "__main__":
	main();

