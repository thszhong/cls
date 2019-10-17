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

def main(): 
	logging.basicConfig(level = logging.INFO,
			format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
			datefmt = '%a, %d %b %Y %H:%M:%S');
			
	logging.info("trainning begin.");

	mnist = read_data_sets('../data/MNIST',one_hot=True)    # MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签

	x = tf.placeholder(tf.float32, [None, 784]);
	w = tf.Variable(tf.zeros([784, 10]));
	b = tf.Variable(tf.zeros([10]));
	y = tf.matmul(x, w) + b;

	y_ = tf.placeholder(tf.float32, [None, 10]);

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_));
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy);

	sess = tf.InteractiveSession();
	tf.global_variables_initializer().run();
	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100);
		sess.run(train_step, feed_dict = {x : batch_xs, y_ : batch_ys});

	logging.info("trainning end.");
	logging.info("testing begin.");

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1));
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
	print(sess.run(accuracy, feed_dict = {x : mnist.test.images, y_:mnist.test.labels}));

	logging.info("testing end.");

if __name__ == "__main__":
	main();

