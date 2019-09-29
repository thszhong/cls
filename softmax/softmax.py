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
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from sklearn.metrics import accuracy_score

def loss_max_right_class_prob(predictions, y):
	return -predictions[numpy.argmax(y)];

def loss_cross_entropy(predictions, y):
	return -numpy.dot(y, numpy.log(predictions));
	
'''
Softmax classifier
linear classifier 
'''
class Softmax:

	def __init__(self, iter_num = 100000, batch_size = 1):
		self.__iter_num = iter_num;
		self.__batch_size = batch_size;
	
	def train(self, train_X, train_Y):
		X = numpy.c_[train_X, numpy.ones(train_X.shape[0])];
		Y = numpy.copy(train_Y);

		self.L = [];

		#initialize parameters
		self.__weight = numpy.random.rand(X.shape[1], 10) * 2 - 1.0;
		self.__step_len = 1e-3; 

		logging.info("weight:%s" % (self.__weight));

		for iter_index in range(self.__iter_num):
			if iter_index % 1000 == 0:
				logging.info("-----iter:%s-----" % (iter_index));
			if iter_index % 100 == 0:
				l = 0;
				for i in range(0, len(X), 100):
					predictions = self.forward_pass(X[i]);
					#l += loss_max_right_class_prob(predictions, Y[i]); 
					l += loss_cross_entropy(predictions, Y[i]); 
				l /= len(X);
				self.L.append(l);

			sample_index = random.randint(0, len(X) - 1);
			logging.debug("-----select sample %s-----" % (sample_index));

			z = numpy.dot(X[sample_index], self.__weight);
			z = z - numpy.max(z);
			predictions = numpy.exp(z) / numpy.sum(numpy.exp(z));
			dw = self.__step_len * X[sample_index].reshape(-1, 1).dot((predictions - Y[sample_index]).reshape(1, -1));
#			dw = self.__step_len * X[sample_index].reshape(-1, 1).dot(predictions.reshape(1, -1));  
#			dw[range(X.shape[1]), numpy.argmax(Y[sample_index])] -= X[sample_index] * self.__step_len;

			self.__weight -= dw;

			logging.debug("weight:%s" % (self.__weight));
			logging.debug("loss:%s" % (l));
		logging.info("weight:%s" % (self.__weight));
		logging.info("L:%s" % (self.L));
	
	def get_weight(self):
		return self.__weight;
	
	def forward_pass(self, x):
		net = numpy.dot(x, self.__weight);
		net = net - numpy.max(net);
		net = numpy.exp(net) / numpy.sum(numpy.exp(net)); 
		return net;

	def predict(self, x):
		x = numpy.append(x, 1.0);
		return self.forward_pass(x);


def main(): 
	logging.basicConfig(level = logging.INFO,
			format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
			datefmt = '%a, %d %b %Y %H:%M:%S');
			
	logging.info("trainning begin.");

	mnist = read_data_sets('../data/MNIST',one_hot=True)    # MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签

	#load data
	train_X = mnist.train.images                #训练集样本
	validation_X = mnist.validation.images      #验证集样本
	test_X = mnist.test.images                  #测试集样本
	#labels
	train_Y = mnist.train.labels                #训练集标签
	validation_Y = mnist.validation.labels      #验证集标签
	test_Y = mnist.test.labels                  #测试集标签

	classifier = Softmax();
	classifier.train(train_X, train_Y);

	logging.info("trainning end. predict begin.");

	test_predict = numpy.array([]);
	test_right = numpy.array([]);
	for i in range(len(test_X)):
		predict_label = numpy.argmax(classifier.predict(test_X[i]));
		test_predict = numpy.append(test_predict, predict_label);
		right_label = numpy.argmax(test_Y[i]);
		test_right = numpy.append(test_right, right_label);

	logging.info("right:%s, predict:%s" % (test_right, test_predict));
	score = accuracy_score(test_right, test_predict);
	logging.info("The accruacy score is: %s "% (str(score)));


	plt.subplot(3, 1, 1);
	plt.plot(classifier.L);
#	plt.show();

	w = classifier.get_weight();
	w = w[:-1, :];
	w = w.reshape(28, 28, 10);

	w_min, w_max = numpy.min(w), numpy.max(w)

	for i in range(10):
   		plt.subplot(3, 5, 5 + i + 1)

   		# Rescale the weights to be between 0 and 255
   		wimg = 255.0 * (w[:, :, i].squeeze() - w_min) / (w_max - w_min)
   		plt.imshow(wimg.astype('uint8'))
   		plt.axis('off')
   		plt.title(i)

	plt.show();

if __name__ == "__main__":
	main();

