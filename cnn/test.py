from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tempfile
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow as tf
FLAGS = None
def weight_variable(shape):
    init=tf.truncated_normal(shape=shape,stddev=0.1,mean=1.)
    return tf.Variable(init)
def bias_variable(shape):
    init=tf.constant(0.1,shape=shape)
    return tf.Variable(init)
def conv2d(x,w):
    return tf.nn.conv2d(x,w,[1,1,1,1],padding="SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def deepnn(x):
    with tf.name_scope('reshape'):
        x_image=tf.reshape(x,[-1,28,28,1])
    #第一层卷积和池化
    with tf.name_scope('conv1'):
        #输入为1张图片 卷积核为5*5 生成32个特征图
        w_conv1=weight_variable([5,5,1,32])
        b_conv1=bias_variable([32])
        h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
    with tf.name_scope('pool1'):
        h_pool1=max_pool_2x2(h_conv1)
    #第二层卷积和池化
    with tf.name_scope("conv2"):
        #输入为32张特征图，卷积核为5*5 输出64张特征图
        w_conv2=weight_variable([5,5,32,64])
        b_conv2=bias_variable([64])
        h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
    with tf.name_scope("pool2"):
        h_pool2=max_pool_2x2(h_conv2)
    #第一层全连接层，将特征图展开为特征向量，与1024个节点连接
    with tf.name_scope("fc1"):
        w_fc1=weight_variable([7*7*64,1024])
        b_fc1=bias_variable([1024])
        h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
    #dropout层，训练时随机让某些隐含层节点权重不工作
    with tf.name_scope("dropout1"):
        keep_prob=tf.placeholder(tf.float32)
        h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
    #第二个全连接层，连接1024个节点，输出one-hot预测
    with tf.name_scope("fc2"):
        w_fc2=weight_variable([1024,10])
        b_fc2=bias_variable([10])
        h_fc2=tf.matmul(h_fc1_drop,w_fc2)+b_fc2
    return h_fc2,keep_prob

def main(_):

  mnist = read_data_sets('../data/MNIST',one_hot=True)    # MNIST_data指的是存放数据的文件夹路径，one_hot=True 为采用one_hot的编码方式编码标签
  #设置输入变量
  x=tf.placeholder(dtype=tf.float32,shape=[None,784])
  #设置输出变量
  y_real=tf.placeholder(dtype=tf.float32,shape=[None,10])
  #实例化网络
  y_pre,keep_prob=deepnn(x)
  #设置损失函数
  with tf.name_scope("loss"):
      cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=y_pre,labels=y_real)
      loss=tf.reduce_mean(cross_entropy)
  #设置优化器
  with tf.name_scope("adam_optimizer"):
      train_step=tf.train.AdamOptimizer(1e-4).minimize(loss)
  #计算正确率：
  with tf.name_scope("accuracy"):
      correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_real, 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  #将神经网络图模型保存
  graph_location=tempfile.mkdtemp()
  print('saving graph to %s'%graph_location)
  train_writer=tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())
  #将训练的网络保存下来
  saver=tf.train.Saver()
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(5000):
          batch=mnist.train.next_batch(50)
          if i%100==0:
              train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_real: batch[1], keep_prob: 1.0})
              print('step %d, training accuracy %g' % (i, train_accuracy))
          sess.run(train_step,feed_dict={x: batch[0], y_real: batch[1], keep_prob: 0.5})
      #在测试集上进行测试
      test_accuracy = 0
      for i in range(200):
          batch = mnist.test.next_batch(50)
          test_accuracy += accuracy.eval(feed_dict={x: batch[0], y_real: batch[1], keep_prob: 1.0}) / 200;

      print('test accuracy %g' % test_accuracy)
      save_path = saver.save(sess, "mnist_cnn_model.ckpt")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='./',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
