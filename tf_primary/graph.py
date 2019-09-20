""" Test tensorflow graph && session """
import sys
import os
import urllib
import math
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', '', 'program mode')
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    # dataset file name
DATASET_LIST = {
    'train_imgs' : 'train-images-idx3-ubyte.gz',
    'train_labels' : 'train-labels-idx1-ubyte.gz',
    'test_imgs' : 't10k-images-idx3-ubyte.gz',
    'test_labels' : 't10k-labels-idx1-ubyte.gz'}

def download_data(base_path, download_folder):
    """ Download mnist dataset, unless it's already here

    Parameters
    ----------
    base_path : str
        download absolute path
    download_folder : str
        download folder name
    """
    # download mnist dataset
    download_path = os.path.join(base_path, download_folder)
    if not os.path.exists(download_path):
        os.mkdir(download_path)
    for key in DATASET_LIST:
        filepath = os.path.join(download_path, DATASET_LIST[key])
        if not os.path.exists(filepath):
            filepath, _ = urllib.request.urlretrieve(SOURCE_URL + DATASET_LIST[key], filepath)
            statinfo = os.stat(filepath)
            print('Successfully downloaded', DATASET_LIST[key], statinfo.st_size, 'bytes.')

def conv_graph(data, keep=1.0):
    """ Create a conv neural network

    Parameters
    ----------
    data : tf.Tensor
        input training data tensor
    keep : tf.Tensor
        dropout keep_prob
    
    Returns
    -------
    output : tf.Tensor
        conv_graph output
    """
    # definition conv/relu layer
    def conv_relu(input, stride, kernel_shape):
        # Create variable named "weights".
        weights = tf.get_variable("weights", kernel_shape,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        # Create variable named "biases".
        biases = tf.get_variable("biases", kernel_shape[-1:],
            initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, weights,
            strides=stride, padding='SAME')
        bias = tf.nn.bias_add(conv, biases)
        return tf.nn.relu(bias)
    # definition fc layer
    def fc(input, shape):
        # Create variable named "weights"
        weights = tf.get_variable("weights", shape,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        # Create variable named "biases".
        biases = tf.get_variable("biases", shape[-1:],
            initializer=tf.constant_initializer(0.1))
        return tf.matmul(input, weights) + biases
    # definition pooling layer
    def pooling(input, stride, kernel_shape):
        pool = tf.nn.max_pool(input, kernel_shape, stride, padding='SAME')
        return pool

    with tf.variable_scope("conv1"):
        relu1 = conv_relu(data, [1,1,1,1], [3,3,1,32])
    with tf.variable_scope("conv2"):
        relu2 = conv_relu(relu1, [1,2,2,1], [3,3,32,64])
    with tf.variable_scope("conv3"):
        relu3 = conv_relu(relu2, [1,2,2,1], [3,3,64,1024])
    with tf.variable_scope("pool1"):
        pool1 = pooling(relu3, [1,7,7,1], [1,7,7,1])
        pool1 = tf.reshape(pool1, [-1,1024])
    with tf.variable_scope("fc1"):
        fc1 = fc(pool1, [1024,512])
        dropout1 = tf.nn.dropout(fc1, keep)
    with tf.variable_scope("fc2"):
        fc2 = fc(dropout1, [512,10])
    with tf.variable_scope("softmax1"):
        output = tf.nn.softmax(fc2)
    return output


def train(dataset_path):
    """ Train mnist dataset

    Parameters
    ----------
    dataset_path : str
        input mnist dataset path
    """
    # create graph
    graph1 = tf.Graph()
    with graph1.as_default():
        # x,y_,lr,keep_prob placeholder
        x = tf.placeholder(tf.float32, [None, 784], name='input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='label')
        lr = tf.placeholder(tf.float32, name='lr')
        keep_prob = tf.placeholder(tf.float32)
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        y = conv_graph(x_reshape, keep_prob)
        # train 
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
        # eval
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # init variables
        init = tf.global_variables_initializer()
        # set model saver
        saver = tf.train.Saver()
    # load data
    mnist = input_data.read_data_sets(dataset_path, one_hot=True)
    # set learning rate
    def update_lr(step):
        max_lr, min_lr, decay_speed = 0.0002, 0.0001, 2000.0
        return min_lr + (max_lr - min_lr) * math.exp(-step / decay_speed)
    # training and evaluation
    with tf.Session(graph=graph1) as sess:
        sess.run(init)
        for i in range(20000):
            batch_x, batch_y = mnist.train.next_batch(50)
            sess.run(train_step,
                feed_dict={x: batch_x, y_: batch_y, lr: update_lr(i), keep_prob: 0.5})
            if (i+1) % 2000 == 0:
                # acc, loss = sess.run([accuracy, cross_entropy],
                #     feed_dict = {x: batch_x, y_: batch_y, keep_prob: 1.0})
                # print("train -> acc: {} loss: {}".format(acc, loss))
                acc, loss = sess.run([accuracy, cross_entropy],
                    feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
                print("iter: {} acc: {} loss: {}".format(i + 1, acc, loss))
                saver.save(sess, "./mnist/mnist.ckpt-{}".format(i+1))

def test(dataset_path):
    """ Train mnist dataset

    Parameters
    ----------
    dataset_path : str
        input mnist dataset path
    """
    # create graph
    graph1 = tf.Graph()
    with graph1.as_default():
        x = tf.placeholder(tf.float32, [1, 784], name="input")
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        y = conv_graph(x_reshape)
        saver = tf.train.Saver()
    # load data
    mnist = input_data.read_data_sets(dataset_path, one_hot=True)
    test_mnist = mnist.test.images[5555].reshape([1, -1])
    test_mnist_label = mnist.test.labels[5555]
    # test
    with tf.Session(graph=graph1) as sess:
        saver.restore(sess, "./mnist/mnist.ckpt-20000")
        out = sess.run(y, feed_dict={x: test_mnist})
        print("test : {}".format(np.argmax(out.flatten(), 0)))
        print("real : {}".format(np.argmax(test_mnist_label, 0)))

def frozen(dataset_path):
    """ Train mnist dataset

    Parameters
    ----------
    dataset_path : str
        input mnist dataset path
    """
    pass

def main(args):
    if FLAGS.mode == 'download':
        download_data(os.getcwd(), "mnist")
    elif FLAGS.mode == 'train':
        train(os.path.join(os.getcwd(), "mnist"))
    elif FLAGS.mode == 'test':
        test(os.path.join(os.getcwd(), "mnist"))
    elif FLAGS.mode == 'frozen':
        frozen(os.path.join(os.getcwd(), "mnist"))
    else:
        raise ValueError("not a valid mode: {}!".format(FLAGS.mode))

if __name__ == '__main__':
    tf.app.run()