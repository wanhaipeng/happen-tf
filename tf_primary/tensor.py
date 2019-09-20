""" Test tensorflow tensor"""
import sys
import tensorflow as tf

def test_tensor():
    """test tensor property"""
    mytensor = tf.ones([1,2,3,4], tf.float32)
    # with tf session activate
    with tf.Session() as sess:
        print("\x1b[32m" + 50*"*" + "\x1b[0m")
        print(" type: {}\n data value: {}\n data type: {}\n data shape: {}".format(
            type(mytensor), mytensor.eval(), mytensor.dtype, mytensor.get_shape()))
        print("\x1b[32m" + 50*"*" + "\x1b[0m")

def test_variable():
    """test basic tf variable tensor"""
    # definition conv/relu layer
    def conv_relu(input, kernel_shape, bias_shape):
        # Create variable named "weights".
        weights = tf.get_variable("weights", kernel_shape,
            initializer=tf.random_normal_initializer())
        # Create variable named "biases".
        biases = tf.get_variable("biases", bias_shape,
            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, weights,
            strides=[1, 1, 1, 1], padding='SAME') # SAME采用补全padding方式，VALID采用丢弃方式
        return tf.nn.relu(conv + biases)
    # create input
    input1 = tf.random_normal([1,6,6,32])
    # create layer
    with tf.variable_scope("conv1"):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input1, [5, 5, 32, 32], [32])
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        output = sess.run({'conv1/Relu:0': relu1})
        print("\x1b[32m" + 50*"*" + "\x1b[0m")
        print(output)
        print("input shape: {}".format(input1.get_shape()))
        print("output shape: {}".format(output['conv1/Relu:0'].shape))
        print("\x1b[32m" + 50*"*" + "\x1b[0m")

test_tensor()
test_variable()