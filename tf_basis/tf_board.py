""" Test tensorflow visualization tool: TensorBoard"""
import sys
import os
import urllib
import math
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('type', '', 'graph_type')

def show_ckpt(base_path):
    """ Display ckpt graph in TensorBoard

    Parameters
    ----------
    base_path : str
        graph base path
    """
    # set path
    ckpt_prefix = base_path +'/mnist.ckpt-20000'
    model_log_path = os.path.join(base_path, "ckpt_log")
    # create graph
    graph1 = tf.Graph()
    saver = tf.train.import_meta_graph(ckpt_prefix + '.meta', clear_devices=True, graph=graph1)
    # test
    with tf.Session(graph=graph1) as sess:
        saver.restore(sess, ckpt_prefix)
        file_writer = tf.summary.FileWriter(model_log_path, sess.graph)
    # list graph op name
    for op in graph1.get_operations():
        print(op.name, op.type)

def show_freeze(base_path):
    """ Display freeze graph in TensorBoard

    Parameters
    ----------
    base_path : str
        graph bath path
    """
    # load freeze model
    model_path = os.path.join(base_path, "frozen_model.pb")
    model_log_path = os.path.join(base_path, "freeze_log")
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    graph1 = tf.Graph()
    with graph1.as_default():
        tf.import_graph_def(graph_def, name='')
    # list frozen graph ops
    for op in graph1.get_operations():
        print(op.name, op.type)
    with tf.Session(graph=graph1) as sess:
        file_writer = tf.summary.FileWriter(model_log_path, sess.graph)

def main(args):
    if FLAGS.type == 'ckpt':
        show_ckpt("./mnist")
    elif FLAGS.type == 'freeze':
        show_freeze("./mnist")
    else:
        raise ValueError("not a valid graph type: {}!".format(FLAGS.mode))

if __name__ == '__main__':
    tf.app.run()