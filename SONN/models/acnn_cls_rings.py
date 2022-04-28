import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, acnn_module_rings

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    normals_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl, normals_pl

def get_model(point_cloud, normals, is_training, bn_decay=None):
    """ Classification A-CNN, input is points BxNx3 and normals BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    #l0_xyz = point_cloud
    #l0_normals = normals
    #l0_points = None

    # Abstraction layers
    #l1_xyz, l1_points, l1_normals = acnn_module_rings(l0_xyz, l0_points, l0_normals, 512, [[0.0, 0.1], [0.1, 0.2]], [16,48], [[32,32,64], [64,64,128]], is_training, bn_decay, scope='layer1')
    #l2_xyz, l2_points, l2_normals = acnn_module_rings(l1_xyz, l1_points, l1_normals, 128, [[0.1, 0.2], [0.3, 0.4]], [16,48], [[64,64,128], [128,128,256]], is_training, bn_decay, scope='layer2')
    #_, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Fully connected layers
    l0_xyz = point_cloud
    l0_normals = normals
    l0_points = None
    p1="att_poola"
    p2="att_poolb"
    # Abstraction layers
    flag=None
    l1_xyz, l1_points, l1_normals,skip1_block1,skip2_block1 = acnn_module_rings(l0_xyz, l0_points, l0_normals, 512,[[0.0, 0.05], [0.05, 0.1]], [16,48], [[32,32,64], [64,64,128]],0,0,flag, is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_normals,skip1_block2,skip2_block2 = acnn_module_rings(l1_xyz, l1_points, l1_normals, 128,[[0.05, 0.1], [0.1, 0.16]], [16,48], [[64,64,128], [128,128,256]],0,0,flag, is_training, bn_decay, scope='layer2')
    s1=skip1_block1
    s2=skip2_block1
    flag=1
    l3_xyz, l3_points, l3_normals,skip1_block3,skip2_block3 = acnn_module_rings(l2_xyz, l2_points, l2_normals, 64,[[0.1, 0.16],[0.16,0.22]], [16,48], [[128,128,256], [256,256,512]],s1,s2, flag,is_training, bn_decay, scope='layer3')
    print("dimentions of l3_points in clsring are :",l3_points.get_shape())
    _, l4_points, _ = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[512,1024,2048], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    net = tf.reshape(l4_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 15, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)