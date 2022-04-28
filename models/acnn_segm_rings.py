import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, acnn_module_rings

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    normals_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl, normals_pl, cls_labels_pl

NUM_CATEGORIES = 16

def get_model(point_cloud, cls_label, normals, is_training, bn_decay=None):
    """ Part segmentation A-CNN, input is points BxNx3 and normals BxNx3, output Bx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_normals = normals
    l0_points = normals
    p1="att_poola"
    p2="att_poolb"
    flag=None
    # Set Abstraction layers
    print("orignal dimentions are :",l0_xyz.get_shape())
    #ip=input("Next :?")
    #l1_xyz, l1_points, l1_normals = acnn_module_rings(l0_xyz, l0_points, l0_normals, 512, [[0.0, 0.1], [0.1, 0.2]], [16,48], [[32,32,64], [64,64,128]], is_training, bn_decay, scope='layer1')
    l1_xyz, l1_points, l1_normals,skip1_block1,skip2_block1 = acnn_module_rings(l0_xyz, l0_points, l0_normals, 512, [[0.0, 0.1], [0.1, 0.2]], [16,48], [[32,32,64], [64,64,128]],0,0,flag, is_training, bn_decay, scope='layer1')
    print("l1_points are :",l1_points.shape)
    #ip=input("Next :?")
    l2_xyz, l2_points, l2_normals,skip1_block2,skip2_block2 = acnn_module_rings(l1_xyz, l1_points, l1_normals, 128, [[0.1, 0.2], [0.3, 0.4]], [16,48], [[64,64,128], [128,128,256]],0,0,flag, is_training, bn_decay, scope='layer2')
    #l2_xyz, l2_points, l2_normals = acnn_module_rings(l1_xyz, l1_points, l1_normals, 128, [[0.1, 0.2], [0.3, 0.4]], [16,48], [[64,64,128], [128,128,256]], is_training, bn_decay, scope='layer2')
    print("l2_points are :",l2_points.shape)
    #ip=input("Next :?")
    s1=skip1_block1
    s2=skip2_block1
    flag=1
    #l3_xyz, l3_points, l3_normals = acnn_module_rings(l2_xyz, l2_points, l2_normals, 64, [[0.3, 0.4], [0.5, 0.6]], [16,48], [[128,128,256], [256,256,512]], is_training, bn_decay, scope='layer3')
    l3_xyz, l3_points, l3_normals,skip1_block3,skip2_block3 = acnn_module_rings(l2_xyz, l2_points, l2_normals, 64, [[0.3, 0.4], [0.5, 0.6]], [16,48], [[128,128,256], [256,256,512]],s1,s2, flag,is_training, bn_decay, scope='layer3')
    print("l3_points are :",l3_points.shape)
    #ip=input("Next :?")
    
    #l4_xyz, l4_points, l4_normals = acnn_module_rings(l3_xyz, l3_points, l3_normals, 32, [[0.5, 0.6], [0.7, 0.8]], [16,48], [[256,256,512], [512,512,1024]], is_training, bn_decay, scope='layer4')
    #print("l4_points are :",l4_points.shape)
    #ip=input("Next :?")
    #print("dimentions of l1_points in clsring are :",l4_points.get_shape())
    #l5_xyz, l5_points, l5_normals = acnn_module_rings(l4_xyz, l4_points, l4_normals, 16, [[0.7, 0.8], [0.9, 1.0]], [16,48], [[512,5125,1024], [1024,1024,2048]], is_training, bn_decay, scope='layer5')
    #print("l5_points are :",l5_points.shape)
    #ip=input("Next :?")
    #print("dimentions of l1_points in clsring are :",l5_points.get_shape())
    
    #print("The dimentions I am going to send to pointnet_sa  for features are :- ", l3_points.get_shape())
    #l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    
    #l6_xyz, l6_points, _ = pointnet_sa_module(l5_xyz, l5_points, npoint=None, radius=None, nsample=None, mlp=[1024,2048,4096], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer6')
    l4_xyz, l4_points, _ = pointnet_sa_module(l3_xyz, l3_points, npoint=None, radius=None, nsample=None, mlp=[512,1024,2048], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    print("Dimentions after pointnetsa are :",l4_points.shape)
    #ip=input("Next :?")
    print("cls_label is :",cls_label)
    
    cls_label_one_hot = tf.one_hot(cls_label, depth=NUM_CATEGORIES, on_value=1.0, off_value=0.0)
    cls_label_one_hot = tf.reshape(cls_label_one_hot, [batch_size, 1, NUM_CATEGORIES])
    cls_label_one_hot = tf.tile(cls_label_one_hot, [1,num_point,1])
    print("cls_label_one_hot in the end is :",cls_label_one_hot.shape)
    #ip=input("Next :?")
    #print("The dimentions I am going to send to pointnet_fp  for features are :- ", l3_points.get_shape())
    #eval=input("please Enter the value :")
    # Feature Propagation layers
    #up_l6_points = pointnet_fp_module(l0_xyz, l6_xyz, None, l6_points, [64], is_training, bn_decay, scope='fa_layer1_up')
    
    #up_l5_points = pointnet_fp_module(l0_xyz, l5_xyz, None, l5_points, [64], is_training, bn_decay, scope='fa_layer2_up')
    up_l4_points = pointnet_fp_module(l0_xyz, l4_xyz, None, l4_points, [64], is_training, bn_decay, scope='fa_layer1_up')
    print("The returned dimentions from fa_layer1  are :",up_l4_points.get_shape())
    #ip=input("Next ?:")
    up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay, scope='fa_layer2_up')
    print("The returned dimentions from fa_layer2  are :",up_l3_points.get_shape())
    #ip=input("Next :?")
    up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay, scope='fa_layer3_up')
    print("The returned dimentions from fa_layer3  are :",up_l2_points.get_shape())
    #ip=input("Next ?:")
    up_l1_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4_up')
    print("The returned dimentions from fa_layer4  are :",up_l1_points.get_shape())
    print("The returned dimentions are :",up_l1_points.get_shape())
    #ip=input("I was here bro :")
    
    concat = tf.concat(axis=-1, values=[ 
                                     up_l4_points,
                                     up_l3_points,
                                     up_l2_points,
                                     up_l1_points,
                                     cls_label_one_hot,
                                     l0_xyz
                                     ])
    #print("after concat :",concat.get_shape())
    #eval1=input("I am going to call shoots :")

    net = tf_util.conv1d(concat, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp2')
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fc4')
    #print("after fcs :",net.get_shape())
    #eval1=input("I am going to call shootsqqqq :")
    return net, end_points


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
