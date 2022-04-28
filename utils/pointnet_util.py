""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping_ring'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/ordering'))
from tf_sampling import farthest_point_sample, gather_point
#from tf_grouping_ring import ring_point, group_point, knn_point
from tf_grouping_ring import ring_point, group_point,group_point3, knn_point
from tf_interpolate import three_nn, three_interpolate
from tf_ordering import order_neighbors
import tensorflow as tf
import numpy as np
import tf_util

def sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec=None, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        tnet_spec: dict (keys: mlp, mlp2, is_training, bn_decay), if None do not apply tnet
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    indecies = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, indecies) # (batch_size, npoint, 3)
    new_normals = gather_point(normals, indecies) # (batch_size, npoint, 3)
    if knn:
        _,idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if tnet_spec is not None:
        grouped_xyz = tnet(grouped_xyz, tnet_spec)
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, new_normals, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', tnet_spec=None, knn=False, use_xyz=True):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)

        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='SAME', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)

        if pooling=='avg':
            new_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = tf_util.max_pool2d(-1*new_points, [1,nsample], stride=[1,1], padding='VALID', scope='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf_util.max_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='maxpool1')
            max_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay)
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx

def ret_shortcut(skip1,skip2,ind,dout1,dout2,bn_decay,bn,name,is_training):
    batch_size = tf.shape(ind)[0]
    m = tf.shape(ind)[1]
    ind_reshaped = tf.reshape(ind, shape=[batch_size, 1, m])
    shortcut1=group_point3(skip1, ind_reshaped)
    shortcut1 = tf.squeeze(shortcut1, [1])
    shortcut1conv = tf_util.conv2d(shortcut1 , dout1, [1,1],padding='SAME', stride=[1,1], bn=bn, is_training=is_training,scope=name+'skip1', bn_decay=bn_decay)
    shortcut2=group_point3(skip2, ind_reshaped)
    shortcut2 = tf.squeeze(shortcut2, [1])
    shortcut2conv = tf_util.conv2d(shortcut2 , dout2, [1,1],
    padding='SAME', stride=[1,1], bn=bn, is_training=is_training,scope=name+'skip2', bn_decay=bn_decay)
    #shortcut2=group_point3(skip2, ind_reshaped)
    #shortcut1conv=shortcut1
    #shortcut2conv=shortcut2
    return shortcut1,shortcut1conv,shortcut2,shortcut2conv
def RG(short1_conv,short2_conv):
    activation_fn=tf.nn.relu
    short1 = activation_fn(short1_conv)
    short2 = activation_fn(short2_conv)
    short1=tf.contrib.layers.group_norm(short1,32,3,trainable=True,scope='gnorm1')
    short2=tf.contrib.layers.group_norm(short2,32,3,trainable=True,scope='gnorm2')
    short1=tf.math.add(short1_conv, short1)
    short2=tf.math.add(short2_conv, short2)
    print("after performing additon operation ,short1 is :",short1.shape)
    print("after performing additon operation ,short2 is :",short2.shape)
    #ip=input("Next :?")
    return short1,short2
def acnn_module_rings(xyz, points, normals, npoint, radius_list, nsample_list, mlp_list,s1,s2, flag,is_training, bn_decay, scope, bn=True, use_xyz=True):
    ''' A-CNN module with rings
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            normals: (batch_size, ndataset, 3) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius_list: list of float32 -- search radiuses (inner and outer) represent ring in local region
            nsample_list: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    # data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        indecies = farthest_point_sample(npoint, xyz)
        new_xyz = gather_point(xyz, indecies) # (batch_size, npoint, 3)
        new_normals = gather_point(normals, indecies) # (batch_size, npoint, 3)
        print("s1 inside function call is :",s1)
        print("s2 inside function call is :",s2)
        batch_size = xyz.get_shape()[0].value
        new_points_list = []
        count=0
        d_out1=mlp_list[0][2]
        d_out2=mlp_list[1][2]
        if flag is not None:
            short1,short1_conv,short2,short2_conv=ret_shortcut(s1,s2,indecies,d_out1,d_out2,bn_decay,bn,scope,is_training)
            short1_conv,short2_conv=RG(short1_conv,short2_conv)
            print("short1 after gnorm is :",short1_conv.shape)
            print("short2 after gnorm is :",short2_conv.shape)
            #ip=input("Next :?")
        for i in range(len(radius_list)):
            count=count+1
            radius_in = radius_list[i][0]
            radius_out = radius_list[i][1]
	        #print("radius_in is :",radius_in)
	        #print("radius_out is :",radius_out)
            #ip=input("Next :?")
            nsample = nsample_list[i]
            idx, _ = ring_point(radius_in, radius_out, nsample, xyz, new_xyz, indecies)
            idx, _, _ = order_neighbors(nsample, idx, xyz, new_xyz, new_normals)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            for j,num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf.concat([grouped_points, grouped_points[:,:,:2,:]], axis=2)
                if flag is not None and j==2:
                    print("I am in if of the inner for loop :")
                    print("Hello :")
                    #print("I am in if of the inner for loop :")
                    #print("Hello :")
                    if count==1:
                        print("value of count1 is :",count)
                        print("value of mlp is:",num_out_channel)
                        grouped_points= tf_util.conv2d2(grouped_points, short1_conv,num_out_channel, [1,3], padding='VALID', stride=[1,1], bn=bn, is_training=is_training,scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
                        print("The returned dimentions from pooling are :",grouped_points.shape)
                        #grouped_count1=mid1
                    if count==2:
                        print("value of count2 is :",count)
                        print("value of mlp is:",num_out_channel)
                        grouped_points= tf_util.conv2d2(grouped_points, short2_conv,num_out_channel, [1,3], padding='VALID', stride=[1,1], bn=bn, is_training=is_training,scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
                        print("The returned dimentions from pooling are :",grouped_points.shape)    
                        #grouped_count2=mid2    
                else:
                    print("I am in else of the inner for loop :")
                    grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1,3],
                                                padding='VALID', stride=[1,1], bn=bn, is_training=is_training,
                                                scope='conv%d_%d'%(i,j), bn_decay=bn_decay)
            if count==1:
                skip1=grouped_points
            if count==2:
                skip2=grouped_points   
            new_points = tf.reduce_max(grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        print("group points being returned are :",new_points_concat.shape)
        #ip=input("Next :?")
        return new_xyz, new_points_concat, new_normals,skip1,skip2


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1
