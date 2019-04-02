# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:58:00 2019

@author: GKX
"""
import tensorflow as tf
import numpy as np


#   reshape 的流程：先将矩阵t变换为一维矩阵，然后再对矩阵的形式进行更改
#   tf_flatten将 a 拉成一维
def tf_flatten(a):
    """Flatten tensor"""
    return tf.reshape(a, [-1])

#   增加维度为2维，后增加维度数据，再拉成一维
def tf_repeat(a, repeats, axis=0):
    """TensorFlow version of np.repeat for 1D"""
    assert len(a.get_shape()) == 1    # if a shape为1D，执行下面语句，否则中断，报错
    a = tf.expand_dims(a, -1)    #   在后面增加一维：（1，1）
    a = tf.tile(a, [1, repeats]) #   扩展 a 的数据 （第一维数量*1，第二维数量*repeats）
    a = tf_flatten(a)      
    return a


#   增加维度为3维，后增加维度数据
def tf_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""
    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a





'''
tf_batch_map_coordinates函数对应的双线性插值操作：
获取采样位置周围的4个坐标点位置
获取采样位置的像素值，双线性插值得到实际的采样结果
'''
def tf_batch_map_coordinates(input, coords, order=1):
    '''
    Parameters
    ----------
    input : tf.Tensor. shape = (b*c, L, h, w)
    coords : tf.Tensor. shape = (b*c,l*h*w,3)
    Returns
    -------
    tf.Tensor. shape = (b*c, L, h, w)
    '''
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    n = tf.argmax(input_shape)
    input_size = input_shape[n]
    n_coords = tf.shape(coords)[1]   #   l*h*w


    #

    # Tensor("conv1_1/strided_slice_8:0", shape=(), dtype=int32)

    '''
    tf.clip_by_value(A, min, max)：
    输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。
    小于min的让它等于min，大于max的元素的值等于max。
    '''
    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)   #   b*c,l*h*w,3


    
    
    #得到目标坐标左上角（left top）的整数坐标
    coords_000 = tf.cast(tf.floor(coords), 'int32')
    #得到右下角的整数坐标
    coords_111 = tf.cast(tf.ceil(coords), 'int32')
    #得到和左上角同一水平高度的三个坐标
    coords_010 = tf.stack([coords_000[..., 0],
                           coords_111[..., 1],
                           coords_000[..., 2]], axis=-1)
    coords_100 = tf.stack([coords_111[..., 0],
                           coords_000[..., 1],
                           coords_000[..., 2]], axis=-1)
    coords_110 = tf.stack([coords_111[..., 0],
                           coords_111[..., 1],
                           coords_000[..., 2]], axis=-1)
    #得到和右下角同一水平高度的三个坐标
    coords_001 = tf.stack([coords_000[..., 0],
                           coords_000[..., 1],
                           coords_111[..., 2]], axis=-1)
    coords_101 = tf.stack([coords_111[..., 0],
                           coords_000[..., 1],
                           coords_111[..., 2]], axis=-1)
    coords_011 = tf.stack([coords_000[..., 0],
                           coords_111[..., 1],
                           coords_111[..., 2]], axis=-1)

    
    #   b*c为5，h*w为4，总数为所有图片所有坐标总数
    
    def _get_vals_by_coords(input, coords,batch_size,n_coords):
        idx = tf_repeat(tf.range(batch_size), n_coords)   #  b*c*L*h*w
        indices = tf.stack([idx, 
                            tf_flatten(coords[..., 0]), 
                            tf_flatten(coords[..., 1]), 
                            tf_flatten(coords[..., 2])], axis=-1)   # (b*c*L*h*w, 3)
    
        vals = tf.gather_nd(input, indices) #   按照索引提出新的数组
        vals = tf.reshape(vals, (batch_size, n_coords))   #  (b*c,L*h*w)
        return vals
    
     #   分别得到八个点的像素值。
    vals000 = _get_vals_by_coords(input, coords_000,batch_size,n_coords)
    vals010 = _get_vals_by_coords(input, coords_010,batch_size,n_coords)
    vals100 = _get_vals_by_coords(input, coords_100,batch_size,n_coords)
    vals110 = _get_vals_by_coords(input, coords_110,batch_size,n_coords)
    
    vals001 = _get_vals_by_coords(input, coords_001,batch_size,n_coords)
    vals101 = _get_vals_by_coords(input, coords_101,batch_size,n_coords)
    vals011 = _get_vals_by_coords(input, coords_011,batch_size,n_coords)
    vals111 = _get_vals_by_coords(input, coords_111,batch_size,n_coords)
    
    '''
     000水平面用双线性插值 得到一个值mapped_vals_000 
     其坐标为(coords_offset_000[..., 0],coords_offset_000[..., 1],0) + coords_000
    '''
    coords_offset_000 = coords - tf.cast(coords_000, 'float32')
    vals_t000 = vals000 + (vals100 - vals000) * coords_offset_000[..., 0]
    vals_b000 = vals010 + (vals110 - vals010) * coords_offset_000[..., 0]
    mapped_val_000 = vals_t000 + (vals_b000 - vals_t000) * coords_offset_000[..., 1]
     
    '''
    111水平面用双线性插值 得到一个值mapped_vals_111
    其坐标为(coords_offset_000[..., 0],coords_offset_000[..., 1],0) + coords_001
    '''
    vals_t000 = vals001 + (vals101 - vals001) * coords_offset_000[..., 0]
    vals_b000 = vals011 + (vals111 - vals011) * coords_offset_000[..., 0]
    mapped_vals_111 = vals_t000 + (vals_b000 - vals_t000) * coords_offset_000[..., 1]
    
    '''
    根据 mapped_val_000 和 mapped_vals_111 得到最终的插值 mapped_vals
    其坐标为 coords_00
    '''
    mapped_vals = mapped_val_000 + (mapped_vals_111 - mapped_val_000) * coords_offset_000[..., 2]
    
    return mapped_vals



def tf_batch_map_offsets(input, offsets, order=1):
    """
    Parameters
    ----------
    input : tf.Tensor. shape = (b*c, L, h, w)
    offsets : tf.Tensor. shape = (b*c, L, h, w,3)
    Returns
    -------
    tf.Tensor. shape = (b*c, L, h, w)
    """

    #  input.shape,offsets.shape: (?, 32, 32, 32)  (?, 32, 32, 32, 3)

    input_shape = tf.shape(input)   #   (b*c,l,h,w)
    batch_size = input_shape[0]

    offsets = tf.reshape(offsets, (batch_size, -1, 3))   #   (b*c,l*h*w,3)

    grid = tf.meshgrid(tf.range(input_shape[1]),
                       tf.range(input_shape[2]),
                       tf.range(input_shape[3]))
    grid = tf.stack(grid, axis=-1)      #   (l,h,w,3)

    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 3))    #   (l*h*w,3)

    
    grid = tf_repeat_2d(grid, batch_size)    #   (b*c,l*h*w,3)
    coords = offsets + grid


    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals














