# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:29:15 2019

@author: GKX
"""

#import keras

from deform_conv.deform_conv import tf_batch_map_offsets
from keras.layers import Conv3D
import tensorflow as tf
from keras.initializers import RandomNormal


class ConvOffset3D(Conv3D):
    """
    ConvOffset2D卷积层学习2D的偏移量，使用双线性插值输出变形后采样值
    """

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init"""

        self.filters = filters
        super(ConvOffset3D, self).__init__(self.filters * 3, (3, 3, 3),
                                           padding='same',
                                           use_bias=False,
                                           # kernel_initializer='zeros',
                                           kernel_initializer=RandomNormal(0, init_normal_stddev),
                                           **kwargs)



    def call(self, x):
        """Return the deformed featured map"""
        # x = tf.transpose(x,[0,2,3,4,1])
        x_shape = x.get_shape()    #   (?, 64, 32, 32, 32)



        #   x_shape = (b,c,L,h,w)
        # 卷积输出得到2倍通道的feature map，获取到偏移量 大小为(b,3c,L,h,w)
        offsets = super(ConvOffset3D, self).call(x)   #    (?, 192, 32, 32, 32)




        # offsets: (b*c,l,h,w,3)
        offsets = self._to_bc_L_h_w_3(offsets, x_shape)   #   (?, 64, 32, 32, 3)



        # 将输入x也切换成这样: (b*c, L, h, w)
        x = self._to_bc_L_h_w(x, x_shape)    #   (?, 64, 32, 32)
        # print('>>>>>>>>>>>>>>>>>>>>>>>', x.shape)

        
        '''
        前面得到了偏移的(b∗c,h,w,2)和变形的输入(b∗c,h,w)。
        先将偏移原本的feature采样位置相加，得到实际的采样position
        调用tf_batch_map_coordinates做双线性插值操作，得到输出
        '''
        
        # 采样前形状：x:(b*c, L, h, w) ; 坐标为offsets（x,y）：(b*c, L, h, w, 3)
        # 双线性采样得到采样后的X_offset: (b*c, L, h, w)
        x_offset = tf_batch_map_offsets(x, offsets)
        # print('----------------------',x_offset.shape)

        # 再变原本的shape，即x_offset: b_c_L_h_w
        x_offset = self._to_b_c_L_h_w(x_offset, x_shape)   #   (?, 64, 32, 32, 32)
        # print('--------------------',x_offset.shape)

        return x_offset

    def compute_output_shape(self, input_shape):
        """Output shape is the same as input shape

        Because this layer does only the deformation part
        """
        return input_shape


#   x_shape = (b,c,L,h,w)
    @staticmethod
    def _to_bc_L_h_w_3(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)  ##  (b, L, h, w, 2c) -> (b*c, L, h, w, 2)"""
        # x = tf.transpose(x, [0, 4, 1, 2, 3])
        x = tf.reshape(x, (-1, 
                           int(x_shape[2]),
                           int(x_shape[3]),
                           int(x_shape[4]), 3))
        return x

    @staticmethod
    def _to_bc_L_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)  ##  (b, L, h, w, c) -> (b*c, L, h, w)"""
        # x = tf.transpose(x, [0, 4, 1, 2, 3])
        x = tf.reshape(x, (-1, 
                           int(x_shape[2]),
                           int(x_shape[3]),
                           int(x_shape[4])))
        return x

    @staticmethod
    def _to_b_c_L_h_w(x, x_shape):
        """##  (b*c, L*h*w) -> (b, c, L, h, w)"""
        x = tf.reshape(x, (-1, 
                           int(x_shape[1]),
                           int(x_shape[2]),
                           int(x_shape[3]),
                           int(x_shape[4])))
        # x = tf.transpose(x, [0, 2, 3, 4, 1])
        return x

    
    
