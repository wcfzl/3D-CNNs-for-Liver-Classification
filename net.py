import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, SpatialDropout3D, Dropout, MaxPooling3D, AveragePooling3D,GlobalMaxPooling3D, Activation, \
    BatchNormalization, LeakyReLU, ReLU, Dense, Flatten, GlobalAveragePooling3D, Reshape, multiply, Add
from keras.optimizers import Adam, SGD
from focal_loss import focal_loss, focal_loss_fixed
from deform_conv.layer import ConvOffset3D
import keras

K.set_image_data_format("channels_first")
keras.initializers.Initializer()
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def squeeze_excitation_layer(x, out_dim, radio=4, activation=LeakyReLU):

    squeeze = GlobalAveragePooling3D()(x)
    excitation = Dense(units=out_dim // radio)(squeeze)
    excitation = activation()(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((out_dim, 1, 1, 1))(excitation)
    scale = multiply([x, excitation])
    return scale

def model_3d_1(input_shape, initial_learning_rate=0.00001, batch_normalization=True,
               instance_normalization=False, activation_name="sigmoid", opt = 'Adam'):
    base_fiters = 64
    inputs = Input(input_shape)
    current_layer = inputs
    print(current_layer._keras_shape)

    layer1_1 = create_convolution_block(input_layer=current_layer,
                                      kernel=(3, 3, 3),
                                      n_filters=base_fiters,
                                      name = '1_1',
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      instance_normalization=instance_normalization,
                                      activation=LeakyReLU)
    layer1_2 = create_convolution_block(input_layer=layer1_1,
                                      kernel=(3, 3, 3),
                                      n_filters=base_fiters,
                                      name = '1_2',
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      instance_normalization=instance_normalization,
                                      activation=LeakyReLU)
    print(layer1_2._keras_shape)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2),name = 'pool1')(layer1_2)
    print('pool1:',pool1._keras_shape)
    layer2_1 = create_convolution_block(input_layer=pool1,
                                      kernel=(3, 3, 3),
                                      n_filters=base_fiters*2,
                                      name='2_1',
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      instance_normalization=instance_normalization,
                                      activation=LeakyReLU)
    layer2_2 = create_convolution_block(input_layer=layer2_1,
                                      kernel=(3, 3, 3),
                                      n_filters=base_fiters*2,
                                      name = '2_2',
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      instance_normalization=instance_normalization,
                                      activation=LeakyReLU)
    print(layer2_2._keras_shape)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2),name = 'pool2')(layer2_2)
    print('pool2:', pool2._keras_shape)
    layer3_1 = create_convolution_block(input_layer=pool2,
                                      kernel=(3, 3, 3),
                                      n_filters=base_fiters*4,
                                      name='3_1',
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      instance_normalization=instance_normalization,
                                      activation=LeakyReLU)
    layer3_2 = create_convolution_block(input_layer=layer3_1,
                                      kernel=(3, 3, 3),
                                      n_filters=base_fiters*4,
                                      name = '3_2',
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      instance_normalization=instance_normalization,
                                      activation=LeakyReLU)
    print(layer3_2._keras_shape)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2),name = 'pool3')(layer3_2)
    print('pool3:', pool3._keras_shape)
    layer4_1 = create_convolution_block(input_layer=pool3,
                                      kernel=(3, 3, 3),
                                      n_filters=base_fiters*8,
                                      name='4_1',
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      instance_normalization=instance_normalization,
                                      activation=LeakyReLU)
    layer4_2 = create_convolution_block(input_layer=layer4_1,
                                      kernel=(3, 3, 3),
                                      n_filters=base_fiters*8,
                                      name='4_2',
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      instance_normalization=instance_normalization,
                                      activation=LeakyReLU)
    print(layer4_2._keras_shape)
    #############
    layer7 = GlobalMaxPooling3D(data_format="channels_first",name = 'Gpool')(layer4_2)
    print(layer7._keras_shape)
    layer7 = Dropout(rate=0.3,name = 'dropout1')(layer7)
    layer8 = Dense(1, activation='sigmoid',name = 'dense1')(layer7)
    print(layer8._keras_shape)
    model = Model(inputs=inputs, outputs=layer8)
    if opt == 'Adam':
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss="binary_crossentropy", metrics=['accuracy'])
    elif opt == 'SGD':
        model.compile(optimizer=SGD(lr=initial_learning_rate),  loss=[focal_loss(alpha=.25, gamma=2)], metrics=['accuracy'])
    return model


def model_3d_2(input_shape, initial_learning_rate=0.00001, batch_normalization=True,
               instance_normalization=False, activation_name="sigmoid", opt = 'Adam'):
    base_fiters = 64
    inputs = Input(input_shape)
    current_layer = inputs
    print(current_layer._keras_shape)
    layer1 = create_convolution_block(input_layer=current_layer,
                                      kernel=(3, 3, 3),
                                      n_filters=base_fiters,
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      instance_normalization=instance_normalization,
                                      activation=LeakyReLU)
    print('-------------------------', layer1.shape)
    layer1 = create_convolution_block(input_layer=current_layer,
                                      kernel=(3, 3, 3),
                                      n_filters=base_fiters,
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      instance_normalization=instance_normalization,
                                      activation=LeakyReLU)
    # layer1 = create_deform_block(input_layer=layer1,
    #                                   kernel=(3, 3, 3),
    #                                   n_filters=base_fiters,
    #                                   name = '1_1',
    #                                   padding='same',
    #                                   batch_normalization=batch_normalization,
    #                                   instance_normalization=instance_normalization,
    #                                   activation=LeakyReLU)
    print('-------------------------', layer1.shape)
    print(layer1._keras_shape)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(layer1)
    print('pool1:',pool1._keras_shape)
    layer2 = create_deform_block(input_layer=pool1,
                                 n_filters=base_fiters*2,
                                 padding='same',
                                 kernel=(3, 3, 3),
                                 name='2_1',
                                 batch_normalization=batch_normalization,
                                 instance_normalization=instance_normalization,
                                 activation=LeakyReLU)
    print('-------------------------',layer2.shape)
    layer2 = create_deform_block(input_layer=layer2,
                                 kernel=(3, 3, 3),
                                 n_filters=base_fiters*2,
                                 name='2_2',
                                 padding='same',
                                 batch_normalization=batch_normalization,
                                 instance_normalization=instance_normalization,
                                 activation=LeakyReLU)
    print(layer2._keras_shape)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2),name = 'pool2')(layer2)
    print('pool2:', pool2._keras_shape)
    layer3_1 = create_deform_block(input_layer=pool2,
                                   n_filters=base_fiters*4,
                                   name='3_1',
                                   padding='same',
                                   batch_normalization=batch_normalization,
                                   instance_normalization=instance_normalization,
                                   activation=LeakyReLU)
    layer3_2 = create_deform_block(input_layer=layer3_1,
                                   kernel=(3, 3, 3),
                                   n_filters=base_fiters*4,
                                   name='3_2',
                                   padding='same',
                                   batch_normalization=batch_normalization,
                                   instance_normalization=instance_normalization,
                                   activation=LeakyReLU)
    print(layer3_2._keras_shape)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2),name = 'pool3')(layer3_2)
    print('pool3:', pool3._keras_shape)
    layer4_1 = create_deform_block(input_layer=pool3,
                                   n_filters=base_fiters*8,
                                   name='4_1',
                                   padding='same',
                                   batch_normalization=batch_normalization,
                                   instance_normalization=instance_normalization,
                                   activation=LeakyReLU)
    layer4_2 = create_deform_block(input_layer=layer4_1,
                                   kernel=(3, 3, 3),
                                   name='4_2',
                                   n_filters=base_fiters*8,
                                   padding='same',
                                   batch_normalization=batch_normalization,
                                   instance_normalization=instance_normalization,
                                   activation=LeakyReLU)
    print(layer4_2._keras_shape)
    #############
    #############
    layer7 = GlobalMaxPooling3D(data_format="channels_first",name = 'GPool')(layer4_2)
    print(layer7._keras_shape)
    layer7 = Dropout(rate=0.3)(layer7)
    layer8 = Dense(1, activation='sigmoid',name = 'dense1')(layer7)
    print(layer8._keras_shape)
    model = Model(inputs=inputs, outputs=layer8)
    if opt == 'Adam':
        model.compile(optimizer=Adam(lr=initial_learning_rate), loss="binary_crossentropy", metrics=['accuracy'],name = 'loss_Adam')
    elif opt == 'SGD':
        model.compile(optimizer=SGD(lr=initial_learning_rate),  loss="binary_crossentropy", metrics=['accuracy'],name = 'loss_SGD')
    return model



def create_convolution_block(input_layer, n_filters, name, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:

    """
    # try:
    #     from keras_contrib.layers.normalization import InstanceNormalization
    # except ImportError:
    #     raise ImportError("Install keras_contrib in order to use instance normalization."
    #                       "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, name="convx"+name)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1, name="BNx"+name)(layer)
    elif instance_normalization:
        layer = InstanceNormalization(axis=1, name="INx"+name)(layer)
    if activation is None:
        return layer
    else:
        return activation(name="AC"+name)(layer)

def create_deform_block(input_layer, n_filters, name ,batch_normalization=False, kernel=(3, 3, 3), activation=None,
                        padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:

    """
    # try:
    #     from keras_contrib.layers.normalization import InstanceNormalization
    # except ImportError:
    #     raise ImportError("Install keras_contrib in order to use instance normalization."
    #                       "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
    offset_filters = input_layer._keras_shape[1]
    layer = ConvOffset3D(filters=offset_filters, name='conv'+name)(input_layer)
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return layer
    else:
        return activation()(layer)

def create_dense_convolution_block_separate(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3),activation=None,
                                      padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    # try:
    #     from keras_contrib.layers.normalization import InstanceNormalization
    # except ImportError:
    #     raise ImportError("Install keras_contrib in order to use instance normalization."
    #                       "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")

    strides_x = (strides[0], 1, 1)
    strides_y = (1, strides[1], 1)
    strides_z = (1, 1, strides[2])
    print(kernel[0],kernel[1],kernel[2])
    conv_x = Conv3D(n_filters, (kernel[0], 1, 1), padding=padding, strides=strides_x)(input_layer)
    if batch_normalization:
        conv_x = BatchNormalization(axis=1)(conv_x)
    elif instance_normalization:
        conv_x = InstanceNormalization(axis=1)(conv_x)
    if activation is None:
        conv_x = Activation('relu')(conv_x)
    else:
        conv_x = activation()(conv_x)
    concat1 = concatenate([conv_x, input_layer], axis=1)
    conv_y = Conv3D(n_filters, (1, kernel[1], 1), padding=padding, strides=strides_y)(concat1)
    if batch_normalization:
        conv_y = BatchNormalization(axis=1)(conv_y)
    elif instance_normalization:
        conv_y = InstanceNormalization(axis=1)(conv_y)
    if activation is None:
        conv_y = Activation('relu')(conv_y)
    else:
        conv_y = activation()(conv_y)
    concat2 = concatenate([concat1, conv_y], axis=1)
    conv_z = Conv3D(n_filters, (1, 1, kernel[2]), padding=padding, strides=strides_z)(concat2)
    if batch_normalization:
        conv_z = BatchNormalization(axis=1)(conv_z)
    elif instance_normalization:
        conv_z = InstanceNormalization(axis=1)(conv_z)
    if activation is None:
        conv_z = Activation('relu')(conv_z)
    else:
        conv_z = activation()(conv_z)
    return conv_z

def create_next_block(input_layer, n_filters, batch_normalization=False, activation=None,
                      padding='same', strides=(1, 1, 1)):
	
    layer1 = create_convolution_block(input_layer=input_layer,
                                      kernel=(1, 1, 1),
                                      n_filters=n_filters,
                                      padding=padding,
                                      batch_normalization=batch_normalization,
                                      activation=activation)
    layer2 = create_convolution_block(input_layer=layer1,
                                      kernel=(3, 3, 3),
                                      strides = strides,
                                      n_filters=n_filters,
                                      padding=padding,
                                      batch_normalization=batch_normalization,
                                      activation=activation)
    layer3 = create_convolution_block(input_layer=layer2,
                                      kernel=(1, 1, 1),
                                      n_filters=n_filters*2,
                                      padding=padding,
                                      batch_normalization=batch_normalization,
                                      activation=None)
    return layer3

def create_elewise_block(input1, input2, activation=None):
    summation_layer = Add()([input1, input2])
    summation_layer = activation()(summation_layer)
    return summation_layer

def res_next32(input_shape, initial_learning_rate=0.00001, batch_normalization=True,
               activation_name="sigmoid", activation=ReLU, opt='Adam'):
    base_filters = 16
    inputs = Input(input_shape)
    current_layer = inputs
    print(current_layer._keras_shape)
    layer1 = create_convolution_block(input_layer=current_layer,
                                      kernel=(3, 3, 3),
                                      n_filters=base_filters,
                                      padding='same',
                                      batch_normalization=batch_normalization,
                                      activation=activation)
    print(layer1._keras_shape)
    #############
    resx1 = create_next_block(input_layer=layer1,
                              n_filters=base_filters*2,
                              padding='same',
                              batch_normalization=batch_normalization,
                              activation=activation)
    rex1_match = create_convolution_block(input_layer=layer1,
                                   		  kernel=(1, 1, 1),
                                          n_filters=base_filters*4,
                                          padding='same',
                                          batch_normalization=batch_normalization,
                                          activation=None)
    rex1_out = create_elewise_block(input1=resx1,
    								input2=rex1_match,
    								activation=activation)
    #############
    resx2 = create_next_block(input_layer=rex1_out,
                              n_filters=base_filters*2,
                              padding='same',
                              batch_normalization=batch_normalization,
                              activation=activation)
    rex2_out = create_elewise_block(input1=resx2,
    								input2=rex1_out,
    								activation=activation)
    print(rex2_out._keras_shape)
    #############
    resx3 = create_next_block(input_layer=rex2_out,
                              n_filters=base_filters*4,
                              strides=(2, 2, 2),
                              padding='same',
                              batch_normalization=batch_normalization,
                              activation=activation)
    rex3_match = create_convolution_block(input_layer=rex2_out,
                                   		  kernel=(1, 1, 1),
                                   		  strides=(2, 2, 2),
                                          n_filters=base_filters*8,
                                          padding='same',
                                          batch_normalization=batch_normalization,
                                          activation=None)
    rex3_out = create_elewise_block(input1=resx3,
    								input2=rex3_match,
    								activation=activation)
    #############
    resx4 = create_next_block(input_layer=rex3_out,
                              n_filters=base_filters*4,
                              padding='same',
                              batch_normalization=batch_normalization,
                              activation=activation)
    rex4_out = create_elewise_block(input1=resx4,
    								input2=rex3_out,
    								activation=activation)
    print(rex4_out._keras_shape)
    #############
    resx5 = create_next_block(input_layer=rex4_out,
                              n_filters=base_filters*8,
                              strides=(2, 2, 2),
                              padding='same',
                              batch_normalization=batch_normalization,
                              activation=activation)
    rex5_match = create_convolution_block(input_layer=rex4_out,
                                   		  kernel=(1, 1, 1),
                                   		  strides=(2, 2, 2),
                                          n_filters=base_filters*16,
                                          padding='same',
                                          batch_normalization=batch_normalization,
                                          activation=None)
    rex5_out = create_elewise_block(input1=resx5,
    								input2=rex5_match,
    								activation=activation)
    #############
    resx6 = create_next_block(input_layer=rex5_out,
                              n_filters=base_filters*8,
                              padding='same',
                              batch_normalization=batch_normalization,
                              activation=activation)
    rex6_out = create_elewise_block(input1=resx6,
    								input2=rex5_out,
    								activation=activation)
    print(rex6_out._keras_shape)
    #############
    resx7 = create_next_block(input_layer=rex6_out,
                              n_filters=base_filters*16,
                              strides=(2, 2, 2),
                              padding='same',
                              batch_normalization=batch_normalization,
                              activation=activation)
    rex7_match = create_convolution_block(input_layer=rex6_out,
                                   		  kernel=(1, 1, 1),
                                   		  strides=(2, 2, 2),
                                          n_filters=base_filters*32,
                                          padding='same',
                                          batch_normalization=batch_normalization,
                                          activation=None)
    rex7_out = create_elewise_block(input1=resx7,
    								input2=rex7_match,
    								activation=activation)
    #############
    resx8 = create_next_block(input_layer=rex7_out,
                              n_filters=base_filters*16,
                              padding='same',
                              batch_normalization=batch_normalization,
                              activation=activation)
    rex8_out = create_elewise_block(input1=resx8,
    								input2=rex7_out,
    								activation=activation)
    print(rex8_out._keras_shape)
    layer7 = GlobalMaxPooling3D(data_format="channels_first")(rex8_out)
    print(layer7._keras_shape)
    layer7 = Dropout(rate=0.3)(layer7)
    layer8 = Dense(1, activation='sigmoid')(layer7)
    print(layer8._keras_shape)
    model = Model(inputs=inputs, outputs=layer8)
    if opt == 'Adam':
    	model.compile(optimizer=Adam(lr=initial_learning_rate), loss=[focal_loss_fixed], metrics=['accuracy'])
    elif opt == 'SGD':
        model.compile(optimizer=SGD(lr=initial_learning_rate),  loss=[focal_loss_fixed], metrics=['accuracy'])
    return model
