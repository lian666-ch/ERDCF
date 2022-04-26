
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.layers import Concatenate, Activation, Add, MaxPooling2D, BatchNormalization
from keras.models import Model
from model.ResNet import identity_block, conv_block

def deconv_layer(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)
    kernel_size = (2 * factor, 2 * factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
    return x

def ERDCF ():

    inputs = Input(shape=(None,None, 3))
    x = inputs  #
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(x)
    x = BatchNormalization(axis=-1, name='bn_conv1')(x)#256
    x = Activation('relu', name='act1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)

    # Block 1
    x1_conv1 = conv_block(x, (64, 64), stage=1, block='a', strides=(1, 1))#64
    x1_conv1_out = Conv2D(21, (1, 1),  padding='same', name='block1_conv_o1')(x1_conv1)
    x1_conv2 = identity_block(x1_conv1, (64, 64), stage=1, block='b')
    x1_conv2_out = Conv2D(21, (1, 1),  padding='same', name='block1_conv_o2')(x1_conv2)
    x1_conv3 = identity_block(x1_conv2, (64, 64), stage=1, block='c')
    x1_conv3_out = Conv2D(21, (1, 1),  padding='same', name='block1_conv_o3')(x1_conv3)
    x1_add = Add()([x1_conv1_out, x1_conv2_out,x1_conv3_out])
    b1 = deconv_layer(x1_add, 4)

    # Block 2
    x2_conv1 = conv_block(x1_conv3, (128, 128), stage=2, block='a', strides=(2, 2))
    x2_conv1_out = Conv2D(21, (1, 1),  padding='same', name='block2_conv_o1')(x2_conv1)
    x2_conv2 = identity_block(x2_conv1, (128, 128), stage=2, block='b')
    x2_conv2_out = Conv2D(21, (1, 1),  padding='same', name='block2_conv_o2')(x2_conv2)
    x2_conv3 = identity_block(x2_conv2, (128, 128), stage=2, block='c')
    x2_conv3_out = Conv2D(21, (1, 1),  padding='same', name='block2_conv_o3')(x2_conv3)
    x2_conv4 = identity_block(x2_conv3, (128, 128), stage=2, block='d')
    x2_conv4_out = Conv2D(21, (1, 1),  padding='same', name='block2_conv_o4')(x2_conv4)
    x2_add = Add()([x2_conv1_out, x2_conv2_out, x2_conv3_out,x2_conv4_out])
    b2 = deconv_layer(x2_add, 8)

    # Block 3
    x3_conv1 = conv_block(x2_conv4, (256, 256), stage=3, block='a', strides=(2, 2))
    x3_conv1_out = Conv2D(21, (1, 1),  padding='same', name='block3_conv_o1')(x3_conv1)
    x3_conv2 = identity_block(x3_conv1, (256, 256), stage=3, block='b')
    x3_conv2_out = Conv2D(21, (1, 1),  padding='same', name='block3_conv_o2')(x3_conv2)
    x3_conv3 = identity_block(x3_conv2, (256, 256), stage=3, block='c')
    x3_conv3_out = Conv2D(21, (1, 1),  padding='same', name='block3_conv_o3')(x3_conv3)
    x3_add = Add()([x3_conv1_out, x3_conv2_out, x3_conv3_out])
    b3 = deconv_layer(x3_add, 16)

    # fuse
    fuse = Concatenate(axis=-1)([b1, b2, b3])
    fuse = Conv2D(1, (1,1), padding='same', use_bias=False, activation=None)(fuse)

    # outputs
    o1    = Activation('sigmoid', name='o1')(b1)
    o2    = Activation('sigmoid', name='o2')(b2)
    o3    = Activation('sigmoid', name='o3')(b3)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    # model
    model = Model(inputs=[inputs], outputs=[o1, o2, o3, ofuse])

    return model