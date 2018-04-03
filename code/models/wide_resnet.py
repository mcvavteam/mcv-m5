# Keras imports
from keras import layers
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

from keras.applications import ResNet50
from keras.models import *
from keras.layers import *
import keras.backend as K


# Paper: https://arxiv.org/abs/1611.10080
# Original caffe code: https://github.com/itijyou/ademxapp

def build_wide_resnet(img_shape=(360, 480, 3), nclasses=8, l2_reg=0.,
                      init='glorot_uniform', path_weights=None,
                      load_pretrained=False, freeze_layers_from=None):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        # weights = 'imagenet'
        print('THERE ARE NOT PRETRAINED WEIGHTS')
        exit(-1)
    else:
        weights = None

    # Create the first convolutional layer
    input = Input(shape=img_shape)
    x = Conv2D(filters=64, kernel_size=(3, 3), use_bias=False,
               padding='same', kernel_regularizer=regularizers.l2(l2_reg), name="conv1a")(input)

    # Implementing Model A of the paper
    # 224-0-3-3-6-3-1-1
    # Block 2 (3 times)
    block_filters = [128, 128]
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[1, 1],
              name='block2_1')
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[1, 1],
              name='block2_2')
    x = block(x, filters_vector=block_filters,
              strides=2,
              dilations_vector=[1, 1],
              name='block2_3')

    # Block 3 (3 times)
    block_filters = [256, 256]
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[1, 1],
              name='block3_1')
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[1, 1],
              name='block3_2')
    x = block(x, filters_vector=block_filters,
              strides=2,
              dilations_vector=[1, 1],
              name='block3_3')

    # Block 4 (6 times)
    block_filters = [512, 512]
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[1, 1],
              name='block4_1')
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[1, 1],
              name='block4_2')
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[1, 1],
              name='block4_3')
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[1, 1],
              name='block4_4')
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[1, 1],
              name='block4_5')
    x = block(x, filters_vector=block_filters,
              strides=2,
              dilations_vector=[1, 1],
              name='block4_6')

    # Block 5 (3 times)
    block_filters = [512, 1024]
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[2, 2],
              name='block5_1')
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[2, 2],
              name='block5_2')
    x = block(x, filters_vector=block_filters,
              strides=1,
              dilations_vector=[2, 2],
              name='block5_3')

    # Block 6 (1 time)
    block_filters = [512, 1024, 2048]
    block_dropout = [0, 0, .3]
    x = bottleneck_block(x, filters_vector=block_filters,
                         strides=1,
                         dilations_vector=[4, 4, 4],
                         dropout_vector=block_dropout,
                         name='block6_1')

    # Block 7 (1 time)
    block_filters = [1024, 2048, 4098]
    block_dropout = [0, .3, .5]
    x = bottleneck_block(x, filters_vector=block_filters,
                         strides=1,
                         dilations_vector=[4, 4, 4],
                         dropout_vector=block_dropout,
                         name='block7_1')

    # Classifier (2 Conv2D based)
    x = conv_stage(x, filters=512,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   dilation_rate=(12, 12),
                   dropout=0,
                   name='classifier_hidden_cs')
    x = conv_stage(x, filters=nclasses,
                   kernel_size=(3, 3),
                   strides=(1, 1),
                   dilation_rate=(12, 12),
                   dropout=0,
                   name='classifier_classif_cs')

    # Upsampling
    x = transp_conv_stage(x, filters=nclasses,
                          kernel_size=(16, 16),
                          strides=(8, 8),
                          name='upsampling_deconv')

    # Crop the borders to fit original size
    curlayer_output_shape = Model(inputs=input, outputs=x).output_shape
    if K.image_dim_ordering() == 'tf':
        curlayer_output_height = curlayer_output_shape[1]
        curlayer_output_width = curlayer_output_shape[2]
    else:
        curlayer_output_height = curlayer_output_shape[2]
        curlayer_output_width = curlayer_output_shape[3]

    x = Cropping2D(cropping=((curlayer_output_height - img_shape[0]) / 2,
                             (curlayer_output_width - img_shape[1]) / 2))(x)

    # Reshape to vector
    curlayer_output_shape = Model(inputs=input, outputs=x).output_shape
    if K.image_dim_ordering() == 'tf':
        curlayer_output_height = curlayer_output_shape[1]
        curlayer_output_width = curlayer_output_shape[2]
    else:
        curlayer_output_height = curlayer_output_shape[2]
        curlayer_output_width = curlayer_output_shape[3]
    x = Reshape(target_shape=(curlayer_output_height*curlayer_output_width, nclasses))(x)

    # Apply softmax
    out = Activation('softmax')(x)


    model = Model(inputs=input, outputs=out)
    model.summary()
    # TODO Freeze some layers
    #if freeze_layers_from is not None:
    #    freeze_layers(model, freeze_layers_from)

    return model


def block(input_tensor, filters_vector, strides, dilations_vector, dropout_vector=(0, 0), l2_reg=0, name=''):
    if len(filters_vector) != 2:
        raise ValueError('filters_vector not of length 2')
    if len(dilations_vector) != 2:
        raise ValueError('dilations_vector not of length 2')
    if len(dropout_vector) != 2:
        raise ValueError('dropout_vector not of length 2')

    x = conv_stage(input_tensor, filters=filters_vector[0],
                   kernel_size=(3, 3),
                   strides=strides,
                   dilation_rate=dilations_vector[0],
                   dropout=dropout_vector[0],
                   name=name + '_cs0')
    x = conv_stage(x, filters=filters_vector[1],
                   kernel_size=(3, 3),
                   dilation_rate=dilations_vector[1],
                   dropout=dropout_vector[1],
                   name=name + '_cs1')

    dim_match = input_tensor.shape[3] == filters_vector[-1] and (strides == (1, 1) or strides == 1)
    if dim_match:
        x = layers.add([x, input_tensor])
    else:
        x = BatchNormalization(axis=1 if K.image_dim_ordering() == 'channels_first' else 3, name=name + '_shortcut_bn')(
            x)
        x = Activation('relu', name=name + '_shortcut_relu')(x)
        shortcut = Conv2D(filters_vector[-1], (1, 1), strides=strides,
                          name=name + '_shortcut_conv')(input_tensor)
        x = layers.add([x, shortcut])
    return x


def bottleneck_block(input_tensor, filters_vector, strides, dilations_vector, dropout_vector=(0, 0, 0), l2_reg=0,
                     name=''):
    if len(filters_vector) != 3:
        raise ValueError('filters_vector not of length 3')
    if len(dilations_vector) != 3:
        raise ValueError('dilations_vector not of length 3')
    if len(dropout_vector) != 3:
        raise ValueError('dropout_vector not of length 3')

    x = conv_stage(input_tensor, filters=filters_vector[0],
                   kernel_size=(1, 1),
                   strides=strides,
                   dilation_rate=dilations_vector[0],
                   dropout=dropout_vector[0],
                   name=name + '_cs0')
    x = conv_stage(x, filters=filters_vector[1],
                   kernel_size=(3, 3),
                   dilation_rate=dilations_vector[1],
                   dropout=dropout_vector[1],
                   name=name + '_cs1')
    x = conv_stage(x, filters=filters_vector[2],
                   kernel_size=(1, 1),
                   dilation_rate=dilations_vector[2],
                   dropout=dropout_vector[2],
                   name=name + '_cs2')

    dim_match = input_tensor.shape[3] == filters_vector[-1] and (strides == (1, 1) or strides == 1)
    if dim_match:
        x = layers.add([x, input_tensor])
    else:
        x = BatchNormalization(axis=1 if K.image_dim_ordering() == 'channels_first' else 3, name=name + '_shortcut_bn')(
            x)
        x = Activation('relu', name=name + '_shortcut_relu')(x)
        shortcut = Conv2D(filters_vector[-1], (1, 1), strides=strides,
                          name=name + '_shortcut_conv')(input_tensor)
        x = layers.add([x, shortcut])
    return x


def conv_stage(x, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), dropout=0, l2_reg=0, name=''):
    x = BatchNormalization(axis=1 if K.image_dim_ordering() == 'channels_first' else 3, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               dilation_rate=dilation_rate,
               use_bias=False,
               kernel_regularizer=regularizers.l2(l2_reg),
               name=name + '_conv')(x)
    if dropout > 0:
        x = Dropout(rate=dropout)(x)
    return x


def transp_conv_stage(x, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), dropout=0, l2_reg=0, name=''):
    x = BatchNormalization(axis=1 if K.image_dim_ordering() == 'channels_first' else 3, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        dilation_rate=dilation_rate,
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(l2_reg),
                        name=name + '_conv')(x)
    if dropout > 0:
        x = Dropout(rate=dropout)(x)
    return x


# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = -1  # TODO change this number

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True
