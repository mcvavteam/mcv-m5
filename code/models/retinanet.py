import keras
from keras import backend as K, Input
from keras.layers import Conv2D, Concatenate, Reshape, UpSampling2D

from keras.models import Model
import tensorflow as tf
import keras_resnet
import keras_resnet.models

def build_retinanet(img_shape=(3, 416, 416), n_classes=80, n_priors=5,
               load_pretrained=False,freeze_layers_from='base_model'):

    # RetinaNet model is only implemented for TF backend
    assert(K.backend() == 'tensorflow')

    inputs=Input(shape=img_shape)

    K.set_image_dim_ordering('th')

    backbone_net = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    _, C3, C4, C5 = backbone_net.outputs  # we ignore C2

    K.set_image_dim_ordering('tf')

    C3 = keras.layers.Permute((3, 2, 1))(C3)
    C4 = keras.layers.Permute((3, 2, 1))(C4)
    C5 = keras.layers.Permute((3, 2, 1))(C5)

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)

    # create regression and classification submodels
    bbox_subnet_model = regression_submodel(n_priors)
    class_subnet_model = classification_submodel(n_classes, n_priors)

    #Concatenation of pyramid levels
    # class_subnet
    class_subnet = [class_subnet_model(f) for f in features]

    class_subnet0 = Conv2D(filters=n_priors*n_classes, kernel_size=(1, 1), strides=(4, 4))(class_subnet[0])
    class_subnet1 = Conv2D(filters=n_priors*n_classes, kernel_size=(1, 1), strides=(2, 2))(class_subnet[1])
    class_subnet2 = class_subnet[2]
    class_subnet3 = UpSampling2D(size=(2, 2), name='class_upsample')(class_subnet[3])

    class_subnet = [class_subnet0,class_subnet1,class_subnet2,class_subnet3]
    class_subnet = Concatenate(axis=3, name='class_subnet_outputs')(class_subnet)
    class_subnet = Conv2D(filters=n_priors*n_classes, kernel_size=(1, 1), strides=(1, 1))(class_subnet)

    # bbox_subnet
    bbox_subnet = [bbox_subnet_model(f) for f in features]

    bbox_subnet0 = Conv2D(filters=n_priors * (4 + 1), kernel_size=(1, 1), strides=(4, 4))(bbox_subnet[0])
    bbox_subnet1 = Conv2D(filters=n_priors * (4 + 1), kernel_size=(1, 1), strides=(2, 2))(bbox_subnet[1])
    bbox_subnet2 = bbox_subnet[2]
    bbox_subnet3 = UpSampling2D(size=(2, 2), name='bbox_upsample')(bbox_subnet[3])

    bbox_subnet = [bbox_subnet0, bbox_subnet1, bbox_subnet2, bbox_subnet3]
    bbox_subnet = Concatenate(axis=3, name='bbox_subnet_outputs')(bbox_subnet)
    bbox_subnet = Conv2D(filters=n_priors*(4+1), kernel_size=(1, 1), strides=(1, 1))(bbox_subnet)

    # Output
    #  - n_priors: anchors
    #  - 4: offsets
    #  - n_classes: probability for each class
    #  - 1: confidence of having an object
    output = Concatenate(axis=3)([class_subnet, bbox_subnet])

    K.set_image_dim_ordering('th')

    output = keras.layers.Permute((3, 2, 1))(output)

    model = Model(inputs=inputs, outputs=output)

    return model

def classification_submodel(num_classes, num_anchors, pyramid_feature_size=256, classification_feature_size=256, name='classification_submodel'):
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=keras.initializers.zeros(),
        name='pyramid_classification',
        **options
    )(outputs)

    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def regression_submodel(num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors*(4+1), name='pyramid_regression', **options)(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def create_pyramid_features(C3, C4, C5, feature_size=256):
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return P3, P4, P5, P6, P7

class UpsampleLike(keras.layers.Layer):
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return tf.image.resize_images(source, (target_shape[1], target_shape[2]))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)