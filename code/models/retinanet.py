import keras
from keras import backend as K, Input
from keras.layers import Conv2D, Concatenate

from keras.models import Model
import tensorflow as tf



def build_retinanet(img_shape=(3, 416, 416), n_classes=80, n_priors=5,
               load_pretrained=False,freeze_layers_from='base_model'):

    # RetinaNet model is only implemented for TF backend
    assert(K.backend() == 'tensorflow')

    K.set_image_dim_ordering('th')
    """
    backbone_net = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)

    _, C3, C4, C5 = backbone_net.outputs  # we ignore C2

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)
    """

    inputs=Input(shape=img_shape)
    class_subnet = Conv2D(filters=n_priors*(n_classes),kernel_size=(32,32),strides=(32,32))(inputs)

    bbox_subnet = Conv2D(filters=n_priors*(4+1),kernel_size=(32,32),strides=(32,32))(inputs)
    # Output
    # n_priors: anchors, 4: offsets, n_classes: probability for each class, 1: confidence of having an object
    output = Concatenate(axis=1)([class_subnet, bbox_subnet])
    model = Model(inputs=inputs, outputs=output)

    return model

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