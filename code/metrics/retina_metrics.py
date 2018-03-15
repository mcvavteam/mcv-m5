import tensorflow as tf
import numpy as np
import keras
from tensorflow.python.framework import ops, dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
from tensorflow.python.ops.nn_ops import _flatten_outer_dims


def RetinaLoss(input_shape=(3, 640, 640), num_classes=45,
               priors=[[0.25, 0.25], [0.5, 0.5], [1.0, 1.0], [1.7, 1.7], [2.5, 2.5]], max_truth_boxes=30, thresh=0.6,
               object_scale=5.0, noobject_scale=1.0, coord_scale=1.0, class_scale=1.0):
    # Def custom loss function using numpy
    def _RetinaLoss(y_true, y_pred, name=None, priors=priors):
        net_out = tf.transpose(y_pred, perm=[0, 2, 3, 1])

        _, h, w, c = net_out.get_shape().as_list()
        b = len(priors)
        anchors = np.array(priors)

        _probs, _confs, _coord, _areas, _upleft, _botright = tf.split(y_true, [num_classes, 1, 4, 1, 2, 2], axis=3)
        _confs = tf.squeeze(_confs, 3)
        _areas = tf.squeeze(_areas, 3)

        class_subnet_out, bbox_subnet_out = tf.split(net_out, [b * (num_classes), b * (4 + 1)], axis=3)

        class_subnet_out_reshape = tf.reshape(class_subnet_out, [-1, h, w, b, num_classes])
        bbox_subnet_out_reshape = tf.reshape(bbox_subnet_out, [-1, h, w, b, 4 + 1])

        # Extract the coordinate prediction from net.out
        coords = bbox_subnet_out_reshape[:, :, :, :, :4]
        coords = tf.reshape(coords, [-1, h * w, b, 4])
        adjusted_coords_xy = tf.sigmoid(coords[:, :, :, 0:2])
        adjusted_coords_wh = tf.exp(tf.clip_by_value(coords[:, :, :, 2:4], -1e3, 10)) * np.reshape(anchors, [1, 1, b,
                                                                                                             2]) / np.reshape(
            [w, h], [1, 1, 1, 2])
        adjusted_coords_wh = tf.sqrt(tf.clip_by_value(adjusted_coords_wh, 1e-10, 1e5))
        coords = tf.concat([adjusted_coords_xy, adjusted_coords_wh], 3)

        adjusted_c = tf.sigmoid(bbox_subnet_out_reshape[:, :, :, :, 4])
        adjusted_c = tf.reshape(adjusted_c, [-1, h * w, b, 1])

        adjusted_prob = tf.nn.softmax(class_subnet_out_reshape)
        adjusted_prob = tf.reshape(adjusted_prob, [-1, h * w, b, num_classes])

        adjusted_bbox_subnet_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c], 3)
        adjusted_class_subnet_out = adjusted_prob
        # adjusted_net_out = tf.concat([adjusted_coords_xy, adjusted_coords_wh, adjusted_c, adjusted_prob], 3)

        wh = tf.square(coords[:, :, :, 2:4]) * np.reshape([w, h], [1, 1, 1, 2])
        area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
        centers = coords[:, :, :, 0:2]
        floor = centers - (wh * .5)
        ceil = centers + (wh * .5)

        # calculate the intersection areas
        intersect_upleft = tf.maximum(floor, _upleft)
        intersect_botright = tf.minimum(ceil, _botright)
        intersect_wh = intersect_botright - intersect_upleft
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect = tf.multiply(intersect_wh[:, :, :, 0], intersect_wh[:, :, :, 1])

        # calculate the best IOU, set 0.0 confidence for worse boxes
        iou = tf.truediv(intersect, _areas + area_pred - intersect)
        best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
        # best_box = tf.grater_than(iou, 0.5) # LLUIS ???
        best_box = tf.to_float(best_box)
        confs = tf.multiply(best_box, _confs)

        # take care of the weight terms
        conid = noobject_scale * (1. - confs) + object_scale * confs
        weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)
        cooid = coord_scale * weight_coo
        weight_pro = tf.concat(num_classes * [tf.expand_dims(confs, -1)], 3)
        proid = class_scale * weight_pro

        true = tf.concat([_coord, tf.expand_dims(confs, 3), _probs], 3)
        true_class = _probs
        true_bbox = tf.concat([_coord, tf.expand_dims(confs, 3)], 3)

        # L1
        loss_bbox = tf.abs(true_bbox - adjusted_bbox_subnet_out)
        loss_bbox = tf.reduce_sum(loss_bbox, axis=3) # Sum by parameters
        loss_bbox = tf.reduce_sum(loss_bbox, axis=2) # Sum by anchors

        # Focal loss
        # Cross entropy: -ylog(y^)-(1-y)log(1-y^)
        gamma=2
        alpha=.25
        loss_class_one = - tf.multiply(true_class, tf.log(adjusted_class_subnet_out))
        loss_class_one = tf.multiply(tf.pow(1-adjusted_class_subnet_out,gamma),loss_class_one)
        loss_class_one = tf.multiply(alpha,loss_class_one)
        loss_class_zero = - tf.multiply(1 - true_class, tf.log(1 - adjusted_class_subnet_out))
        loss_class_zero = tf.multiply(tf.pow(adjusted_class_subnet_out, gamma), loss_class_zero)
        loss_class_zero = tf.multiply(1-alpha, loss_class_zero)
        loss_class = loss_class_one + loss_class_zero
        loss_class = tf.reduce_sum(loss_class, axis=3)  # Sum by class
        loss_class = tf.reduce_sum(loss_class, axis=2)  # Sum by anchors

        # The training loss is the sum the focal loss and the standard smooth L1 loss used for box regression
        loss = loss_bbox + loss_class
        loss = tf.reduce_sum(loss,axis=1) # 1 scalar per image

        return tf.reduce_mean(loss, axis=0)  # 1 scalar per full batch

    return _RetinaLoss


"""
def focal_loss(y_true, y_pred,alpha=0.25, gamma=2.0):
    labels         = y_true
    classification = y_pred
    # compute the divisor: for each image in the batch, we want the number of positive anchors

    # clip the labels to 0, 1 so that we ignore the "ignore" label (-1) in the divisor
    divisor = tf.where(keras.backend.less_equal(labels, 0), keras.backend.zeros_like(labels), labels)
    divisor = keras.backend.max(divisor, axis=2, keepdims=True)
    divisor = keras.backend.cast(divisor, keras.backend.floatx())

    # compute the number of positive anchors
    divisor = keras.backend.sum(divisor, axis=1, keepdims=True)

    #  ensure we do not divide by 0
    divisor = keras.backend.maximum(1.0, divisor)

    # compute the focal loss
    alpha_factor = keras.backend.ones_like(labels) * alpha
    alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
    focal_weight = alpha_factor * focal_weight ** gamma

    cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

    # normalise by the number of positive anchors for each entry in the minibatch
    cls_loss = cls_loss / divisor

    # filter out "ignore" anchors
    anchor_state = keras.backend.max(labels, axis=2)  # -1 for ignore, 0 for background, 1 for object
    indices      = tf.where(keras.backend.not_equal(anchor_state, -1))

    cls_loss = tf.gather_nd(cls_loss, indices)

    # divide by the size of the minibatch
    return keras.backend.sum(cls_loss) / keras.backend.cast(keras.backend.shape(labels)[0], keras.backend.floatx())
"""
