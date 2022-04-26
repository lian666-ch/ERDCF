import tensorflow as tf
from keras import backend as K


def pixel_error(y_true, y_pred):

    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32, name='predictions')
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)
    return tf.reduce_mean(error, name='pixel_error')

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def F_score(y_true, y_pred):#

    pred = tf.cast(tf.greater(y_pred, 0.1), tf.float32)#
    y_true_f = K.flatten(y_true)#TP+FN
    y_pred_f = K.flatten(pred)#TP+FP
    intersection = K.sum(y_true_f * y_pred_f)  # TP
    F_score = 2 * intersection / (K.sum(y_true_f) + K.sum(y_pred_f))#2*TP/(2*TP+FN+FP)
    return F_score

