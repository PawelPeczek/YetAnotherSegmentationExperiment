import tensorflow.python.keras.backend as K
from tensorflow.python.keras.losses import categorical_crossentropy


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def ce_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))
