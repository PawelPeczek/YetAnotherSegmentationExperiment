from typing import Tuple

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Conv2D, UpSampling2D,\
    Softmax, BatchNormalization, Add

from src.models.base import SegmentationModel


class ICNet(SegmentationModel):

    def build_model(self, input_shape: Tuple[int, int, int]) -> Model:
        x = Input(shape=input_shape, name="input")
        big_branch_out = self.__build_big_output_head(x=x)
        sub_sampled_branches_out = self.__build_sub_sampled_branches_out(x=x)
        medium_branch_up = UpSampling2D(size=(2, 2), name="medium_branch_up")(
            sub_sampled_branches_out
        )
        medium_branch_up_refine = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            dilation_rate=(2, 2),
            activation="relu",
            name="medium_branch_up_refine"
        )(medium_branch_up)
        fuse_add = Add(name="big_medium_fuse_add")(
            [big_branch_out, medium_branch_up_refine]
        )
        fuse_add_bn = BatchNormalization(name=f"big_medium_fuse_add_bn")(
            fuse_add
        )
        cls_conv = Conv2D(
            filters=self._num_classes,
            kernel_size=(1, 1),
            padding="same",
            name="output"
        )(fuse_add_bn)
        flatten_out = Softmax(name="output_soft_max")(cls_conv)
        return Model(x, flatten_out)

    def __build_big_output_head(self, x: tf.Tensor) -> tf.Tensor:
        for idx, layer_filter in enumerate([16, 16, 32]):
            x = Conv2D(
                filters=layer_filter,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name=f"big_conv_{idx}"
            )(x)
            x = BatchNormalization(
                name=f"big_conv_{idx}_bn"
            )(x)
        return x

    def __build_sub_sampled_branches_out(self, x: tf.Tensor) -> tf.Tensor:
        x = Conv2D(
            filters=16,
            kernel_size=(3, 3),
            padding="same",
            strides=(2, 2),
            activation="relu",
            name="half_x_sub_sampling_conv"
        )(x)
        for idx, layer_filter in enumerate([64, 64, 64]):
            x = Conv2D(
                filters=layer_filter,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name=f"medium_conv_{idx}"
            )(x)
            x = BatchNormalization(
                name=f"medium_conv_{idx}_bn"
            )(x)
        medium_branch_out = x
        x = Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            strides=(2, 2),
            activation="relu",
            name="quater_x_sub_sampling_conv"
        )(medium_branch_out)
        for idx, layer_filter in enumerate([128, 128, 256]):
            x = Conv2D(
                filters=layer_filter,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                name=f"small_conv_{idx}"
            )(x)
            x = BatchNormalization(
                name=f"small_conv_{idx}_bn"
            )(x)
        small_branch_up = UpSampling2D(size=(2, 2), name="small_branch_up")(x)
        small_branch_up_refine = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            dilation_rate=(2, 2),
            activation="relu",
            name="small_branch_up_refine"
        )(small_branch_up)
        fuse_add = Add(name="medium_small_fuse_add")(
            [medium_branch_out, small_branch_up_refine]
        )
        fuse_result = Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            dilation_rate=(2, 2),
            activation="relu",
            name="medium_small_fuse_refine"
        )(fuse_add)
        return BatchNormalization(name=f"medium_small_fuse_bn")(fuse_result)
