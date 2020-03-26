from typing import Tuple, List

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import Conv2D, UpSampling2D, Concatenate, \
    MaxPooling2D, Softmax, BatchNormalization


class UNet:

    def __init__(self, num_classes: int):
        self.__num_classes = num_classes

    def build_model(self, input_shape: Tuple[int, int, int]) -> Model:
        x = Input(shape=input_shape, name="input")
        sub_sampling_out, sub_sampling_stack = self.__build_sub_sampling_stack(
            x=x
        )
        filter_extraction_conv = Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            padding="same",
            name="filter_extraction_conv"
        )(sub_sampling_out)
        filter_extraction_conv_bn = BatchNormalization(
            name=f"filter_extraction_conv_bn"
        )(filter_extraction_conv)
        up_sampling_input = Conv2D(
            filters=512,
            kernel_size=(3, 3),
            padding="same",
            name="up_sampling_input"
        )(filter_extraction_conv_bn)
        up_sampling_output = self.__build_up_sampling_stack(
            x=up_sampling_input,
            sub_sampling_stack=sub_sampling_stack
        )
        out = Conv2D(
            filters=self.__num_classes,
            kernel_size=(1, 1),
            padding="same",
            name="output"
        )(up_sampling_output)
        flatten_out = Softmax(name="output_soft_max")(out)
        return Model(x, flatten_out)

    def __build_sub_sampling_stack(self,
                                   x: tf.Tensor
                                   ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        filters = [64, 128, 256, 512]
        sub_sampling_stack = []
        for idx, stack_filters in enumerate(filters):
            x = self.__build_triple_conv_block(
                x=x,
                num_filters=stack_filters,
                stack_name=f"sub_stack_{idx+1}"
            )
            sub_sampling_stack.append(x)
            x = MaxPooling2D(
                pool_size=(2, 2),
                name=f"sub_stack_{idx+2}_input"
            )(x)
        return x, sub_sampling_stack

    def __build_triple_conv_block(self,
                                  x: tf.Tensor,
                                  num_filters: int,
                                  stack_name: str
                                  ) -> tf.Tensor:
        for i in range(3):
            x = Conv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                padding="same",
                name=f"{stack_name}_{i+1}"
            )(x)
        x = BatchNormalization(name=f"{stack_name}_bn")(x)
        return x

    def __build_up_sampling_stack(self,
                                  x: tf.Tensor,
                                  sub_sampling_stack: List[tf.Tensor]
                                  ) -> tf.Tensor:
        for idx, sub_sampling_layer in enumerate(sub_sampling_stack[::-1]):
            x = self.__build_up_sampling_conv_block(
                smaller_input=x,
                bigger_input=sub_sampling_layer,
                block_idx=idx+1
            )
            x = BatchNormalization(name=f"up_sample_{idx+1}_bn")(x)
        return x

    def __build_up_sampling_conv_block(self,
                                       smaller_input: tf.Tensor,
                                       bigger_input: tf.Tensor,
                                       block_idx: int
                                       ) -> tf.Tensor:
        up_sampled_smaller_input = UpSampling2D(size=(2, 2))(smaller_input)
        bottleneck_conv = Conv2D(
            filters=up_sampled_smaller_input.shape[-1].value // 2,
            kernel_size=(2, 2),
            padding="same",
            name=f"up_sample_bottleneck_conv_{block_idx}"
        )(up_sampled_smaller_input)
        x = Concatenate(name=f"up_sample_concat_{block_idx}")(
            [bottleneck_conv, bigger_input]
        )
        conv_3x3_filters = x.shape[-1].value // 2
        for idx in range(2):
            x = Conv2D(
                filters=conv_3x3_filters,
                kernel_size=(3, 3),
                padding="same",
                name=f"up_sample_{block_idx}_conv_3x3_{idx+1}"
            )(x)
        return x
