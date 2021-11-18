from typing import List

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16


class VGG16_Patch(keras.Model):
    layer_name: List[str] = [
        "block1_conv1",
        "block1_conv2",
        "block2_conv1",
        "block2_conv2",
        "block3_conv1",
        "block3_conv2",
        "block3_conv3",
        "block4_conv1", #optional
        "block4_conv2", #optional
        "block4_conv3",
        "block5_conv1", #optional
        "block5_conv2", #optional
        "block5_conv3"]

    def __init__(self, optimize_mode: str, use_all_features: bool):
        super().__init__()
        if optimize_mode != 'caffe':
            weights = './sources/vgg_norm.h5'
        else:
            weights = 'imagenet'
        vgg_base = VGG16(include_top=False, weights=weights)
        vgg_base.trainable = False

        if not use_all_features:
            exclude = ["block4_conv1","block4_conv2","block5_conv1","block5_conv2"]
            use_layers = [ln for ln in self.layer_name if ln not in exclude]
        else:
            use_layers = self.layer_name.copy()
        outputs = [vgg_base.get_layer(name).output for name in use_layers]

        self.vgg = Model(inputs=vgg_base.inputs, outputs=outputs)
        self.dims = 3 + sum(o.shape[-1] for o in outputs) # 2179
        self.layer_counts = len(use_layers) + 1

        if optimize_mode == 'torch_uniform':
            self.preprocess_function = self.uniform_prerpocess
        elif optimize_mode == 'caffe':
            self.preprocess_function = self.caffe_preprocess
        else:
            self.preprocess_function = self.torch_preprocess

    def uniform_prerpocess(self, inputs: tf.Tensor) -> tf.Tensor:
        x = (inputs + 1.) / 2.
        return self.torch_preprocess(x)

    def caffe_preprocess(self, inputs: tf.Tensor) -> tf.Tensor:
        return preprocess_input(255 * inputs, mode='caffe')
    
    def torch_preprocess(self, inputs: tf.Tensor) -> tf.Tensor:
        return preprocess_input(255 * inputs, mode='torch')

    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        return [tf.identity(inputs)] + self.vgg(self.preprocess_function(inputs))
