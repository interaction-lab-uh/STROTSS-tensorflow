from typing import List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16


class VGG16_Patch(Model):
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
        if optimize_mode in ['vgg', 'uniform', 'paper']:
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
        self.dims = 3 + sum([o.shape[-1] for o in outputs]) # 2179

        if optimize_mode == 'uniform':
            self.function = self.uniform_prerpocess
        elif optimize_mode == 'caffe':
            self.function = self.caffe_preprocess
        elif optimize_mode == 'vgg':
            self.function = self.torch_preprocess
        else:
            self.function = lambda x: x

    @property
    def num_outputs(self) -> int:
        return len(self.layer_name) + 1

    def uniform_prerpocess(self, inputs: tf.Tensor):
        x = (inputs + 1.) / 2.
        return preprocess_input(255 * x, mode='torch')

    def caffe_preprocess(self, inputs: tf.Tensor):
        return preprocess_input(255 * inputs, mode='caffe')
    
    def torch_preprocess(self, inputs: tf.Tensor):
        return preprocess_input(255 * inputs, mode='torch')
    
    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        return [tf.identity(inputs)] + self.vgg(self.function(inputs))
