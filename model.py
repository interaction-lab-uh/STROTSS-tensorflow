from typing import List

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16


class VGG16_Patch(Model):
    use_layer_name: List[str] = [
        "block1_conv1",
        "block1_conv2",
        "block2_conv1",
        "block2_conv2",
        "block3_conv1",
        "block3_conv2",
        "block3_conv3",
        "block4_conv3",
        "block5_conv3"]

    def __init__(self, optimize_mode: str = 'vgg'):
        super().__init__()
        self.vgg = self.build_vgg(optimize_mode)
        if optimize_mode == 'uniform':
            self.function = self.uniform_prerpocess
        elif optimize_mode == 'caffe':
            self.function = self.caffe_preprocess
        elif optimize_mode == 'vgg':
            self.function = self.torch_preprocess
        else:
            self.function = lambda x: x
    
    def build_vgg(self, optimize_mode: str):
        if optimize_mode in ['vgg', 'uniform', 'paper']:
            weights = 'vgg_norm.h5'
        else:
            weights = 'imagenet'
        print('VGG16 weight:', weights)
        vgg_base = VGG16(include_top=False, weights=weights)
        vgg_base.trainable = False
        outputs = [vgg_base.get_layer(name).output for name in self.use_layer_name]
        self.feature_dim = sum(out.shape[-1] for out in outputs) + 3
        model = Model(inputs=vgg_base.inputs, outputs=outputs)
        return model

    @property
    def dims(self) -> int:
        return self.feature_dim

    @property
    def num_outputs(self) -> int:
        return len(self.use_layer_name) + 1

    def uniform_prerpocess(self, inputs: tf.Tensor):
        x = (inputs + 1.) / 2.
        return self.torch_preprocess(x)

    def caffe_preprocess(self, inputs: tf.Tensor):
        return preprocess_input(255 * inputs, mode='caffe')
    
    def torch_preprocess(self, inputs: tf.Tensor):
        return preprocess_input(255 * inputs, mode='torch')
    
    def call(self, inputs: tf.Tensor) -> List[tf.Tensor]:
        return [tf.identity(inputs)] + self.vgg(self.function(inputs))
