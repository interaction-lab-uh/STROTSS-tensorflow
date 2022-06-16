from typing import Optional

import tensorflow as tf


_URL = 'https://ppza53893.github.io/box/weights/{filename}'
_STROTSS_DEFAULTS = ['block1_conv1',
                     'block1_conv2',
                     'block2_conv1',
                     'block2_conv2',
                     'block3_conv1',
                     'block3_conv2',
                     'block3_conv3',
                     'block4_conv3',
                     'block5_conv3']

class VGG(tf.Module):
    """feature extractor."""

    def __init__(self, layers: Optional[list] = None,
                 vgg_type: str='16',
                 use_keras_weight: bool = False,
                 name: Optional[str] = None):
        vgg_type = str(vgg_type)
        assert vgg_type in ['16', '19']
        layers = layers or _STROTSS_DEFAULTS

        super().__init__(name = name)

        if not use_keras_weight:
            filename = f'vgg{vgg_type}_norm.h5'
            url = _URL.format(filename=filename)
            weight_path = tf.keras.utils.get_file(filename, url)
            self.mean = tf.reshape(tf.convert_to_tensor([0.485, 0.456, 0.406], dtype=tf.float32), (1, 1, 1, -1))
            self.std = tf.reshape(tf.convert_to_tensor([0.229, 0.224, 0.225], dtype=tf.float32), (1, 1, 1, -1))
        else:
            weight_path = 'imagenet'
            self.preprocess = lambda x: getattr(tf.keras.applications, f'vgg{vgg_type}').preprocess_input(x*255)

        if vgg_type == '16':
            vgg_base = tf.keras.applications.VGG16
        else:
            vgg_base = tf.keras.applications.VGG19
        vgg = vgg_base(include_top=False, weights=weight_path)
        vgg.trainable = False
        outputs = [vgg.get_layer(layer).output for layer in layers]
    
        self.net = tf.keras.Model(vgg.input, outputs)

    def preprocess(self, inputs: tf.Tensor) -> tf.Tensor:
        return (inputs - self.mean) / self.std

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.net(self.preprocess(inputs))
