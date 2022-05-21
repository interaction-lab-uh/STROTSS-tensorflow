import logging

import requests
import tensorflow as tf

logger = logging.getLogger('STROTSS')


def _request_vgg_model(filename: str):
    try:
        logger.info(f'Downloading {filename}...')
        result = requests.get(f'https://ppza53893.github.io/box/weights/{filename}')
        with open(f'./{filename}', 'wb') as f:
            f.write(result.content)
        del result
    except:
        raise Exception(
            f'Unable to download {filename}. '
            f'Please download directly from "https://ppza53893.github.io/box/weights/{filename}".') 


class VGG(tf.Module):
    def __init__(self, layers: list,
                 vgg_type: str='19',
                 use_keras_weight: bool = False,
                 **kwargs):
        mean = kwargs.pop('mean', [0.485, 0.456, 0.406])
        std = kwargs.pop('std', [0.229, 0.224, 0.225])
        super().__init__(**kwargs)
        vgg_type = str(vgg_type)
        assert vgg_type in ['16', '19']

        if not use_keras_weight:
            filename = f'vgg{vgg_type}_norm.h5'
            if not tf.io.gfile.exists(f'./{filename}'):
                _request_vgg_model(filename)
            weight_path = f'./{filename}'
            self.mean = tf.reshape(tf.convert_to_tensor(mean, dtype=tf.float32), (1, 1, 1, -1))
            self.std = tf.reshape(tf.convert_to_tensor(std, dtype=tf.float32), (1, 1, 1, -1))
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
    def __call__(self, inputs):
        return self.net(self.preprocess(inputs))
