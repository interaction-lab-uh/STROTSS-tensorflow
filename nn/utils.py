import logging
import sys
import time
from typing import Optional

import tensorflow as tf


logger = logging.getLogger(__name__)


def make_logger(name: str):
    global logger
    logger = logging.getLogger(name)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s', "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)


def _validate_and_get_shape(base: tf.Tensor):
    if base.shape.rank == 3:
        h, w, _ = base.shape.as_list()
    elif base.shape.rank == 4:
        _, h, w, c = base.shape.as_list()
    else:
        raise ValueError(f"Invalid rank: {base.shape.rank}")
    return h, w


def resize(image: tf.Tensor, max_size: Optional[int]) -> tf.Tensor:
    if max_size is None:
        return image
    h, w = _validate_and_get_shape(image)
    factor = max(h/max_size, w/max_size)
    return tf.image.resize(image, (int(h/factor), int(w/factor)))


def resize_like(image: tf.Tensor, base: tf.Tensor) -> tf.Tensor:
    return tf.image.resize(image, _validate_and_get_shape(base))


def load_image(path: str,
               max_size: Optional[int] = 512,
               dtype: tf.dtypes.DType = tf.float32,
               batch_expand: bool = True) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_ACCURATE")
    img = tf.image.convert_image_dtype(img, dtype)
    img = resize(img, max_size)

    if batch_expand:
        return img[tf.newaxis]
    return img


def write_image(image: tf.Tensor, path: str):
    encoded = tf.image.encode_jpeg(image, format="rgb", quality=100)
    tf.io.write_file(path, encoded)
    logger.info(f"Wrote image to {path}")


def set_gpu(index: int = 0):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if index >= len(gpus):
            raise ValueError(f"Invalid GPU ID: {index}")
        try:
            tf.config.set_visible_devices(gpus[index], 'GPU')
        except (RuntimeError, ValueError) as e:
            logger.error(e, exc_info=1)
        else:
            logger.debug(f"Set GPU to {index}")
    else:
        logger.info("GPU not found. Using CPU.")


class Timer:
    def __init__(self):
        self._start = 0.
        self._stop = 0.
        self._elapsed = 0.

    def start(self):
        self._start = time.time()

    def stop(self):
        self._stop = time.time()
        self._elapsed = round(self._stop - self._start, 3)
        self._start = 0.
        self._stop = 0.

    @property
    def elapsed_time(self):
        return self._elapsed

