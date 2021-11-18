import os
import shutil
import time
from typing import Any, List, Optional, Tuple, Union
from logging import getLogger
logger = getLogger('strotss')
logger.setLevel(10)

import cv2
import numpy as np
import tensorflow as tf


METHOD = ['tf', 'tensorflow', 'cv2']
ACCEPTED_NAME = [
    ['b', 'batch'],
    ['h', 'height'],
    ['w', 'width'],
    ['c', 'channel', 'channels', 'd', 'dim', 'dims']]


def ljust_print(string: str):
    column = shutil.get_terminal_size().columns
    r_string = string.ljust(column)
    return r_string


def get_shape_ndims(
    tensor_or_array: Union[tf.Tensor, tf.Variable, np.ndarray],
    return_shape: bool = False,
    return_ndim: bool = False) -> Union[Tuple[int, int], tf.TensorShape, int]:
    if isinstance(tensor_or_array, (tf.Tensor, tf.Variable)):
        shape = tensor_or_array.shape
        ndim = shape.ndims
    elif isinstance(tensor_or_array, np.ndarray):
        shape = tensor_or_array.shape
        ndim = tensor_or_array.ndim
    else:
        raise ValueError('Unknown type: {}'.format(type(tensor_or_array)))
    if return_shape and not return_ndim:
        return shape
    elif not return_shape and return_ndim:
        return ndim
    return shape, ndim


def get_shape_by_name(
    tensor_or_array: Union[tf.Tensor, tf.Variable, np.ndarray],
    *targets) -> Union[Tuple[int], int]:
    """Get tensor shapes according to `*targets`"""
    shape, ndim = get_shape_ndims(tensor_or_array)

    accepted_name = tf.nest.flatten(ACCEPTED_NAME)
    is_channels_last = tf.keras.backend.image_data_format() == 'channels_last'
    collects = []
    for _name in tf.nest.flatten(targets):
        if not isinstance(_name, str):
            raise ValueError('Argument `targets` must be str, not {}.'.format(type(_name)))
        if _name.lower() not in accepted_name:
            raise ValueError('{} is not supported.'.format(_name))

        _name = _name.lower()
        if ndim < 4 and _name in ACCEPTED_NAME[0]:
            raise ValueError('Input tensor or array must be 4D or greater.')
        elif ndim < 3 and _name in ACCEPTED_NAME[3]:
            raise ValueError('Input tensor or array must be 3D or greater.')

        if _name in ACCEPTED_NAME[0]:
            key = 0
        elif _name in ACCEPTED_NAME[1]:
            key = 1 if is_channels_last else 2
        elif _name in ACCEPTED_NAME[2]:
            key = 2 if is_channels_last else 3
        elif _name in ACCEPTED_NAME[3]:
            key = 3 if is_channels_last else 1
        collects.append(shape[key - int(ndim == 3)])
    
    if len(collects) == 1:
        return collects[0]
    
    return tuple(collects)


def to_tensor(
    data: Any,
    dtype: Optional[Union[tf.DType, np.dtype, str]]=None,
    as_constant: bool = False,
    name: Optional[str]=None) -> tf.Tensor:
    """Convert object to Tensor"""
    if as_constant:
        return tf.constant(data, dtype=dtype, name=name)
    return tf.convert_to_tensor(data, dtype=dtype)


def get_h_w(
    image: Union[tf.Tensor, tf.Variable, np.ndarray]) -> Tuple[int, int]:
    """Get height, width of tensor."""
    return get_shape_by_name(image, 'height', 'width')


def resize_image(
    image: Union[tf.Tensor, tf.Variable, np.ndarray],
    size: Union[Tuple[int, int], int, str],
    method: str='bilinear',
    **kwargs) -> tf.Tensor:
    """Resize image."""
    if isinstance(size, str):
        if size not in ['half', 'downsample']:
            raise ValueError('Unknown size operation: {}'.format(size))
        _size = get_h_w(image)
        size = (max(1, _size[0]//2), max(1, _size[1]//2))
    elif isinstance(size, int):
        _size = get_h_w(image)
        factor = max(_size) / size
        size = (max(1, int(_size[0]/factor)), max(1, int(_size[1]/factor)))
    return tf.image.resize(image, size=size, method=method, **kwargs)


def resize_like(
    image: Union[tf.Tensor, tf.Variable, np.ndarray],
    base_image: Union[tf.Tensor, tf.Variable, np.ndarray],
    method: str='bilinear',
    **kwargs) -> tf.Tensor:
    """Resize `image` according to `base_image` shape."""
    return resize_image(image, get_h_w(base_image), method=method, **kwargs)


def cv2_imread_fixed(path: str) -> np.ndarray:
    """Support for Japanese folders"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = np.fromfile(path, np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)[...,:3] # BGR -> RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_image(
    image: Union[tf.Tensor, tf.Variable, np.ndarray],
    dst: str):
    """Support for Japanese folders"""
    image = np.array(image, np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, ext = os.path.splitext(dst)
    result, n = cv2.imencode(ext, image)

    if result:
        try:
            with open(dst, 'w+b') as f:
                n.tofile(f)
        except OSError:
            raise OSError(
                'Failed to write image ({}). If it is open in another application, please close it.'.format(dst))
    else:
        raise RuntimeError('Failed to save image.')


def set_optimizer_lr(
    optimizer, 
    factor: Optional[float] = None,
    learning_rate: Optional[float] = None):
    current_lr = tf.keras.backend.get_value(optimizer.lr)

    if learning_rate is None:
        if factor is None:
            return
        new_lr = current_lr * factor
    else:
        new_lr = learning_rate
    tf.keras.backend.set_value(optimizer.lr, new_lr)


def read_image(
    path: str,
    load_img_method: str = 'cv2',
    optimize_mode: str = 'uniform'
    ) -> tf.Tensor:
    """Read image, and preprocess according to `optimize_mode`."""
    if load_img_method == 'cv2' or load_img_method not in METHOD:
        if load_img_method not in METHOD:
            logger.warning('Unsupported load image method. using opencv2.')
        image = cv2_imread_fixed(path)
        image = to_tensor(image)
    elif load_img_method in ['tf', 'tensorflow']:
        ext: str = os.path.splitext(path)[1]
        if ext.lower() in ['.jpg', 'jpeg']:
            load_function = tf.image.decode_jpeg
            kwg = {'dct_method': "INTEGER_ACCURATE"}
        else:
            load_function = tf.image.decode_image
            kwg = {'expand_animations': False}
        raw = tf.io.read_file(path)
        image = load_function(raw, channels=3, **kwg)

    # preprocess
    image = tf.cast(image, tf.float32)[tf.newaxis]
    if optimize_mode == 'torch_uniform':
        # range -> [-1, 1]
        image = image / 127.5 - 1.
    else: # range -> [0, 1]
        image = image / 255.
    return image


def add_signature(
    img: Union[tf.Tensor, tf.Variable, np.ndarray],
    text: str,
    fontFace: int = cv2.FONT_HERSHEY_SIMPLEX,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    lineType: int = cv2.LINE_AA,
    bottomLeftOrigin: bool = False) -> np.ndarray:
    """
    Put text on image.
    Returns:
        Image with text on it.
    """
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fontScale = max(min(max(img.shape)/1024, 1.0), 0.1)
    (text_w, text_h), _ = cv2.getTextSize(
                text=text,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=fontScale,
                thickness=thickness)
    crop_area = img[img.shape[0]-text_h:img.shape[0], img.shape[1]-text_w:img.shape[1]]
    means = np.mean(crop_area, axis=(0, 1), keepdims=True).mean()
    cv2.putText(
        img,
        text,
        org=(img.shape[1] - text_w, img.shape[0] - text_h),
        fontFace=fontFace,
        fontScale=fontScale,
        color=[255*int(means < 128)]*3,
        thickness=thickness,
        lineType=lineType,
        bottomLeftOrigin=bottomLeftOrigin)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_regions(
    content_r_path: str, style_r_path: str,
    threth_denominator: int = 255, threth_min_counts: int = -1,
    noregion: bool=False) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """Read region images."""

    assert 0 < threth_denominator <= 256, 'Argument `threth_denominator` must be in 1 to 255.'
    assert -1 <= threth_min_counts, 'Argument `threth_min_counts` must be greater than -1.'

    cr_image = cv2_imread_fixed(content_r_path)
    sr_image = cv2_imread_fixed(style_r_path)

    if noregion:
        return (
            [np.ones_like(cr_image[...,0], np.float32)[..., np.newaxis]],
            [np.ones_like(sr_image[...,0], np.float32)[..., np.newaxis]])

    if threth_min_counts == -1:
        threth_min_counts = min(min(sr_image.shape[:2])//16, 5)

    cr_image = cr_image // threth_denominator * threth_denominator
    sr_image = sr_image // threth_denominator * threth_denominator

    uniques, counts = np.unique(sr_image.reshape(-1, 3), axis=0, return_counts=True)
    uniques = uniques[counts > threth_min_counts]

    content_regions = []
    style_regions = []

    for unique in uniques:
        c_cond: np.ndarray = (cr_image[...,0]==unique[0]) & (cr_image[...,1]==unique[1]) & (cr_image[...,2]==unique[2])
        s_cond: np.ndarray = (sr_image[...,0]==unique[0]) & (sr_image[...,1]==unique[1]) & (sr_image[...,2]==unique[2])
        if np.any(c_cond) and np.any(s_cond):
            content_regions.append(
                to_tensor(c_cond.astype(np.float32)[..., np.newaxis]))
            style_regions.append(
                to_tensor(s_cond.astype(np.float32)[..., np.newaxis]))

    if len(content_regions) == 0:
        raise RuntimeError('Could not find regions.')

    print(f'Regions: {len(content_regions)}')
    return (content_regions, style_regions)


class Timer:
    def __init__(self):
        self._start = 0.
        self._stop = 0.
        self._is_stop = False
        self.total = 0.

    def start(self):
        self._start = time.time()
    
    def stop(self, save_time=False, return_time=True):
        self._stop = time.time()
        self._is_stop = True
        ret = self.time() if return_time else None
        if save_time:
            self.save()
        return ret

    def time(self, ndigits: int=3):
        result = round(self._stop - self._start, ndigits)
        return result

    def save(self):
        if not self._is_stop:
            self.stop()
        self.total += self.time(ndigits=5)
        self._start = 0.
        self._stop = 0.
