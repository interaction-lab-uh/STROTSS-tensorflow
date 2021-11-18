from typing import Callable, List, Optional, Union

import numpy as np
import tensorflow as tf

from . import utils

np_rng: np.random.Generator = None
tf_rng: tf.random.Generator = None


def set_rng(
    numpy_rng: Optional[np.random.Generator],
    tensorflow_rng: Optional[tf.random.Generator]):
    """set rng"""
    global np_rng, tf_rng
    if numpy_rng is not None and hasattr(numpy_rng, 'shuffle'):
        np_rng = numpy_rng
    else:
        np_rng = np.random.default_rng()
    
    if tensorflow_rng is not None and hasattr(tensorflow_rng, 'from_seed'):
        tf_rng = tensorflow_rng
    else:
        tf_rng = tf.random.Generator.from_seed(np_rng.integers(0, 2**32 - 1))


def get_np_rng() -> np.random.Generator:
    return np_rng


def create_laplasian(
    image: tf.Tensor) -> tf.Tensor:
    """
    create laplasian

    `output = input-upsample(downsample(input,size=x1/2,method=bilinear),size=x2,method=bilinear)`
    """
    down_up = utils.resize_like(utils.resize_image(image, 'downsample'), image)
    return image - down_up


def fold_lap(laplasian_pyramids: List[tf.Tensor]) -> tf.Tensor:
    """fold laplasian pyramid."""
    ret = laplasian_pyramids[-1]
    lap_iter = reversed(laplasian_pyramids[:-1])
    if tf.executing_eagerly():
        for lap in lap_iter:
            ret = lap + tf.image.resize(ret, (lap.shape[1], lap.shape[2]), method='bilinear')
    else:
        for lap in lap_iter:
            ret = lap + utils.resize_like(ret, lap)
    return ret


def create_laplasian_pyramids(
    image: tf.Tensor,
    pyramid_level: int = 5) -> List[tf.Tensor]:
    """create laplasian pyramid."""
    ret = []
    current = tf.identity(image)
    for _ in range(pyramid_level):
        ret.append(create_laplasian(current))
        current = utils.resize_image(current, 'downsample')
    ret.append(current)
    return ret


def to_mean_color(
    image: tf.Tensor, return_dtype: Optional[tf.DType] = tf.float32) -> tf.Tensor:
    """get mean color"""
    return_dtype = return_dtype or tf.float32

    axis = [1, 2] if utils.get_shape_ndims(image, return_ndim=True) == 4 else [0, 1]
    mean_tensor = image.mean(axis = axis, keepdims=True).astype(return_dtype)
    return mean_tensor


def to_variable(data: Union[List[tf.Tensor], List[np.ndarray]]) -> List[tf.Variable]:
    """create trainable variables."""
    flats = tf.nest.flatten(data)
    variables = [tf.Variable(f) for f in flats]
    return variables


def init_zeros(shape=()) -> tf.Tensor:
    """used to initial value (=0)"""
    return tf.zeros(shape=shape, dtype=tf.float32)


def clip_and_denormalize(
    image: tf.Tensor,
    base_image: Optional[tf.Tensor]=None,
    optimize_mode: str = 'vgg') -> np.ndarray:
    """clip and denormalize. output range is [0, 255]"""
    if optimize_mode == 'torch_uniform':
        _min, _max = -1., 1.
    else:
        _min, _max = 0., 1.
    clipped = image.clip(_min, _max)
    # renorm
    clipped = clipped - clipped.min()
    clipped = clipped / clipped.max()
    clipped = (clipped.numpy() * 255).astype(np.uint8)

    if base_image is not None:
        clipped = utils.resize_like(clipped, base_image)
    return clipped


def nowrap_bilinear_resampling(
    content_feature: tf.Tensor,
    style_feature: tf.Tensor,
    indices: tf.Tensor,
    shapes: tf.Tensor) -> tf.Tensor:
    """Bilinear resampling."""
    with tf.name_scope('bilinear_resampling'):
        kh = shapes[0] # feature height
        kw = shapes[1] # feature width
        kc = shapes[2] # feature channel

        xx = indices[..., 0] # 1024
        xy = indices[..., 1] # 1024

        xgrid = tf.math.floor(xx)
        dx = xx - xgrid # dx
        ygrid = tf.math.floor(xy)
        dy = xy - ygrid # dy

        # reahape for compute interpolation
        wrap_a = ((1.-dx) * (1.-dy)).reshape((1, -1, 1, 1))
        wrap_b = ((1.-dx) * dy).reshape((1, -1, 1, 1))
        wrap_c = (dx * (1.-dy)).reshape((1, -1, 1, 1))
        wrap_d = (dx * dy).reshape((1, -1, 1, 1))

        # cast & clip
        xgrid = xgrid.astype(tf.int32).clip(0, kh-1)
        ygrid = ygrid.astype(tf.int32).clip(0, kw-1)
        xgrid_bias = (xgrid+1).clip(0, kh-1)
        ygrid_bias = (ygrid+1).clip(0, kw-1)

        indice_a = xgrid * kw + ygrid
        indice_b = xgrid * kw + ygrid_bias
        indice_c = xgrid_bias * kw + ygrid
        indice_d = xgrid_bias * kw + ygrid_bias

        # reshape
        content_feature = content_feature.reshape((1, kh*kw, 1, kc))
        style_feature = style_feature.reshape((1, kh*kw, 1, kc))

        # compute interpolation
        content_feature = (
            tf.gather(content_feature, indice_a, axis=1)*wrap_a +\
                tf.gather(content_feature, indice_b, axis=1)*wrap_b +\
                    tf.gather(content_feature, indice_c, axis=1)*wrap_c +\
                        tf.gather(content_feature, indice_d, axis=1)*wrap_d)
        style_feature = (
            tf.gather(style_feature, indice_a, axis=1)*wrap_a +\
                tf.gather(style_feature, indice_b, axis=1)*wrap_b +\
                    tf.gather(style_feature, indice_c, axis=1)*wrap_c +\
                        tf.gather(style_feature, indice_d, axis=1)*wrap_d)

        # shape -> [1, 1024, 1, kc]
        return content_feature, style_feature


def create_grids(sizes: np.array, steps: np.array):
    """create grid for bilinear sampling."""
    gx_r = np.arange(sizes[0], dtype=np.int32)
    gy_r = np.arange(sizes[1], dtype=np.int32)

    # init offsets
    gx_start = np_rng.integers(0, steps[0], dtype=tf.int32)
    gy_start = np_rng.integers(0, steps[1], dtype=tf.int32)

    # slice
    gx_r = gx_r[gx_start::steps[0]]
    gy_r = gy_r[gy_start::steps[1]]

    # create meshgrid
    gx, gy = np.meshgrid(gx_r, gy_r)

    # flatten
    return gx.flatten(), gy.flatten()


def create_indices_w_np(
    gx: np.ndarray,
    gy: np.ndarray,
    mask: np.ndarray,
    step_ind: int) -> np.ndarray:
    gx = gx[:,np.newaxis]
    gy = gy[:,np.newaxis]

    indice_mask = mask[gx, gy, 0]
    indices = np.concatenate([gx[:,np.newaxis], gy[:,np.newaxis]], axis=-1)[indice_mask]
    indices = indices[:step_ind]

    return utils.to_tensor(indices, tf.float32)


def nowrap_create_indices(
    content_region: tf.Tensor,
    sizes: tf.Tensor,
    steps: tf.Tensor,
    step_ind: tf.Tensor):
    """create indices. used train_step."""
    with tf.name_scope('create_indices'):
        # create meshgrid
        # base
        gx_r = tf.range(sizes[0], dtype=tf.int32)
        gy_r = tf.range(sizes[1], dtype=tf.int32)

        # init offsets
        gx_start = tf_rng.uniform((), 0, steps[0], dtype=tf.int32)
        gy_start = tf_rng.uniform((), 0, steps[1], dtype=tf.int32)

        # slice
        gx_r = gx_r[gx_start::steps[0]]
        gy_r = gy_r[gy_start::steps[1]]

        # create meshgrid
        gx, gy = tf.meshgrid(gx_r, gy_r)

        # ravel & shuffle
        # if not shuffle output result is horrible
        gx = tf.random.shuffle(gx.ravel())
        gy = tf.random.shuffle(gy.ravel())
        
        # create indices
        # shape > [?, 2]
        indices = tf.concat([gx[:,tf.newaxis], gy[:,tf.newaxis]], axis=1)

        # pop (indice==True)
        # output = [?, 2]
        indice_mask = tf.gather_nd(content_region[..., 0], indices)
        indices = indices[indice_mask]

        # use 1024 locations
        # shape -> [1024, 2]
        indices = indices[:step_ind]

        # cast to float32 (for bilinear sampling)
        return tf.cast(indices, tf.float32)


create_indices: Callable[[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = tf.function(
    nowrap_create_indices,
    input_signature=[
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.bool),
        tf.TensorSpec(shape=[4], dtype=tf.int32),
        tf.TensorSpec(shape=[2], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.int32)])


def create_style_features(
    style_features: List[tf.Tensor],
    style_region: tf.Tensor,
    n_loop: int,
    max_samples: int) -> tf.Tensor:
    """
    create style features
        output shape: `[1, n_loop*max_samples, 1, total_dim]`
    """
    # resize region shape
    style_region = utils.resize_like(style_region, style_features[0]).numpy()
    # fix region
    bias = (style_region.max() < 0.1).astype(np.float32)
    mask = (style_region.flatten() + bias) > 0.5

    # create indices
    h, w = utils.get_h_w(style_features[0])
    xx, xy = np.meshgrid(np.arange(h), np.arange(w))
    xx = np.expand_dims(xx.flatten(), 1)
    xy = np.expand_dims(xy.flatten(), 1)
    indices = np.concatenate([xx, xy], 1)[mask]

    # num resample indices, 
    samples = min(max_samples, indices.shape[0])

    # numpy operation
    for i in range(n_loop):
        indices_copy = indices.copy()
        np_rng.shuffle(indices_copy)
        xx = indices_copy[:samples, 0]
        xy = indices_copy[:samples, 1]

        feats = None
        # target shape is [1, h, w, c]
        for j, target in enumerate(style_features):
            if j > 0 and style_features[j].shape[1] < style_features[j-1].shape[1]:
                # if downscaled, reduce indices
                # e.g. [1, 64, 64, 128] -> [1, 32, 32, 256]
                xx = xx / 2.0
                xy = xy / 2.0
            # get h & w
            h, w = utils.get_h_w(target)
            # xx an xy is np.float32. clip and convert to np.int32.
            xx = np.clip(xx, 0, h-1).astype(np.int32)
            xy = np.clip(xy, 1, w-1).astype(np.int32)
            # index slicing
            # shape -> [1, 1000, 1, channel]
            collected = target.numpy()[:,xx,xy][:,:,np.newaxis]
            concat_arrays = [collected]
            # add
            if j > 0:
                concat_arrays.insert(0, feats)
            feats = np.concatenate(concat_arrays, axis=3)
        
        # loop end: feats -> [1, 1000, 1, 2179]
        concat_arrays = [feats]
        # add
        if i > 0:
            concat_arrays.insert(0, ret_style_feats)
        ret_style_feats = np.concatenate(concat_arrays, axis=1)
    
    # loop end: feats -> [1, 5000, 1, 2179]
    return utils.to_tensor(ret_style_feats, tf.float32)
