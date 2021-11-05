from typing import List, Optional, Union

import numpy as np
import tensorflow as tf

from . import utils


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
    for lap in laplasian_pyramids[:-1][::-1]:
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
    mean_tensor = tf.reduce_mean(image, axis=axis, keepdims=True)
    mean_tensor = tf.cast(mean_tensor, dtype=tf.float32)
    return mean_tensor


def to_variable(data: Union[List[tf.Tensor], List[np.ndarray]]) -> List[tf.Variable]:
    """create trainable variables."""
    flats = tf.nest.flatten(data)
    variables = [tf.Variable(f) for f in flats]
    return variables


def init_value(shape=()) -> tf.Tensor:
    """used to initial value (=0)"""
    return tf.zeros(shape=shape, dtype=tf.float32)


def clip_and_normalize(
    image: tf.Tensor,
    base_image: Optional[tf.Tensor]=None,
    optimize_mode: str = 'vgg') -> np.ndarray:
    """clip and normalize. output range is [0, 255]"""
    if optimize_mode == 'uniform':
        _min, _max = -1., 1.
    elif optimize_mode == 'vgg':
        _min, _max = -1.7, 1.7
    elif optimize_mode == 'paper':
        _min, _max = -0.5, 0.5
    else:
        _min, _max = 0., 1.
    clipped = tf.clip_by_value(image, _min, _max)
    # renorm
    clipped = clipped - tf.reduce_min(clipped)
    clipped = clipped / tf.reduce_max(clipped)
    clipped = (clipped.numpy() * 255).astype(np.uint8)

    if base_image is not None:
        clipped = utils.resize_like(clipped, base_image)
    return clipped


def bilinear_resampling(
    content_feature: tf.Tensor,
    style_feature: tf.Tensor,
    indices: tf.Tensor,
    shapes: tf.Tensor) -> tf.Tensor:
    """Bilinear sampling."""
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
        wrap_a = tf.reshape((1.-dx) * (1.-dy), (1, -1, 1, 1))
        wrap_b = tf.reshape((1.-dx) * dy, (1, -1, 1, 1))
        wrap_c = tf.reshape(dx * (1.-dy), (1, -1, 1, 1))
        wrap_d = tf.reshape(dx * dy, (1, -1, 1, 1))

        # cast & clip
        xgrid = tf.clip_by_value(tf.cast(xgrid, tf.int32), 0, kh-1)
        ygrid = tf.clip_by_value(tf.cast(ygrid, tf.int32), 0, kw-1)
        xgrid_bias = tf.clip_by_value(xgrid+1, 0, kh-1)
        ygrid_bias = tf.clip_by_value(ygrid+1, 0, kw-1)

        indice_a = xgrid * kw + ygrid
        indice_b = xgrid * kw + ygrid_bias
        indice_c = xgrid_bias * kw + ygrid
        indice_d = xgrid_bias * kw + ygrid_bias

        # reshape
        content_feature = tf.reshape(content_feature, (1, kh*kw, 1, kc))
        style_feature = tf.reshape(style_feature, (1, kh*kw, 1, kc))

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


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.bool),
        tf.TensorSpec(shape=[2], dtype=tf.int32),
        tf.TensorSpec(shape=[2], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.int32)])
def create_indices(
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
        gx_start = tf.random.uniform((), 0, steps[0], dtype=tf.int32)
        gy_start = tf.random.uniform((), 0, steps[1], dtype=tf.int32)

        # slice
        gx_r = gx_r[gx_start::steps[0]]
        gy_r = gy_r[gy_start::steps[1]]

        # create meshgrid
        gx, gy = tf.meshgrid(gx_r, gy_r)

        # flatten
        gx = tf.reshape(gx, [-1])
        gy = tf.reshape(gy, [-1])

        # shuffle
        # if not shuffle output result is horrible
        gx = tf.random.shuffle(gx)
        gy = tf.random.shuffle(gy)

        # collect
        # shape -> [?,]
        indice_mask = tf.gather(
            tf.gather(content_region[...,0], gx, axis=0), gy, axis=1)[:,0]
        
        # create indices
        # shape > [?, 2]
        indices = tf.concat([gx[:,tf.newaxis], gy[:,tf.newaxis]], axis=1)

        # pop (indice==True)
        # output = [?, 2]
        indices = indices[indice_mask]

        # use 1024 locations
        # shape -> [1024, 2]
        indices = indices[:step_ind]

        # cast to float32 (for bilinear sampling)
        return tf.cast(indices, tf.float32)


def create_style_features(
    style_features: List[tf.Tensor],
    style_region: tf.Tensor,
    n_loop: int,
    max_samples: int) -> tf.Tensor:
    """
    create style features
        output shape: `[1, n_loop*max_samples, 1, total_dim]`
    """
    ret_style_feats = None

    # resize region shape
    style_region = utils.resize_like(style_region, style_features[0])
    # fix region
    bias = (style_region.numpy().max() < 0.1).astype(np.float32)
    mask = (style_region.numpy().flatten() + bias) > 0.5

    # create indices
    h, w = utils.get_h_w(style_features[0])
    xx, xy = np.meshgrid(np.arange(h), np.arange(w))
    xx = np.expand_dims(xx.flatten(), 1)
    xy = np.expand_dims(xy.flatten(), 1)
    _indices = np.concatenate([xx, xy], 1)
    _indices = _indices[mask]
    indices = _indices.copy()

    # num resample indices, 
    samples = min(max_samples, indices.shape[0])

    # numpy operation
    for i in range(n_loop):
        np.random.shuffle(indices)
        xx = indices[:samples, 0]
        xy = indices[:samples, 1]

        feats = None
        # target shape is [1, h, w, c]
        for j, target in enumerate(style_features):
            if j > 0 and style_features[j].shape[1] < style_features[j-1].shape[1]:
                # if downscaled, reduce indices
                # e.g. [1, 64, 64, 128] -> [1, 32, 32, 256]
                xx = xx /2.0
                xy = xy /2.0
            # get h & w
            h, w = utils.get_h_w(target)
            # xx an xy is np.float32. clip and convert to np.int32.
            _xx = np.clip(xx, 0, h-1).astype(np.int32)
            _xy = np.clip(xy, 1, w-1).astype(np.int32)
            # index slicing
            # shape -> [1, 1000, 1, channel]
            collected = target.numpy()[:,_xx,_xy][:,:,np.newaxis]
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
    
    # loop end: feats -> [1, 5000,]
    return utils.to_tensor(ret_style_feats, tf.float32)
