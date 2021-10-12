from typing import List, Optional, Union

import numpy as np
import tensorflow as tf

import utils


def gauss(
    image: Union[tf.Tensor, tf.Variable, np.ndarray],
    scales: int):
    outs = [utils.resize_image(image, scale) for scale in scales]
    return outs


def make_laplasian(
    image: tf.Tensor) -> tf.Tensor:
    down_up = utils.resize_like(utils.resize_image(image, 'downsample'), image)
    return image - down_up


def fold_lap(laplasian_pyramids: List[tf.Tensor]) -> tf.Tensor:
    ret = laplasian_pyramids[-1]
    for lap in laplasian_pyramids[:-1][::-1]:
        ret = lap + utils.resize_like(ret, lap)
    return ret


def create_laplasian(
    image: tf.Tensor,
    pyramid_level: int = 5) -> List[tf.Tensor]:
    ret = []
    current = tf.identity(image)
    for _ in range(pyramid_level):
        ret.append(make_laplasian(current))
        current = utils.resize_image(current, 'downsample')
    ret.append(current)
    return ret


def to_mean_color(
    image: tf.Tensor, return_dtype: Optional[tf.DType] = tf.float32) -> tf.Tensor:
    return_dtype = return_dtype or tf.float32

    axis = [1, 2] if utils.get_shape_ndims(image, return_ndim=True) == 4 else [0, 1]
    mean_tensor = tf.reduce_mean(image, axis=axis, keepdims=True)
    mean_tensor = tf.cast(mean_tensor, dtype=tf.float32)
    return mean_tensor


def to_variable(data: Union[List[tf.Tensor], List[np.ndarray]]) -> List[tf.Variable]:
    """Create trainable variables."""
    flats = tf.nest.flatten(data)
    variables = [tf.Variable(f) for f in flats]
    return variables


def init_value(shape=()) -> tf.Tensor:
    return tf.zeros(shape=shape, dtype=tf.float32)


def clip_and_normalize(
    image: tf.Tensor,
    base_image: Optional[tf.Tensor]=None,
    optimize_mode: str = 'vgg') -> tf.Tensor:
    if optimize_mode == 'uniform':
        _min, _max = -1., 1.
        clipped = tf.clip_by_value(image, _min, _max)
        clipped = (clipped - tf.reduce_min(clipped))
        clipped = clipped/tf.reduce_max(clipped)
        return (clipped.numpy() * 255).astype(np.uint8)
    elif optimize_mode == 'vgg':
        _min, _max = -1.7, 1.7
    elif optimize_mode == 'paper':
        _min, _max = -0.5, 0.5
    else:
        _min, _max = 0., 1.
        clipped = image
        clipped = (clipped - tf.reduce_min(clipped))/(tf.reduce_max(clipped) - tf.reduce_min(clipped))
        return (clipped.numpy() * 255).astype(np.uint8)
    clipped = tf.clip_by_value(image, _min, _max)
    if base_image is not None:
        clipped = utils.resize_like(clipped, base_image)
    # renorm
    clipped = (clipped - tf.reduce_min(clipped))/(tf.reduce_max(clipped) - tf.reduce_min(clipped))
    return (clipped.numpy() * 255).astype(np.uint8)


def bilinear_resampling(
    content_feature: tf.Tensor,
    style_feature: tf.Tensor,
    indices: tf.Tensor,
    shapes: tf.Tensor) -> tf.Tensor:
    """Bilinear Interpolation, but compute as 1d."""
    # trace test
    with tf.name_scope('bilinear_resampling'):
        # batch is 1.
        #_, h, w, c = style_feature.shape
        #_, hc, wc, cc = content_feature.shape

        kh = shapes[0]
        kw = shapes[1]
        kc = shapes[2]

        xx = indices[..., 0] # (1024, )
        xy = indices[..., 1] # (1024, )

        xxm = tf.math.floor(xx)
        xxr = xx - xxm
        xym = tf.math.floor(xy)
        xyr = xy - xym

        w00 = tf.reshape((1.-xxr) * (1.-xyr), (1, -1, 1, 1))
        w01 = tf.reshape((1.-xxr) * xyr, (1, -1, 1, 1))
        w10 = tf.reshape(xxr * (1.-xyr), (1, -1, 1, 1))
        w11 = tf.reshape(xxr * xyr, (1, -1, 1, 1))

        xxm = tf.clip_by_value(tf.cast(xxm, tf.int32), 0, kh-1)
        xym = tf.clip_by_value(tf.cast(xym, tf.int32), 0, kw-1)
        xxm_bias = tf.clip_by_value(xxm+1, 0, kh-1)
        xym_bias = tf.clip_by_value(xym+1, 0, kw-1)

        s00 = xxm * kw + xym
        s01 = xxm * kw + xym_bias
        s10 = xxm_bias * kw + xym
        s11 = xxm_bias * kw + xym_bias

        content_feature = tf.reshape(content_feature, (1, kh*kw, 1, kc))
        style_feature = tf.reshape(style_feature, (1, kh*kw, 1, kc))

        # c
        content_feature = (
            tf.gather(content_feature, s00, axis=1)*w00 +\
                tf.gather(content_feature, s01, axis=1)*w01 +\
                    tf.gather(content_feature, s10, axis=1)*w10 +\
                        tf.gather(content_feature, s11, axis=1)*w11)
        # s
        style_feature = (
            tf.gather(style_feature, s00, axis=1)*w00 +\
                tf.gather(style_feature, s01, axis=1)*w01 +\
                    tf.gather(style_feature, s10, axis=1)*w10 +\
                        tf.gather(style_feature, s11, axis=1)*w11)

        return content_feature, style_feature


def gather_xyz(
    tensor: tf.Tensor,
    x_indices: tf.Tensor,
    y_indices: Optional[tf.Tensor] = None,
    z_indices: Optional[tf.Tensor] = None,
    collect_axis: int = 0,
    keepdims: bool = False):
    gathered = tf.gather(tensor, x_indices, axis=collect_axis)
    should_keep = []
    if isinstance(y_indices, tf.Tensor):
        gathered = tf.gather(gathered, y_indices, axis=collect_axis+1)
        should_keep.append(collect_axis+1)
    if isinstance(z_indices, tf.Tensor):
        gathered = tf.gather(gathered, z_indices, axis=collect_axis+2)
        should_keep.append(collect_axis+2)
    if not keepdims and len(should_keep) == 0:
        return gathered
    base = slice(None, None, None)
    s, t = [], []
    for i in range(utils.get_shape_ndims(tensor, return_ndim=True)):
        if i not in should_keep:
            s.append(base); t.append(base)
        else:
            s.append(0); t.append(None)
    gathered = gathered[tuple(s)]
    if keepdims:
        gathered = gathered[tuple(t)]
    return gathered


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.bool),
        tf.TensorSpec(shape=[2], dtype=tf.int32),
        tf.TensorSpec(shape=[2], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.int32)])
def create_indices(content_region, sizes, steps, step_ind):
    gx, gy = tf.meshgrid(
        tf.range(sizes[0])[tf.random.uniform((), 0, steps[0], dtype=tf.int32)::steps[0]],
        tf.range(sizes[1])[tf.random.uniform((), 0, steps[1], dtype=tf.int32)::steps[1]])
    gx = tf.random.shuffle(tf.reshape(gx, [-1]))
    gy = tf.random.shuffle(tf.reshape(gy, [-1]))
    cm = tf.gather(tf.gather(content_region[...,0], gx, axis=0), gy, axis=1)[:,0]
    retval = tf.concat([gx[:,tf.newaxis], gy[:,tf.newaxis]], axis=1)
    retval = tf.cast(retval[cm][:step_ind], tf.float32)
    return retval


def create_style_features(
    style_features: List[tf.Tensor],
    style_region: tf.Tensor,
    n_loop: int,
    max_samples: int) -> tf.Tensor:
    """output shape: (batch, max_samples*n_loop, 1, channels)"""
    masked_features = None
    style_region = utils.resize_like(style_region, style_features[0])

    for _ in range(n_loop):
        bias = (style_region.numpy().max() < 0.1).astype(np.float32)
        mask = (style_region.numpy().flatten() + bias) > 0.5

        # create indices
        h, w = utils.get_h_w(style_features[0])
        xx, xy = np.meshgrid(np.arange(h), np.arange(w))
        xx = np.expand_dims(xx.flatten(), 1)
        xy = np.expand_dims(xy.flatten(), 1)
        indices = np.concatenate([xx, xy], 1)
        indices = indices[mask]

        samples = min(max_samples, indices.shape[0])

        np.random.default_rng().shuffle(indices)
        xx = indices[:samples, 0]
        xy = indices[:samples, 1]

        feats = None
        for i, target in enumerate(style_features):
            if i > 0 and style_features[i].shape[1] < style_features[i-1].shape[1]:
                xx = xx /2.0
                xy = xy /2.0
            
            h, w = utils.get_h_w(target)
            _xx = utils.to_tensor(np.clip(xx, 0, h-1).astype(np.int32))
            _xy = utils.to_tensor(np.clip(xy, 1, w-1).astype(np.int32))

            # -> (batch, samples, 1, channels)
            _feature = gather_xyz(target, _xx, _xy, collect_axis=1, keepdims=True)
            if feats is None:
                feats = tf.identity(_feature)
            else:
                feats = tf.concat([feats, _feature], axis=3)

        # shape: [batch_size, min_index, 1, channels*len(outputs)]
        batch, channels = utils.get_shape_by_name(feats, 'b', 'c')

        feats = tf.reshape(feats, (batch, -1, 1, channels))
        if masked_features is None:
            masked_features = tf.identity(feats)
        else:
            masked_features = tf.concat([masked_features, feats], axis=1)

    return masked_features
