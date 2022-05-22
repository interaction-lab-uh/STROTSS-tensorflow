import math
from functools import partialmethod
from typing import List, Optional, Union, Tuple

import numpy as np
import tensorflow as tf

from nn import utils
from nn.rand import tf_rng


def _clip_and_cast(x, minval, maxval, dtype) -> tf.Tensor:
    minval = tf.cast(minval, x.dtype)
    maxval = tf.cast(maxval, x.dtype)
    x = tf.clip_by_value(x, minval, maxval)
    x = tf.cast(x, dtype)
    return x


class Sampling(tf.Module):
    def __init__(self, sample_size: int, **kwargs):
        super().__init__(**kwargs)
        self.sample_size = sample_size

    def _sample(self,
                xs: List[tf.Tensor],
                indices: tf.Tensor,
                bilinear_sampling: bool) -> tf.Tensor:
        feats = None
        index = None
        for i in range(len(xs)):
            current = xs[i]
            if i > 0 and current.shape[1] < xs[i-1].shape[1]:
                if index is None:
                    index = 1 if not (math.log2(current.shape[1]) % 1) else 2
                y = xs[i-1].shape[index] / current.shape[index]
                indices /= y

            current = tf.squeeze(current)
            h, w, *_ = current.shape
            gx, gy = tf.split(indices, 2, axis=1)

            if bilinear_sampling:
                gx = tf.reshape(gx, [-1])
                gy = tf.reshape(gy, [-1])
                gxf = tf.math.floor(gx)
                dx = gx - gxf
                gyf = tf.math.floor(gy)
                dy = gy - gyf

                wrap_a = tf.reshape((1.-dx) * (1.-dy), (-1, 1))
                wrap_b = tf.reshape((1.-dx) * dy, (-1, 1))
                wrap_c = tf.reshape(dx * (1.-dy), (-1, 1))
                wrap_d = tf.reshape(dx * dy, (-1, 1))

                gxf = _clip_and_cast(gxf, 0, h-1, tf.int32)
                gyf = _clip_and_cast(gyf, 0, w-1, tf.int32)
                gxf_b = _clip_and_cast(gxf+1, 0, h-1, tf.int32)
                gyf_b = _clip_and_cast(gyf+1, 0, w-1, tf.int32)

                ind_a = gxf * w + gyf
                ind_b = gxf * w + gyf_b
                ind_c = gxf_b * w + gyf
                ind_d = gxf_b * w + gyf_b

                current = tf.reshape(current, [h*w, -1])
                gathered = (tf.gather(current, ind_a, axis=0) * wrap_a +
                            tf.gather(current, ind_b, axis=0) * wrap_b +
                            tf.gather(current, ind_c, axis=0) * wrap_c +
                            tf.gather(current, ind_d, axis=0) * wrap_d)
            else:
                gx = _clip_and_cast(gx, 0, h-1, tf.int32)
                gy = _clip_and_cast(gy, 0, w-1, tf.int32)
                gather_indices = tf.concat([gx, gy], axis=1)
                gathered = tf.gather_nd(current, gather_indices)
            
            if feats is not None:
                feats = tf.concat([feats, gathered], axis=1)
            else:
                feats = gathered
        return feats

    def _make_indices(self,
                      base_tensor: tf.Tensor,
                      bilinear_sampling: bool,
                      mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        _, h, w, *_ = base_tensor.shape
        if bilinear_sampling:
            area = math.sqrt((h*w)//(128**2))
            step_x, step_y = max(1, math.floor(area)), max(1, math.ceil(area))
            rand_off_x = tf_rng.uniform((), 0, step_x, dtype=tf.int32)
            rand_off_y = tf_rng.uniform((), 0, step_y, dtype=tf.int32)

            X = tf.range(h)[rand_off_x::step_x]
            Y = tf.range(w)[rand_off_y::step_y]

            X, Y = tf.meshgrid(X, Y)
        else:
            X, Y = tf.meshgrid(tf.range(h), tf.range(w))

        X = tf.reshape(X, [-1, 1])
        Y = tf.reshape(Y, [-1, 1])
        ret = tf.concat([X, Y], axis=1)

        if mask is not None:
            mask = tf.image.resize(mask, [h, w])
            if tf.reduce_max(mask) < 0.1:
                mask = tf.greater(mask + 1, 0.5)
            else:
                mask = tf.greater(mask, 0.5)
            mask = tf.gather_nd(mask, ret)
            mask = tf.reshape(mask, [-1])
            ret = ret[mask]

        # The author's implementation of shuffling is incorrect.
        # https://github.com/nkolkin13/STROTSS/blob/a5706f96d4a408594f3a3d3ad98c1a1b2581f9e8/stylize_objectives.py#L211
        # x and y coordinates must not be shuffled individually, but in pairs
        # Shuffling them individually would affect style/content images in areas other than the mask
        ret = tf.random.shuffle(ret)
        ret = tf.cast(ret[:self.sample_size], tf.float32)
        return ret

    @tf.Module.with_name_scope
    def __call__(self,
                 xs: List[tf.Tensor],
                 ys: Optional[List[tf.Tensor]] = None,
                 mask: Optional[tf.Tensor] = None,
                 bilinear_sampling: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        indices = self._make_indices(xs[0], bilinear_sampling, mask)
        ret = self._sample(xs, indices, bilinear_sampling)
        if ys:
            ret_y = self._sample(ys, indices, bilinear_sampling)
            return ret, ret_y
        return ret

    bilinear = partialmethod(__call__, bilinear_sampling=True)


def make_laplacian(x, return_downscale=False):
    hw = tf.shape(x)[1:3]
    hwd = tf.maximum(hw//2, 1)
    temp = tf.image.resize(x, hwd, method='bilinear')
    pyr = x - tf.image.resize(temp, hw, method='bilinear')
    if return_downscale:
        return pyr, temp 
    return pyr


def make_laplacian_pyramid(x: tf.Tensor, levels: int = 5) -> List[tf.Tensor]:
    xs = []
    curx = x
    for _ in range(levels):
        pyr, curx = make_laplacian(curx, return_downscale=True)
        xs.append(pyr)
    xs.append(curx)
    return xs


def fold_laplacian_pyramid(xs: List[tf.Tensor]) -> tf.Tensor:
    ret = xs[-1]
    for x in reversed(xs[:-1]):
        ret = x + tf.image.resize(ret, tf.shape(x)[1:3], method='bilinear')
    return ret


def convert_rgb_to_yuv(x: tf.Tensor) -> tf.Tensor:
    return tf.image.rgb_to_yuv(x[:, :3])


def postprocess(final: tf.Tensor) -> tf.Tensor:
    final = tf.clip_by_value(final, 0, 1)
    final = final - tf.reduce_min(final)
    final = final / tf.reduce_max(final)
    final = tf.cast(final * 255, tf.uint8)
    return final[0]


def load_mask(content_path: str, style_path: str, max_size: Optional[int],
              pixel_threth: int = 255, sample_threth: int = 10000):
    c_mask = utils.load_image(content_path, max_size, dtype=tf.uint8, batch_expand=False).numpy()
    c_mask = c_mask // pixel_threth * pixel_threth
    s_mask = utils.load_image(style_path, max_size, dtype=tf.uint8, batch_expand=False).numpy()
    s_mask = s_mask // pixel_threth * pixel_threth

    uniques, counts = np.unique(c_mask.reshape(-1, 3), axis=0, return_counts=True)

    # to avoid too small mask
    uniques = uniques[counts >= sample_threth]

    c_ret = []
    s_ret = []
    for unique in uniques:
        c_condition = (c_mask[...,0] == unique[0]) & (c_mask[...,1] == unique[1]) & (c_mask[...,2] == unique[2])
        s_condition = (s_mask[...,0] == unique[0]) & (s_mask[...,1] == unique[1]) & (s_mask[...,2] == unique[2])
        if np.any(c_condition) and np.any(s_condition):
            c_ret.append(tf.cast(c_condition, tf.float32)[..., tf.newaxis])
            s_ret.append(tf.cast(s_condition, tf.float32)[..., tf.newaxis])
    if not c_ret:
        raise Exception('No mask found')

    return c_ret, s_ret
