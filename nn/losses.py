import tensorflow as tf


def mse(x: tf.Tensor, y: tf.Tensor, axis=None, keepdims=False) -> tf.Tensor:
    return tf.reduce_mean(tf.square(x - y), axis=axis, keepdims=keepdims)


def mae(x: tf.Tensor, y: tf.Tensor, axis=None, keepdims=False) -> tf.Tensor:
    return tf.reduce_mean(tf.abs(x - y), axis=axis, keepdims=keepdims)


def cosine_distance(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x = tf.nn.l2_normalize(x, axis=1)
    y = tf.nn.l2_normalize(y, axis=1)
    return 1 - tf.matmul(x, y, transpose_b=True)


def l2_distance(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x_sq = tf.reshape(tf.reduce_sum(x ** 2, axis=1), (-1, 1))
    y_sq = tf.reshape(tf.reduce_sum(y ** 2, axis=1), (1, -1))

    matrix = x_sq + y_sq - 2. * tf.matmul(x, y, transpose_b=True)
    matrix = tf.maximum(matrix, 1e-06) / tf.cast(tf.shape(x)[1], dtype=matrix.dtype)
    return tf.sqrt(matrix)


dist_metrics = {'cosine': cosine_distance, 'l2': l2_distance,
                'both': lambda x, y: cosine_distance(x, y)+l2_distance(x, y)}


def reshape_2d(x: tf.Tensor, channel_axis: int=-1) -> tf.Tensor:
    if x.shape.dims == 2:
        return x
    x = tf.squeeze(x)
    x = tf.reshape(x, (-1, tf.shape(x)[channel_axis]))
    return x


def moment_matching(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x = reshape_2d(x)
    y = reshape_2d(y)

    xm = tf.reduce_mean(x, axis=0, keepdims=True)
    ym = tf.reduce_mean(y, axis=0, keepdims=True)

    cx = x - xm
    cy = y - ym

    xv = tf.matmul(cx, cx, transpose_a=True) / tf.cast(tf.shape(x)[0], x.dtype)
    yv = tf.matmul(cy, cy, transpose_a=True) / tf.cast(tf.shape(y)[0], y.dtype)

    return mae(xv, yv) + mae(xm, ym)


def self_similarity(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    x = reshape_2d(x)
    y = reshape_2d(y)
    
    x_dist = cosine_distance(x, x)
    x_dist = x_dist / tf.maximum(tf.reduce_sum(x_dist, axis=0), 1e-12)

    y_dist = cosine_distance(y, y)
    y_dist = y_dist / tf.maximum(tf.reduce_sum(y_dist, axis=0), 1e-12)
    
    loss = mae(x_dist, y_dist)
    return loss * tf.cast(tf.shape(y)[0], loss.dtype)


def relaxed_emd(x: tf.Tensor, y: tf.Tensor, distance: str='cosine') -> tf.Tensor:
    x = reshape_2d(x)
    y = reshape_2d(y)
    
    C = dist_metrics[distance](x, y)
    
    R_X = tf.reduce_mean(tf.reduce_min(C, axis=1))
    R_Y = tf.reduce_mean(tf.reduce_min(C, axis=0))
    
    return tf.maximum(R_X, R_Y)


def sinkhorn_knopp(x: tf.Tensor, y: tf.Tensor,
                   distance: str = 'cosine',
                   l: int = 10,
                   N_iter: int = 30) -> tf.Tensor:
    # TODO: untested
    tf.assert_greater(l, 0, message="l must be greater than 0")
    x = reshape_2d(x)
    y = reshape_2d(y)
    
    M = dist_metrics[distance](x, y)
    K = tf.exp(-M * l)
    shape = (tf.shape(K)[0], 1)
    
    p = tf.ones_like(shape) / tf.cast(tf.shape(x)[0], tf.float32)
    
    u = tf.ones_like(shape)
    v = tf.ones_like(shape)
    
    for _ in tf.range(N_iter):
        u = p / tf.maximum(K @ v, 1e-12)
        v = p / tf.maximum(tf.matmul(K, u, transpose_a=True), 1e-12)

    return tf.reduce_sum(u * ((K * M) @ v))
