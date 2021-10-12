import tensorflow as tf


FEATURE_DIM = 2179
SUBSAMPS = 5000
INDICES = 1024


def set_parameters(
    feature_dimensions: int,
    subsamps_size: int,
    indices_size: int):
    global FEATURE_DIM, SUBSAMPS, INDICES
    FEATURE_DIM = feature_dimensions
    SUBSAMPS = subsamps_size
    INDICES = indices_size


################################ some operations ################################


def reshape_to_2d(params: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('reshape_to_2d'):
        x = tf.transpose(params, (3, 0, 1, 2)) # [2179, b, h, w]
        x = tf.reshape(x, (FEATURE_DIM, -1)) # [2179, b*h*w]
        x = tf.transpose(x) # [b*h*w, 2179]
        return x


def abs_mean(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('abs_mean'):
        _abs  = tf.abs(y_true - y_pred)
        return tf.reduce_mean(_abs)


def remd(dismat: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('relaxed_earth_movers_distance'):
        j_min = tf.reduce_min(dismat, axis=1)
        i_min = tf.reduce_min(dismat, axis=0)

        loss = tf.maximum(tf.reduce_mean(j_min), tf.reduce_mean(i_min))
        return loss


def rt2d_color_transform(param: tf.Tensor) -> tf.Tensor:
    # color transform matrix
    # TODO: find this matrix source...
    const = tf.constant([
        [0.577350,0.577350,0.577350],
        [-0.577350,0.788675,-0.211325],
        [-0.577350,-0.211325,0.788675]], tf.float32)

    with tf.name_scope('rt2d_color_transform'):
        x = tf.transpose(param, (3, 0, 1, 2)) # [3, b, h, w]
        x = tf.reshape(x, (3, -1)) # [3, b*h*w]
        x = const @ x
        return tf.transpose(x)


def l2_distance(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('l2_distance'):
        x = tf.reshape(tf.reduce_sum(y_pred**2, axis=1), (-1, 1))
        y = tf.reshape(tf.reduce_sum(y_true**2, axis=1), (1, -1))

        # (x-y)^2 = (x^2+y^2-2xy)
        dist = x + y - (2.0 * tf.matmul(y_pred, tf.transpose(y_true)))
        dist = tf.maximum(dist, 1e-05) / tf.constant(y_pred.shape[1], dist.dtype)
        return tf.sqrt(dist)


def cosine_distance(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    with tf.name_scope('Cosine_distance'):
        x_sum = tf.maximum(tf.reduce_sum(y_pred**2, axis=1, keepdims=True), 1e-05)
        x_norm = y_pred / tf.sqrt(x_sum)

        y_sum = tf.maximum(tf.reduce_sum(y_true**2, axis=1, keepdims=True), 1e-05)
        y_norm = y_true / tf.sqrt(y_sum)

        sim = x_norm @ tf.transpose(y_norm)
        return 1. - sim


################################ loss functions ################################


def self_similarity_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute self-similarity loss."""
    with tf.name_scope('self_similarity'):
        y_true = reshape_to_2d(y_true[..., :FEATURE_DIM])
        y_pred = reshape_to_2d(y_pred[..., :FEATURE_DIM])

        # cosine distance
        d_true = cosine_distance(y_true, y_true)
        d_pred = cosine_distance(y_pred, y_pred)

        loss = abs_mean(d_true, d_pred)
        return loss


def moment_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute moment loss."""
    with tf.name_scope('moment'):
        y_true = tf.squeeze(y_true[..., :FEATURE_DIM]) # 2179, 5000
        y_pred = tf.squeeze(y_pred[..., :FEATURE_DIM]) # 2179, 1024
        
        # mean
        m_true = tf.reduce_mean(y_true, axis=0, keepdims=True)
        m_pred = tf.reduce_mean(y_pred, axis=0, keepdims=True)

        # centering
        cent_pred = y_true - m_true
        cent_true = y_pred - m_pred

        # invariant variance
        var_pred = tf.transpose(cent_pred) @ cent_pred
        var_true = tf.transpose(cent_true) @ cent_true
        var_true = var_true / tf.constant(SUBSAMPS-1, var_true.dtype)
        var_pred = var_pred / tf.constant(INDICES-1, var_pred.dtype)

        loss = abs_mean(m_true, m_pred) + abs_mean(var_true, var_pred)
        return loss


def relaxed_emd_palette_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    r"""Compute REMD (Relaxed Earth Mover's Distance) loss, for palette loss (lp)"""
    with tf.name_scope('palette_remd'):
        y_true = rt2d_color_transform(y_true[..., :3])
        y_pred = rt2d_color_transform(y_pred[..., :3])

        # add l2 distance metric
        dismat = cosine_distance(y_true, y_pred) + l2_distance(y_true, y_pred)
        return remd(dismat)


def relaxed_emd_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    r"""Compute REMD (Relaxed Earth Mover's Distance) loss."""
    with tf.name_scope('remd'):
        y_true = reshape_to_2d(y_true[..., :FEATURE_DIM])
        y_pred = reshape_to_2d(y_pred[..., :FEATURE_DIM])

        dismat = cosine_distance(y_true, y_pred)
        return remd(dismat)


def compute_loss(
    f_ic: tf.Tensor,
    f_is: tf.Tensor,
    f_ics: tf.Tensor,
    alpha: tf.Tensor):
    with tf.name_scope('compute_loss'):
        inv_alpha = 1./tf.maximum(alpha, 1.)

        l_c = self_similarity_loss(f_ic, f_ics)

        l_m = moment_loss(f_is, f_ics)

        l_r = relaxed_emd_loss(f_is, f_ics)

        l_p = relaxed_emd_loss(f_is, f_ics)

        return ((alpha*l_c) + l_m + l_r + (inv_alpha*l_p)) / (2. + alpha + inv_alpha)
