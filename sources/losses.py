from logging import getLogger
logger = getLogger('strotss')
logger.setLevel(10)

import tensorflow as tf

# Global params
FEATURE_DIM = 2179
SUBSAMPS = 5000
INDICES = 1024

# REMD or SEMD
EMD_ALGORITHM = None

# SEMD
SEMD_EPS = 1e-01
SEMD_MAX_ITER = 30

def set_parameters(
    feature_dimensions: int,
    subsamps_size: int,
    indices_size: int):
    global FEATURE_DIM, SUBSAMPS, INDICES
    FEATURE_DIM = feature_dimensions
    SUBSAMPS = subsamps_size
    INDICES = indices_size


def set_emd_algorithm(mode: str, semd_eps: float, semd_max_iter: int):
    global EMD_ALGORITHM
    global SEMD_EPS, SEMD_MAX_ITER
    if mode == 'remd':
        EMD_ALGORITHM = remd
    else:
        logger.info('Using Sinkhorn distance. It may be slower than REMD.')
        EMD_ALGORITHM = semd
        SEMD_EPS = semd_eps
        SEMD_MAX_ITER = semd_max_iter


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


def color_space_transform(param: tf.Tensor) -> tf.Tensor:  
    # RGB -> YUV transform matrix
    krnls = tf.constant([
        [0.299, -0.14714119, 0.61497538],
        [0.587, -0.28886916, -0.51496512],
        [0.114, 0.43601035, -0.10001026]], tf.float32)

    with tf.name_scope('color_space_transform'):
        x = tf.transpose(param, (3, 0, 1, 2)) # [3, b, h, w]
        x = tf.reshape(x, (3, -1)) # [3, b*h*w]
        x = krnls @ x
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


def semd(dismat: tf.Tensor):
    with tf.name_scope('sinkhorn_earth_movers_distance'):
        # m, n
        # m = 1/1024, n = 1/5000
        m = tf.constant(1/INDICES, dtype=tf.float32, shape=(INDICES,1)) # [1024, 1]
        n = tf.constant(1/SUBSAMPS, dtype=tf.float32, shape=(SUBSAMPS,1)) # [5000, 1]

        # copy m, n
        u = tf.identity(m) # [1024, 1]
        v = tf.identity(n) # [5000, 1]

        # exponential
        k = tf.exp(-dismat/SEMD_EPS) # [1024, 5000]
        kp = tf.constant(INDICES, tf.float32) * k #[1024, 5000]

        count = tf.constant(0, tf.int32) # count = 0
        max_iter = tf.constant(SEMD_MAX_ITER, tf.int32)

        def loop_condition(count, u, v):
            return count < max_iter

        def loop_body(count, u, v):
            ktu = tf.maximum(tf.transpose(k) @ u, 1e-12) # [5000, 1024] @ [1024, 1] -> [5000, 1]
            v = n / ktu # [5000, 1]
            u = 1. / tf.maximum(kp @ v, 1e-12) # [1024, 5000] @ [5000, 1] -> [1024, 1]

            count = count + 1
            return count, u, v

        _, u, v = tf.while_loop(
            cond = loop_condition,
            body = loop_body,
            loop_vars= [count, u, v])

        # 1. u*k -> [1024, 1] * [1024, 5000] -> [1024, 5000]
        # 2. u*k*v.t -> [1024, 5000] * [1, 5000] -> [1024, 5000]
        # 3. u*k*v.t*dismat -> [1024, 5000] * [1024, 5000] -> [1024, 5000]
        loss = tf.reduce_sum(u * k * tf.transpose(v) * dismat)
        return loss


################################ loss functions ################################


def self_similarity_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute self-similarity loss."""
    with tf.name_scope('self_similarity'):
        y_true = reshape_to_2d(y_true[..., :FEATURE_DIM]) # [2179, 1024]
        y_pred = reshape_to_2d(y_pred[..., :FEATURE_DIM]) # [2179, 1024]

        # cosine distance
        d_true = cosine_distance(y_true, y_true)
        d_true = d_true / tf.maximum(tf.reduce_sum(d_true, axis=0), 1e-12)
        d_pred = cosine_distance(y_pred, y_pred)
        d_pred = d_pred / tf.maximum(tf.reduce_sum(d_pred, axis=0), 1e-12)

        loss = abs_mean(d_true, d_pred) * tf.cast(y_true.shape[0], tf.float32)

        return loss


def moment_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute moment loss."""
    with tf.name_scope('moment'):
        y_true = tf.squeeze(y_true[..., :FEATURE_DIM]) # [2179, 5000]
        y_pred = tf.squeeze(y_pred[..., :FEATURE_DIM]) # [2179, 1024]
        
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


def palette_emd_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    r"""Compute EMD (Earth Mover's Distance) loss, for palette (lp)"""
    with tf.name_scope('palette_remd'):
        y_true = color_space_transform(y_true[..., :3]) # [3, 5000]
        y_pred = color_space_transform(y_pred[..., :3]) # [3, 1024]

        # add l2 distance metric
        dismat = cosine_distance(y_true, y_pred) + l2_distance(y_true, y_pred)
        return EMD_ALGORITHM(dismat)


def emd_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    r"""Compute EMD (Earth Mover's Distance) loss."""
    with tf.name_scope('remd'):
        y_true = reshape_to_2d(y_true[..., :FEATURE_DIM]) # [2179, 5000]
        y_pred = reshape_to_2d(y_pred[..., :FEATURE_DIM]) # [2179, 1024]

        dismat = cosine_distance(y_true, y_pred)
        return EMD_ALGORITHM(dismat)


def compute_loss(
    f_ic: tf.Tensor,
    f_is: tf.Tensor,
    f_ics: tf.Tensor,
    alpha: tf.Tensor):
    with tf.name_scope('compute_loss'):
        inv_alpha = 1./tf.maximum(alpha, 1.)
        l_c = self_similarity_loss(f_ic, f_ics)
        l_m = moment_loss(f_is, f_ics)
        l_r = emd_loss(f_is, f_ics)
        l_p = palette_emd_loss(f_is, f_ics)
        # \mathcal{L}(E, I_c, I_s) = \\
        # \frac{\alpha*l_c + l_m + l_r + \frac{1}{\alpha}l_p}{2+\alpha+\frac{1}{\alpha}}
        return ((alpha*l_c) + l_m + l_r + (inv_alpha*l_p)) / (2. + alpha + inv_alpha)
