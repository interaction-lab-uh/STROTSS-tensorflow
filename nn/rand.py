# reproducibility
import os

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import random
import numpy as np
import tensorflow as tf

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


np_rng = np.random.default_rng(0)
tf_rng = tf.random.Generator.from_seed(0)
