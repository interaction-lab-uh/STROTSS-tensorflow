import os
import argparse

debug_state = not __debug__

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('content', help='Content image path.', type=str)
    args.add_argument('style', help='Style image path.', type=str)
    args.add_argument('--output', '-o', help='Output image path.', type=str, required=False)
    args.add_argument('--output_same_shape', help='If set, output image is same shape as content image. deafult=False', action='store_true', required=False)
    args.add_argument('--content_region', help='Content region path. default=None.', type=str, required=False)
    args.add_argument('--style_region', help='Style region path. default=None.', type=str, required=False)
    args.add_argument('--alpha', help='Loss weight(see original paper). default=16.0', default=16., type=float, required=False)
    args.add_argument('--scale_level', '-l', help='Max scale for strotss, default=4 (=512 pixel) ', default=4, type=int, required=False)
    args.add_argument('--signature_text', help='Text to sign in image. default=None(not signed)', type=str, required=False)
    # optional
    args.add_argument('--train_iteration', '-i', help='Iteration. default=200', default=200, type=int, required=False)
    args.add_argument('--threth_denominator',
        help='Valid when region path is set. Used to detect labels in region masks.'
        'For example, value=255 supports 6 colors, 128 supports 21 colors. default is 128', default=128, type=float, required=False)
    args.add_argument('--threth_min_counts',
        help='Valid when region path is set. Used to detect labels in region masks. '
        'Region masks with an area less than this value will be ignored. default=-1, which is determined automatically',
        default=-1, type=int, required=False)
    args.add_argument('--optimize_mode', help='Preprocess mode. `caffe`(keras), `torch` or `torch_uniform`. default=`torch`.'
        '(NOTE: preprocess_mode=`caffe` cannot generate image correctly.)',
            default='torch', type=str, choices=['caffe', 'torch', 'torch_uniform'], required=False)
    args.add_argument('--save_all_outputs', help='Save output image each scales. deafult=False', action='store_true', required=False)
    args.add_argument('--use_all_vgg_layers', help='If True, all vgg layers will be used. deafult=False', action='store_true', required=False)

    if debug_state:
        args.add_argument('--inner_loops', help='See original code (inner_loop). default=5.', default=5, type=int, required=False)
        args.add_argument('--max_subsamps', help='See original code (samps). default=1000.', default=1000, type=int, required=False)
        args.add_argument('--num_sample_grids', help='See kolkin\'s paper. default=1024.', default=1024, type=int, required=False)

    # semd
    args.add_argument('--emd_mode', default='remd', type=str, choices=['remd', 'semd'], required=False)

    # sinkhorn
    args.add_argument('--semd_n', default=30, type=int, required=False)
    args.add_argument('--semd_eps', default=1e-01, type=float, required=False)

    # tfrecord
    args.add_argument('--record', help='If True, write a Graph to tensorBoard.', action='store_true', required=False)

    parsed_args = args.parse_args()

    if parsed_args.scale_level < 4:
        # less than 512px cannot be generated bacause of indices.
        raise ValueError('scale_level must be >= 4')

    import random
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.ops import numpy_ops
    numpy_ops.enable_numpy_methods_on_tensor()

    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['PYTHONHASHSEED'] = '0'
    
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    rng_np = np.random.default_rng(0)
    rng_tf = tf.random.Generator.from_seed(0)

    from sources import strotss, losses, tensor_ops

    losses.set_emd_algorithm(
        mode=parsed_args.emd_mode,
        semd_eps=parsed_args.semd_eps,
        semd_max_iter=parsed_args.semd_n)
    tensor_ops.set_rng(rng_np, rng_tf)
    if debug_state:
        strotss.set_global_parameters(
            in_loop=parsed_args.inner_loops,
            max_subsamps=parsed_args.max_subsamps,
            num_sample_grids=parsed_args.num_sample_grids)
    strotss.STROTSS(
        content_path            = parsed_args.content,
        style_path              = parsed_args.style,
        content_region_path     = parsed_args.content_region,
        style_region_path       = parsed_args.style_region,
        output_path             = parsed_args.output,
        keep_shape              = parsed_args.output_same_shape,
        alpha                   = parsed_args.alpha,
        train_iteration         = parsed_args.train_iteration,
        scale_max_level         = parsed_args.scale_level,
        threth_denominator      = parsed_args.threth_denominator,
        threth_min_counts       = parsed_args.threth_min_counts,
        save_all_outputs        = parsed_args.save_all_outputs,
        use_all_vgg_layers      = parsed_args.use_all_vgg_layers,
        optimize_mode           = parsed_args.optimize_mode,
        signature_text          = parsed_args.signature_text,
        record_to_tensorboard   = parsed_args.record,)
