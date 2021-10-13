import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--content', '-c', help='Content image path.', type=str, required=True)
    args.add_argument('--style', '-s', help='Style image path.', type=str, required=True)
    args.add_argument('--output', '-o', help='Output image path.', type=str, required=False)
    args.add_argument('--output_same_shape', help='If set, output image is same shape as content image. deafult=False', action='store_true', required=False)
    args.add_argument('--content_region', help='Content region path. default=None.', default=None, required=False)
    args.add_argument('--style_region', help='Style region path. default=None.', default=None, required=False)
    args.add_argument('--alpha', help='Loss weight(see original paper). default=16.0', default=16., type=float, required=False)
    args.add_argument('--train_iteration', '-i', help='Iteration. default=250', default=250, type=int, required=False)
    args.add_argument('--scale_level', '-l', help='Max scale for strotss, default=4 (=512 pixel) ', default=4, type=int, required=False)
    args.add_argument('--threth_denominator', '-d',
        help='Valid when region path is set. Used to detect labels in region masks.'
        'For example, value=255 supports 6 colors, 128 supports 21 colors. default is 255.', default=255, type=int, required=False)
    args.add_argument('--threth_min_counts', '-m',
        help='Valid when region path is set. Used to detect labels in region masks. '
        'Region masks with an area less than this value will be ignored. default=-1, which is determined automatically',
        default=-1, type=int, required=False)
    args.add_argument('--optimize_mode', help='Preprocess mode. `caffe`(keras weight), `vgg`(pytorch weight),'
        '`paper`(pytorch weight, original implement) or `uniform`(pytrch weight). default=`paper`.'
        '(NOTE: preprocess_mode=`caffe` cannot generate image correctly.)',
            default='paper', type=str, choices=['caffe', 'vgg', 'paper', 'uniform'], required=False)
    args.add_argument('--quiet', help='wheter log is quiet or not. deafult=False', action='store_true', required=False)
    args.add_argument('--save_all_outputs', help='Save output image each levels. deafult=False', action='store_true', required=False)
    args.add_argument('--advanced_inner_loops', help='see original code (inner_loop). default=5.', default=5, type=int, required=False)
    args.add_argument('--advanced_max_subsamps', help='see original code (samps). default=1000.', default=1000, type=int, required=False)
    args.add_argument('--advanced_num_sample_grids', help='see kolkin\'s paper. default=1024.', default=1024, type=int, required=False)

    parsed_args = args.parse_args()

    from sources import strotss
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
        optimize_mode           = parsed_args.optimize_mode,
        quiet                   = parsed_args.quiet,
        advanced_options        = [
            parsed_args.advanced_inner_loops,
            parsed_args.advanced_max_subsamps,
            parsed_args.advanced_num_sample_grids])
