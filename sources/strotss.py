import os
import math
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if 'CUDA_PATH' in os.environ and __debug__:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
    if os.name == 'nt':
        xla_flags = '--xla_gpu_cuda_data_dir="{}"'.format(os.environ['CUDA_PATH']).replace('\\', '/')
        os.environ['XLA_FLAGS'] = xla_flags

from datetime import datetime
from typing import List, Optional

import tensorflow as tf

if not __debug__:
    tf.config.run_functions_eagerly(True)
if 'CUDA_PATH' in os.environ and __debug__:
    tf.config.optimizer.set_jit(True)

from tensorflow.keras.optimizers import RMSprop

from . import losses
from . import model
from . import utils
from . import tensor_ops


QUIET = False


class STROTSS_core:
    """
    Training class.
    This is because STROTSS need to learn the parameters for different scales.
    """

    def __init__(
        self,
        feature_extractor: model.VGG16_Patch,
        content_regions: List[tf.Tensor],
        style_regions: List[tf.Tensor],
        alpha: float,
        iteration: int,
        advanced_args: List[int]):

        # model, optimizer, regions
        self.feature_extractor = feature_extractor
        self.optimizer = RMSprop(rho=0.99, epsilon=1e-08)
        self.content_regions = content_regions
        self.style_regions = style_regions

        # hyper-parameters
        self.alpha = utils.to_tensor(alpha, tf.float32)
        self.num_regions = utils.to_tensor(len(content_regions), tf.float32)

        # iteration
        self.iteration = iteration

        # for reshape
        # this is used to reduce the re-tracing
        self.flat_shape = utils.to_tensor(
            (1,-1,1,feature_extractor.dims),dtype=tf.int32, as_constant=True)
    
        # args
        self.in_loop = advanced_args[0]
        self.max_samps = advanced_args[1]
        self.samp_indices = advanced_args[2]

        # for multi-scale-strategy
        self.parameters = []
        self.content_image_features = []
        self.style_image_features = []
        self.known_shape = None
        self.resized_content_regions = []
        self.steps = None

        # convert to tf function and add signature to avoid re-tracing..
        self.build_tf_function()

    def build_tf_function(self):
        fdim = self.feature_extractor.dims
        subsamps_size = self.in_loop*self.max_samps
        losses.set_parameters(
            feature_dimensions=fdim,
            subsamps_size=subsamps_size,
            indices_size=self.samp_indices)
        
        # input_signature can protect re-tracing
        self.compute_loss_fn = tf.function(
            losses.compute_loss,
            input_signature=[
                tf.TensorSpec(
                    shape=[1, self.samp_indices, 1, fdim], dtype=tf.float32), #f_ic
                tf.TensorSpec(
                    shape=[1, subsamps_size, 1, fdim], dtype=tf.float32), #f_is
                tf.TensorSpec(
                    shape=[1, self.samp_indices, 1, fdim], dtype=tf.float32), #f_ics
                tf.TensorSpec(shape=(), dtype=tf.float32)]) # alpha
        self.bsampling = tf.function(
            tensor_ops.bilinear_resampling,
            input_signature=[
                tf.TensorSpec(shape=[1, None, None, None], dtype=tf.float32), # content feature
                tf.TensorSpec(shape=[1, None, None, None], dtype=tf.float32), # stylized feature
                tf.TensorSpec(shape=[self.samp_indices, 2], dtype=tf.float32), # indices
                tf.TensorSpec(shape=[3], dtype=tf.int32)]) # content_shape

    def train_step(self):
        """Re-tracing = (num_regions)*(num vgg model outputs)*(num multi scales)"""
        with tf.GradientTape() as tape:
            # get stylized features
            stylized = tensor_ops.fold_lap(self.parameters)
            stylized_image_features = self.feature_extractor(stylized)

            # initialize loss = 0.0
            loss = tensor_ops.init_value()

            # loops per regions.
            for sif, mask in zip(self.style_image_features, self.resized_content_regions):

                # initialize features = 0.
                # this is to avoid error of autograph convertion.
                f_ics = tensor_ops.init_value()
                f_ic = tensor_ops.init_value()

                # reshape style features.
                f_is = tf.reshape(sif, self.flat_shape)

                with tf.name_scope('extract_features'):
                    target_indices = tensor_ops.create_indices(
                        mask, self.known_shape[0,:2], self.steps, self.samp_indices)
                    for j, (cf, gf) in enumerate(zip(self.content_image_features, stylized_image_features)):
                        if j > 0 and self.known_shape[j, 0] < self.known_shape[j-1, 0]:
                            target_indices = target_indices / 2.
                        cf, gf = self.bsampling(cf, gf, target_indices, self.known_shape[j])

                        if j > 0:
                            f_ics = tf.concat([f_ics, gf], axis=3) # 2179
                            f_ic = tf.concat([f_ic, cf], axis=3) # 2179
                        else:
                            f_ics = gf
                            f_ic = cf

                loss += self.compute_loss_fn(f_ic, f_is, f_ics, self.alpha)
            # mean
            loss /= self.num_regions

        # backward
        grad = tape.gradient(loss, self.parameters)
        return loss, grad

    def multi_scale_strategy(
        self,
        style_image: tf.Tensor, # unused
        content_image: tf.Tensor,
        init_strotss_image: tf.Tensor,
        init_lr: float):
        # create variable
        self.parameters = tensor_ops.to_variable(
            tensor_ops.create_laplasian(init_strotss_image))

        # optimizer
        utils.set_optimizer_lr(self.optimizer, learning_rate=init_lr)

        # content features
        self.content_image_features = self.feature_extractor(content_image)

        # create known shapes
        # this is used to reduce the re-tracing
        kshape = []
        for cf in self.content_image_features:
            kshape.append(utils.get_shape_by_name(cf,'h','w','c'))
        self.known_shape = utils.to_tensor(kshape, tf.int32)
        
        if isinstance(self.samp_indices, int):
            self.samp_indices = utils.to_tensor(self.samp_indices, tf.int32, True)

        # style features
        self.style_image_features = []
        style_feat = self.feature_extractor(style_image)
        for style_region in self.style_regions:
            self.style_image_features.append(
                tensor_ops.create_style_features(
                    style_features=style_feat,
                    style_region=style_region,
                    n_loop=self.in_loop,
                    max_samples=self.max_samps))        

        # prepare
        h, w = content_image.shape[2:]
        areas = float(((h*w)//16384)**0.5)
        self.steps = utils.to_tensor(
            [max(1, math.floor(areas)), max(1, math.ceil(areas))], tf.int32, True)
        self.resized_content_regions = [
            tf.cast(utils.resize_like(cr, content_image), tf.bool) for cr in self.content_regions]

        # convert to tf.function
        train_step = tf.function(self.train_step)

        # run
        for i in range(self.iteration):
            if i> 0 and i%200 == 0:
                utils.set_optimizer_lr(self.optimizer, factor=0.1)

            loss, grad = train_step()
            self.optimizer.apply_gradients(zip(grad, self.parameters))

            if not QUIET:
                print_str = 'Step/Iter: {}/{} - Loss: {:.4f}'.format(i+1, self.iteration, loss)
                if i < self.iteration - 1:
                    print_str = utils.ljust_print(print_str)
                    print('\r'+print_str, end='', flush=True)
                else:
                    print('\r'+print_str, end=' - ', flush=True)

        self.alpha /= 2.0
        return tensor_ops.fold_lap(self.parameters)


def STROTSS(
    content_path: str,
    style_path: str,
    content_region_path: Optional[str],
    style_region_path: Optional[str],
    output_path: Optional[str],
    keep_shape: bool,
    alpha: float,
    train_iteration: int,
    scale_max_level: int,
    threth_denominator: int,
    threth_min_counts: int,
    save_all_outputs: bool,
    optimize_mode: str,
    quiet: bool,
    advanced_options: List[int]):

    global QUIET
    QUIET = quiet

    timer = utils.Timer()

    # set scale
    scales = [2<<(5+i) for i in range(scale_max_level)]
    
    if not quiet:
        scale_from_to = ''
        for i, s in enumerate(scales):
            if i>0:
                scale_from_to += ' -> {}'.format(s)
            else:
                scale_from_to += str(s)
        print('Multi scale:',scale_from_to)
        if optimize_mode == 'caffe':
            print('NOTE: preprocess_mode=`caffe` cannot generate image correctly.')
        

    # read image
    content_image = utils.read_image(content_path, optimize_mode=optimize_mode)
    style_image = utils.read_image(style_path, optimize_mode=optimize_mode)

    if not quiet:
        print('Input content: {}x{}'.format(*utils.get_h_w(content_image))+\
            ' - Input style: {}x{}'.format(*utils.get_h_w(style_image)))

    # extract regions
    content_regions, style_regions = utils.extract_regions(
        content_r_path=content_region_path or content_path,
        style_r_path=style_region_path or style_path,
        threth_denominator=threth_denominator,
        threth_min_counts=threth_min_counts,
        noregion = not (content_region_path and style_region_path))

    output_path = output_path or datetime.now().strftime("%Y%m%d-%H%M%S")+'.png'

    strotss_in = STROTSS_core(
        feature_extractor=model.VGG16_Patch(optimize_mode=optimize_mode),
        content_regions=content_regions,
        style_regions=style_regions,
        alpha=alpha,
        iteration=train_iteration,
        advanced_args=advanced_options)

    # multi-scale strategy (64px -> ...)
    for i, scale in enumerate(scales):
        # iniitalize learning rate
        learning_rate = 2e-03

        # resize images
        content = utils.resize_image(content_image, scale)
        style = utils.resize_image(style_image, scale)
        if not quiet:
            sys.stdout.write(
                'Scale: {}'.format(scale) +\
                ' - Content size: {}x{}'.format(*utils.get_h_w(content)) +\
                ' - Style size: {}x{}'.format(*utils.get_h_w(style))+'\n')

        # make laplasian
        lap_content = tensor_ops.make_laplasian(content)

        # iniitalize output
        if i == 0:
            # first: add style mean color to bottom laplasian image.
            strotss_result = lap_content + tensor_ops.to_mean_color(style)
        elif i > 0 and scale != scales[-1]:
            # second ~ before last: add output image to `n`th laplasian image.
            strotss_result = utils.resize_like(strotss_result, content)
            strotss_result = lap_content + strotss_result
        else:
            # last: resize only, change initial learning rate.
            strotss_result = utils.resize_like(strotss_result, content)
            learning_rate = 1e-3
  
        # start
        timer.start()
        strotss_result = strotss_in.multi_scale_strategy(
            style_image = style,
            content_image = content,
            init_strotss_image = strotss_result,
            init_lr = learning_rate)
        t = timer.stop(save_time=True, return_time=True)
        if not quiet:
            print('Time: {}s'.format(t))

        # if requied
        if save_all_outputs:
            f, ext = os.path.splitext(output_path)
            utils.write_image(
                tensor_ops.clip_and_normalize(
                    strotss_result[0],
                    None,
                    optimize_mode=optimize_mode), f + '_scale_{}'.format(scale) + ext)
    # write image
    base = None if not keep_shape else content_image
    
    utils.write_image(
        tensor_ops.clip_and_normalize(
            strotss_result[0], base, optimize_mode=optimize_mode), output_path)
    sys.stdout.write('Total training time: {:.3f}s\n'.format(timer.total))
    sys.stdout.write('Saved image to {}.\n'.format(output_path))
