import os
import math

from datetime import datetime
from typing import Any, List, Optional, Callable, Tuple
from logging import getLogger

logger = getLogger('strotss')
logger.setLevel(10)

import tensorflow as tf

from . import losses
from . import model
from . import utils
from . import tensor_ops


IN_LOOP = 5
MAX_SUBSAMPS = 1000
NUM_SAMPLE_GRIDS = 1024


def set_global_parameters(
    in_loop: int,
    max_subsamps: int,
    num_sample_grids: int):
    global IN_LOOP, MAX_SUBSAMPS, NUM_SAMPLE_GRIDS
    IN_LOOP = in_loop
    MAX_SUBSAMPS = max_subsamps
    NUM_SAMPLE_GRIDS = num_sample_grids


class STROTSS_trainer:
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
        iteration: int):

        # model, optimizer, regions
        self.feature_extractor = feature_extractor
        self.optimizer = tf.keras.optimizers.RMSprop(rho=0.99, epsilon=1e-08)
        self.content_regions = content_regions
        self.style_regions = style_regions

        # hyper-parameters
        self.alpha = utils.to_tensor(alpha, tf.float32)

        # total regions, used compute loss
        self.num_regions = utils.to_tensor(len(content_regions), tf.float32)

        # iteration
        self.iteration = iteration
    
        # indices
        self.samp_indices = utils.to_tensor(NUM_SAMPLE_GRIDS, tf.int32)

        # for multi-scale-strategy
        self.parameters = []
        self.content_features = []
        self.style_features = []
        self.featue_shapes = None
        self.resized_content_regions = []
        self.steps = None

        # convert to tf function and add signature to avoid re-tracing..
        self.build_tf_function()

    def build_tf_function(self):
        fdim = self.feature_extractor.dims
        subsamps_size = IN_LOOP*MAX_SUBSAMPS
        losses.set_global_parameters(
            feature_dimensions=fdim,
            subsamps_size=subsamps_size,
            indices_size=NUM_SAMPLE_GRIDS)
        
        # input_signature can protect re-tracing
        self.compute_loss: Callable[
            [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = tf.function(
                losses.nowrap_compute_loss,
                input_signature=[
                    tf.TensorSpec(
                        shape=[1, NUM_SAMPLE_GRIDS, 1, fdim], dtype=tf.float32), #f_ic
                    tf.TensorSpec(
                        shape=[1, subsamps_size, 1, fdim], dtype=tf.float32), #f_is
                    tf.TensorSpec(
                        shape=[1, NUM_SAMPLE_GRIDS, 1, fdim], dtype=tf.float32), #f_ics
                    tf.TensorSpec(shape=(), dtype=tf.float32)]) # alpha
        self.bilinear_resampling: Callable[
            [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor] = tf.function(
                tensor_ops.nowrap_bilinear_resampling,
                input_signature=[
                    tf.TensorSpec(shape=[1, None, None, None], dtype=tf.float32), # content feature
                    tf.TensorSpec(shape=[1, None, None, None], dtype=tf.float32), # stylized feature
                    tf.TensorSpec(shape=[NUM_SAMPLE_GRIDS, 2], dtype=tf.float32), # indices
                    tf.TensorSpec(shape=[4], dtype=tf.int32)]) # content_shape

    def train_step(self):
        """Re-tracing = (num_regions)*(num vgg model outputs)*(num multi scales)"""
        with tf.GradientTape() as tape:
            # get stylized features
            stylized = tensor_ops.fold_lap(self.parameters)
            stylized_features = self.feature_extractor(stylized)
            # initialize loss = 0.0
            loss = tensor_ops.init_zeros()
            # Loop by region
            for sif, mask in zip(self.style_features, self.resized_content_regions):
                # reshape style features.
                f_is = sif.reshape((1,-1,1,self.feature_extractor.dims))
                # create indices
                target_indices = tensor_ops.create_indices(
                    mask,
                    self.featue_shapes[0],
                    self.steps,
                    self.samp_indices)

                # resample features (index 0)
                f_ic, f_ics = self.bilinear_resampling(
                    self.content_features[0],
                    stylized_features[0],
                    target_indices,
                    self.featue_shapes[0])
                # loops per features
                for j in range(1, self.feature_extractor.layer_counts):
                    target_indices = target_indices /\
                        tf.cast(self.featue_shapes[j,3], target_indices.dtype)

                    # resample features
                    cf, xf = self.bilinear_resampling(
                        self.content_features[j],
                        stylized_features[j],
                        target_indices,
                        self.featue_shapes[j])

                    # concat along channel dimension
                    f_ic = tf.concat([f_ic, cf], axis=3)
                    f_ics = tf.concat([f_ics, xf], axis=3)

                loss += self.compute_loss(f_ic, f_is, f_ics, self.alpha)
            # mean
            loss /= self.num_regions

        # backward
        grad = tape.gradient(loss, self.parameters)
        return loss, grad

    def multi_scale_strategy(
        self,
        style_image: tf.Tensor,
        content_image: tf.Tensor,
        init_strotss_image: tf.Tensor,
        init_lr: float):
        # create variable
        self.parameters = tensor_ops.to_variable(
            tensor_ops.create_laplasian_pyramids(init_strotss_image))

        # optimizer
        utils.set_optimizer_lr(self.optimizer, learning_rate=init_lr)

        # content features
        self.content_features = self.feature_extractor(content_image)

        # create known shapes
        # this is used to reduce the re-tracing
        self.featue_shapes = []
        for i, cf in enumerate(self.content_features):
            fs = list(utils.get_shape_by_name(cf, 'h','w','c'))
            fs.append(1 + int(i > 0 and fs[0] < self.featue_shapes[i-1][0]))
            self.featue_shapes.append(fs)
        self.featue_shapes = utils.to_tensor(self.featue_shapes, dtype=tf.int32)

        # style features
        self.style_features = []
        style_feat = self.feature_extractor(style_image)
        for style_region in self.style_regions:
            self.style_features.append(
                tensor_ops.create_style_features(
                    style_features=style_feat,
                    style_region=style_region,
                    n_loop=IN_LOOP,
                    max_samples=MAX_SUBSAMPS))

        # prepare
        h, w = content_image.shape[2:]
        areas = math.sqrt((h*w)//16384)
        self.steps = utils.to_tensor(
            [max(1, math.floor(areas)), max(1, math.ceil(areas))], tf.int32, True)
        self.resized_content_regions = [
            tf.cast(utils.resize_like(cr, content_image), tf.bool) for cr in self.content_regions]

        # convert to tf.function
        train_step: Callable[
            [], Tuple[tf.Tensor, List[tf.Tensor]]] = tf.function(self.train_step)

        # run
        for i in range(self.iteration):
            if i> 0 and i%200 == 0:
                utils.set_optimizer_lr(self.optimizer, factor=0.1)

            loss, grad = train_step()
            self.optimizer.apply_gradients(zip(grad, self.parameters))

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
    use_all_vgg_layers: bool,
    optimize_mode: str,
    signature_text: Optional[str],
    record_to_tensorboard: bool):

    timer = utils.Timer()

    # set scale
    scales = [2<<(5+i) for i in range(scale_max_level)]

    scale_from_to = ''
    for i, s in enumerate(scales):
        if i>0:
            scale_from_to += ' -> {}'.format(s)
        else:
            scale_from_to += str(s)
    print('Scale: '+scale_from_to)

    if optimize_mode == 'caffe':
        logger.warning('preprocess_mode=`caffe` cannot generate image correctly.')

    # read image
    content_image = utils.read_image(content_path, optimize_mode=optimize_mode)
    style_image = utils.read_image(style_path, optimize_mode=optimize_mode)

    print('Input content: {}x{}'.format(*utils.get_h_w(content_image))+\
        ' - Input style: {}x{}'.format(*utils.get_h_w(style_image)))

    # extract regions
    content_regions, style_regions = utils.extract_regions(
        content_r_path=content_region_path or content_path,
        style_r_path=style_region_path or style_path,
        threth_denominator=threth_denominator,
        threth_min_counts=threth_min_counts,
        noregion = not (content_region_path and style_region_path))

    current_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_path or current_date+'.png'

    strotss_in = STROTSS_trainer(
        feature_extractor=model.VGG16_Patch(
            optimize_mode=optimize_mode,
            use_all_features=use_all_vgg_layers),
        content_regions=content_regions,
        style_regions=style_regions,
        alpha=alpha,
        iteration=train_iteration)

    # multi-scale strategy (64px -> ...)
    for i, scale in enumerate(scales):
        if record_to_tensorboard:
            logdir = 'logs/{}/{}'.format(current_date, scale)
            writer = tf.summary.create_file_writer(logdir)
        # iniitalize learning rate
        learning_rate = 2e-03

        # resize images
        content = utils.resize_image(content_image, scale)
        style = utils.resize_image(style_image, scale)
        print(
            'Scale: {}'.format(scale) +\
            ' - Content size: {}x{}'.format(*utils.get_h_w(content)) +\
            ' - Style size: {}x{}'.format(*utils.get_h_w(style)))

        # laplasian
        lap_content = tensor_ops.create_laplasian(content)

        # iniitalize output
        if i == 0:
            # first: add style mean color to bottom laplasian image.
            strotss_result = lap_content + tensor_ops.to_mean_color(style)
        elif i > 0 and scale != scales[-1]:
            # second ~ before last: add output image to `n`th laplasian image.
            strotss_result = utils.resize_like(strotss_result, content)
            strotss_result = strotss_result + lap_content
        else:
            # last: resize only, change initial learning rate.
            strotss_result = utils.resize_like(strotss_result, content)
            learning_rate = 1e-3
  
        # start
        timer.start()
        if record_to_tensorboard:
            tf.summary.trace_on()
        strotss_result = strotss_in.multi_scale_strategy(
            style_image = style,
            content_image = content,
            init_strotss_image = strotss_result,
            init_lr = learning_rate)
        t = timer.stop(save_time=True, return_time=True)
        print('Time: {}s'.format(t))
        if record_to_tensorboard:
            with writer.as_default():
                tf.summary.trace_export(
                    name='strotss_lv_{}'.format(scale),
                    step=0,
                    profiler_outdir=logdir)
            writer.close()

        # if requied
        if save_all_outputs:
            f, ext = os.path.splitext(output_path)
            utils.write_image(
                tensor_ops.clip_and_denormalize(
                    strotss_result[0],
                    base_image = None,
                    optimize_mode=optimize_mode), f + '_scale_{}'.format(scale) + ext)
    # write image
    base = None if not keep_shape else content_image

    strotss_result = tensor_ops.clip_and_denormalize(
        strotss_result[0], base, optimize_mode=optimize_mode)
    if signature_text:
        if signature_text.lower() == 'auto':
            signature_text = output_path
        strotss_result = utils.add_signature(strotss_result, signature_text)
    
    utils.write_image(strotss_result, output_path)
    print('Total training time: {:.3f}s'.format(timer.total))
    print('Saved image to {}.\n'.format(output_path))
    if record_to_tensorboard:
        print('Saved tensorboard log to logs/{}.\n'.format(current_date))
