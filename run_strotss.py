import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tqdm

from nn import strotss_utils as strotss
from nn import utils
from nn.losses import moment_matching, relaxed_emd, self_similarity
from nn.model import VGG

utils.make_logger('STROTSS')


class ContentLoss(tf.Module):
    @tf.Module.with_name_scope
    def __call__(self, target: tf.Tensor, prediction: tf.Tensor) -> tf.Tensor:
        return self_similarity(prediction, target)


class StyleLoss(tf.Module):
    def __init__(self, target: tf.Tensor, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.target = target
        self.inv_alpha = 1 / max(alpha, 1)

    @tf.Module.with_name_scope
    def __call__(self, prediction: tf.Tensor) -> tf.Tensor:
        l_m = moment_matching(self.target, prediction)
        l_remd = relaxed_emd(self.target, prediction)
        target = strotss.convert_rgb_to_yuv(self.target)
        pred = strotss.convert_rgb_to_yuv(prediction)
        l_palette = relaxed_emd(target, pred, distance='both')
        return l_m + l_remd + (self.inv_alpha*l_palette)


def run(args: argparse.Namespace):
    timer = utils.Timer()
    timer.start()

    use_layers = ['block1_conv1',
                  'block1_conv2',
                  'block2_conv1',
                  'block2_conv2',
                  'block3_conv1',
                  'block3_conv2',
                  'block3_conv3',
                  'block4_conv3',
                  'block5_conv3']
    vgg = VGG(use_layers, vgg_type='16', use_keras_weight=args.use_keras_weight)

    content = utils.load_image(args.content_path, max_size=args.max_size)
    style = utils.load_image(args.style_path, max_size=args.max_size)

    if args.content_mask and args.style_mask:
        content_masks, style_masks = strotss.load_mask(args.content_mask, args.style_mask, max_size=args.max_size)
        utils.logger.info(f'Loaded {len(content_masks)} masks.')
        use_mask = True
    elif not args.content_mask and not args.style_mask:
        use_mask = False
    else:
        raise ValueError('Either both content and style masks must be provided or neither.')

    # content loss, optimizer
    loss_content = ContentLoss()
    opt = tf.keras.optimizers.RMSprop(rho=0.99, epsilon=1e-08, learning_rate=args.lr)

    alpha = args.alpha * 16.0 * (3500 if args.use_keras_weight else 1)

    # sampling
    sampling = strotss.Sampling(1024)

    for i in range(args.level):
        scl = 2<<(5+i)

        # utils.resize
        scl_content = utils.resize(content, scl)
        scl_style = utils.resize(style, scl)
        
        # laplacian
        laplacian = strotss.make_laplacian(scl_content)

        # init variables
        if i == 0:
            stylized = laplacian + tf.reduce_mean(scl_style, axis=(1, 2), keepdims=True)
        elif 0 < i < args.level-1:
            stylized = utils.resize_like(stylized, scl_content) + laplacian
            tf.keras.backend.set_value(opt.lr, args.lr)
        else:
            stylized = utils.resize_like(stylized, scl_content)
            tf.keras.backend.set_value(opt.lr, args.lr/2)
        st_variables = [tf.Variable(img) for img in strotss.make_laplacian_pyramid(stylized)]  

        # constant loss parameter
        loss_denom = (2. + alpha + 1. / max(alpha, 1.))      

        # content, style features
        content_feat = [scl_content] + vgg(scl_content)
        style_feat = [scl_style] + vgg(scl_style)
        if use_mask:
            loss_styles = []
            for sm in style_masks:
                loss_s_cls = StyleLoss(sampling(style_feat, mask=sm), alpha=alpha)
                loss_styles.append(loss_s_cls)

            # train step
            @tf.function
            def train_step():
                with tf.GradientTape() as tape:
                    img = strotss.fold_laplacian_pyramid(st_variables)
                    pred = [img] + vgg(img)

                    loss = 0.
                    # for tqdm
                    loss_s_a = 0.
                    loss_c_a = 0.
                    for i in range(len(content_masks)):
                        c_feat, p_feat = sampling.bilinear(content_feat, pred, mask=content_masks[i])
                        loss_c = loss_content(c_feat, p_feat)
                        loss_s = loss_styles[i](p_feat)
                        loss += (alpha * loss_c + loss_s) / loss_denom
                        loss_c_a += loss_c
                        loss_s_a += loss_s
                    loss /= len(content_masks)
                grads = tape.gradient(loss, st_variables)
                loss_c_a /= len(content_masks)
                loss_s_a /= len(content_masks)
                return {'loss': loss, 'loss_c': loss_c_a, 'loss_s': loss_s_a, 'grads': grads}
        else:
            # style loss
            loss_style = StyleLoss(sampling(style_feat), alpha=alpha)

            # train step
            @tf.function
            def train_step():
                with tf.GradientTape() as tape:
                    img = strotss.fold_laplacian_pyramid(st_variables)
                    pred = [img] + vgg(img)
                    c_feat, p_feat = sampling.bilinear(content_feat, pred)

                    loss_c = loss_content(c_feat, p_feat)
                    loss_s = loss_style(p_feat)
                    loss = (alpha * loss_c + loss_s) / loss_denom
                grads = tape.gradient(loss, st_variables)
                return {'loss': loss, 'loss_c': loss_c, 'loss_s': loss_s, 'grads': grads}

        # run
        with tqdm.tqdm(range(args.max_iter)) as pbar:
            for it in pbar:
                result = train_step()
                opt.apply_gradients(zip(result['grads'], st_variables))
                pbar.set_description(f"Scale: {scl:4d} - It: {it+1:4d}")
                pbar.set_postfix({'loss': f'{result["loss"]:.3f}',
                                  'loss_c': f'{result["loss_c"]:.3f}',
                                  'loss_s': f'{result["loss_s"]:.3f}'})

        stylized = strotss.fold_laplacian_pyramid(st_variables)
        alpha /= 2.

    final = strotss.postprocess(stylized)

    timer.stop()
    utils.logger.info(f"Done in {timer.elapsed_time:.2f}s.")
    utils.write_image(final, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("content_path", type=str)
    parser.add_argument("style_path", type=str)
    parser.add_argument("--content_mask", type=str, default=None)
    parser.add_argument("--style_mask", type=str, default=None)
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument('--level', type=float, default=4)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--use_keras_weight", action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--output_path", "-o", type=str, default=f"output.jpg")
    args = parser.parse_args()
    utils.set_gpu(args.gpu_id)
    run(args)
