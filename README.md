# STROTSS-tensorflow
Tensorflow implementation of [Style Transfer by Relaxed Optimal Transport and Self-Similarity](https://arxiv.org/abs/1904.12785).

Content |  Style | Output
:-------------------------:|:-------------------------:|:-------------------------:
<img height="200" src='https://github.com/ppza53893/STROTSS-tensorflow/blob/main/content_im.jpg?raw=true'> |  <img height="200" src='https://github.com/ppza53893/STROTSS-tensorflow/blob/main/style_im.jpg?raw=true'>|  <img height="200" src='https://github.com/ppza53893/STROTSS-tensorflow/blob/main/output.png?raw=true'>

## Environment
Tested on tensorflow >= 2.6.0. check `requirements.txt`.

## Usage
```text
python main.py content_im.jpg style_im.jpg -o output.jpg
```

### Change content weight
```text
python main.py content_im.jpg style_im.jpg -o output.jpg --alpha 8.0
```

### Using mask
```text
python main.py content_im.jpg style_im.jpg -o output.jpg --content_region content_im_guidance.jpg --style_region style_im_guidance.jpg
```

### Change scale level
```text
python main.py content_im.jpg style_im.jpg -o output.jpg --scale_level 5
```

### Using all vgg features
```text
python main.py content_im.jpg style_im.jpg -o output.jpg --use_all_vgg_layers
```

### Using Sinkhorn distance 
```text
python main.py content_im.jpg style_im.jpg -o output.jpg --emd_mode semd
```

eps = 1e-03, N = 50

```text
python main.py content_im.jpg style_im.jpg -o output.jpg --emd_mode semd --semd_eps 1e-03 --semd_n 50
```

### Others
See the other optionss:  ```python main.py -h```.
