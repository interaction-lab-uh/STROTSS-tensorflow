# STROTSS-tensorflow
Tensorflow implementation of [Style Transfer by Relaxed Optimal Transport and Self-Similarity](https://arxiv.org/abs/1904.12785).

## Environment
Tested on tensorflow 2.6.0. check `requirements.txt`.

## Usage
```text
python main.py -c content_im.jpg -s style_im.jpg -o output.jpg
```

### Change content weight
```text
python main.py -c content_im.jpg -s style_im.jpg -o output.jpg --alpha 8.0
```

### Add regions
```text
python main.py -c content_im.jpg -s style_im.jpg -o output.jpg --content_region content_im_guidance.jpg --style_region style_im_guidance.jpg
```

### Change scale level
```text
python main.py -c content_im.jpg -s style_im.jpg -o output.jpg --scale_level 5
```

### Use all vgg features
```text
python main.py -c content_im.jpg -s style_im.jpg -o output.jpg --use_all_vgg_layers
```

### Use Sinkhorn distance 
```text
python main.py -c content_im.jpg -s style_im.jpg -o output.jpg --emd_mode semd
```

eps = 1e-03, N = 50

```text
python main.py -c content_im.jpg -s style_im.jpg -o output.jpg --emd_mode semd --semd_eps 1e-03 --semd_n 50
```

### Others
See the other optionss:  ```python main.py -h```.
