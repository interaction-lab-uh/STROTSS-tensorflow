# STROTSS-tensorflow

Tensorflow implementation of [Style Transfer by Relaxed Optimal Transport and Self-Similarity](https://arxiv.org/abs/1904.12785).

Content |  Style | Output
:-------------------------:|:-------------------------:|:-------------------------:
<img height="200" src='https://github.com/ppza53893/STROTSS-tensorflow/blob/main/content_im.jpg?raw=true'> |  <img height="200" src='https://github.com/ppza53893/STROTSS-tensorflow/blob/main/style_im.jpg?raw=true'>|  <img height="200" src='https://github.com/ppza53893/STROTSS-tensorflow/blob/main/output.png?raw=true'>

## Environment

Tested on tensorflow >= 2.6.0. check `requirements.txt`.

## Usage

```bash
python run_strotss.py content_im.jpg style_im.jpg -o output.jpg
```
