import sys
import shutil
import functools
import importlib


__all__ = [
    'dp',
    'm3e',
    'm3p',
    'debug_set_module',
    'debug_use_torch',
    'debug_use_tensorflow',
    'debug_use_tensorflow_numpy']


MODULE = None
MODULE_NAME = None
PRINT_FUNC = None


def dp(*print_args, name=None, exit_=True):
    if __debug__:
        return
    column = shutil.get_terminal_size().columns
    PRINT_FUNC('\033[38;5;011m'+'#'*column+'\033[0m')
    if name:
        PRINT_FUNC('debug::',name)
    for arg in print_args:
        if hasattr(arg, 'numpy'):
            if MODULE_NAME == 'torch':
                arg = arg.detach().cpu()
            PRINT_FUNC('\t',arg.numpy())
        else:
            PRINT_FUNC(arg)
    PRINT_FUNC('\033[38;5;011m'+'#'*column+'\033[0m')
    if exit_:
        exit()


def debug_set_module(module):
    global MODULE, MODULE_NAME, PRINT_FUNC
    MODULE = module
    MODULE = MODULE.__name__
    PRINT_FUNC = print


def debug_use_torch():
    global MODULE, MODULE_NAME, PRINT_FUNC
    MODULE = importlib.import_module('torch')
    MODULE_NAME = 'torch'
    PRINT_FUNC = print


def debug_use_tensorflow():
    debug_use_tensorflow_numpy()


def debug_use_tensorflow_numpy():
    global MODULE, MODULE_NAME, PRINT_FUNC
    MODULE = importlib.import_module('tensorflow._api.v2.experimental.numpy')
    MODULE_NAME = 'tensorflow'
    PRINT_FUNC = functools.partial(
        importlib.import_module('tensorflow').print, output_stream=sys.stdout)


def mmm(tensor, name=None, exit_=False):
    if MODULE_NAME == 'torch':
        _shape = tensor.size()
    elif MODULE_NAME == 'tensorflow':
        _shape = tensor.shape
    else:
        _shape = None
    _mean = MODULE.mean(tensor)
    _min = MODULE.min(tensor)
    _max = MODULE.max(tensor)
    if name:
        name += ' (shape,mean,min,max)'
    else:
        name = 'shape,mean,min,max'
    dp(_shape, _mean, _min, _max, name=name, exit_=exit_)


def m3p(tensor, name=None):
    if isinstance(tensor, (list, tuple)):
        for i, t in enumerate(tensor):
            if name:
                _name = name+' index #{}'.format(i)
            else:
                _name = 'index #{}'.format(i)
            mmm(t, name=_name, exit_=False)
    else:
        mmm(tensor, name=name, exit_=False)


def m3e(tensor, name=None):
    if isinstance(tensor, (list, tuple)):
        for i, t in enumerate(tensor):
            if name:
                _name = name+' index #{}'.format(i)
            else:
                _name = 'index #{}'.format(i)
            if i< len(tensor) -1:
                mmm(t, name=_name, exit_=False)
            else:
                mmm(t, name=_name, exit_=True)
    else:
        mmm(tensor, name=name, exit_=True)

