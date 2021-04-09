from __future__ import division
from __future__ import print_function

from easydict import EasyDict as edict
import yaml
import numpy as np


__C = edict()
cfg = __C
__C.VIEW_NUM = 15
__C.INTRINSIC_FILE = '../Calibration/result_dir/intrinsic.json'
__C.EXTRINSIC_FILE = '../Calibration/result_dir/extrinsic.json'
__C.VIDEOS_DIR = ''
__C.HAND = 'RIGHT'
__C.OPT_VERT_ETERATION = 1000
__C.OPT_SEG_ETERATION = 100
__C.MANO_FILE = './data/MANO_RIGHT.pkl'
__C.VIDEOS_DIR = '/data/cyj/etri/hand_videos'
__C.RESULT_DIR = '/data/cyj/etri/result'
__C.BATCH_SIZE = 1
__C.SEG_SIZE = 224
__C.AXIS = 0


def _merge_a_into_b(a, b):
    """Merge config dictionary 'a' into b."""
    assert type(a) is edict, "config 'a' is not dictionary"

    for k, v in a.items():
        if not k in b:
            raise KeyError(f'{k} is not a valid config option')

        b_type = type(b[k])
        if b_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'type mismatch: expected {type(v)} but got {type(b[k])}')

        if type(v) is edict:
            try:
                _merge_a_into_b(a, b)
            except:
                raise KeyError(f"error in config key: {k}")
        else:
            b[k] = v
def cfg_from_file(filename):
    """Load a config file with yaml format and merge it into the options"""
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)