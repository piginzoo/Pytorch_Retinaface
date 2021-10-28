# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 32,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'batch_size': 24,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256,
}


common_dict = {
    'nms_threshold':0.4,
    'top_k':5000,
    'keep_top_k':750,
    'confidence_threshold':0.02,
    'visualization_threshold':0.5,
    'tboard_dir': 'logs/tboard',
    'val_batch_size': 24,
    'val_batch_num': 20
}
from collections import namedtuple
namedTupleConstructor = namedtuple('myNamedTuple', ' '.join(sorted(common_dict.keys())))
CFG = namedTupleConstructor(**common_dict)


