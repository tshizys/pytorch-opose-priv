# --------------------------------------------------------
# Copyright (c) 2015 BUPT-Priv
# Licensed under The MIT License [see LICENSE for details]
# Written by Yang Lu
# --------------------------------------------------------

import os
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict
import numpy as np

cur_pth = os.getcwd()

__C = edict()
cfg = __C

#
# Global options
#
__C.workers = 8  # number of data loading workers
__C.pixel_mean = [0.485, 0.456, 0.406]  # mean value of imagenet
__C.pixel_std = [0.229, 0.224, 0.225]  # std value of imagenet
__C.rng_seed = 3  # manual seed
__C.eps = 1e-14  # A very small number that's used many times
__C.eps5 = 1e-5  # A small number that's used many times
__C.gpu_ids = '0,1'  # gpu_ids, like: 0,1,2,3

#
# Classification options
#
__C.CLS = edict()
__C.CLS.arch = 'resnet18'  # model architecture
__C.CLS.data_root = '~/Database/ILSVRC2017/Data/CLS-LOC/'  # path to dataset
__C.CLS.train_folder = 'train'  # train folder
__C.CLS.val_folder = 'val'  # val folder
__C.CLS.epochs = 100  # number of total epochs to run
__C.CLS.train_batch = 256  # train batchsize of all gpus
__C.CLS.test_batch = 200  # test batchsize
__C.CLS.base_lr = 0.1  # base learning rate
__C.CLS.lr_schedule = [30, 60]  # decrease learning rate at these epochs
__C.CLS.gamma = 0.1  # base_lr is multiplied by gamma on lr_schedule
__C.CLS.momentum = 0.9  # momentum
__C.CLS.weight_decay = 1e-4  # weight_decay
__C.CLS.fix_bn = False  # fix bn params
__C.CLS.num_classes = 1000  # number of classes
__C.CLS.base_size = 256  # base size
__C.CLS.crop_size = 224  # crop size
__C.CLS.rotation = []  # list, randomly rotate the image by angle, etc. [-10, 10]
__C.CLS.pixel_jitter = []  # list, random pixel jitter, etc. [-20, 20]
__C.CLS.grayscale = 0  # float, randomly convert image to gray-scale with a probability, etc. 0.1
__C.CLS.disp_iter = 20  # display iteration
__C.CLS.ckpt = 'ckpts/imagenet/resnet18/'  # path to save checkpoint
__C.CLS.resume = ''  # path to latest checkpoint
__C.CLS.start_epoch = 0  # manual epoch number (useful on resume)
__C.CLS.pretrained = ''  # path to pretrained model
__C.CLS.cosine_lr = True  # using cosine learning rate
__C.CLS.validate = True  # validate
__C.CLS.evaluate = False  # evaluate

#
# Semantic Segmentation options
#
__C.SEG = edict()
__C.SEG.arch_encoder = 'resnet50_dilated16'  # architecture of net_encoder
__C.SEG.arch_decoder = 'c1_bilinear'  # architecture of net_decoder
__C.SEG.weights_encoder = ''  # weights to finetune net_encoder
__C.SEG.weights_decoder = ''  # weights to finetune net_decoder
__C.SEG.fc_dim = 2048  # fc_dim
__C.SEG.data_root = '~/Database/ADE2017/images/'
__C.SEG.list_train = '~/Database/ADE2017/training.txt'
__C.SEG.list_val = '~/Database/ADE2017/validation.txt'
__C.SEG.ignore_label = 255
__C.SEG.flip = True
__C.SEG.scale_range = [0.5, 2.0]
__C.SEG.label_shift = 0
__C.SEG.batch_size_per_gpu = 4  # input batch size of each gpu
__C.SEG.epochs = 100  # number of total epochs to run
__C.SEG.lr_encoder = 1e-3  # encoder base learning rate
__C.SEG.lr_decoder = 1e-2  # decoder base learning rate
__C.SEG.lr_pow = 0.9  # power in poly to drop LR
__C.SEG.momentum = 0.9  # momentum
__C.SEG.weight_decay = 1e-4  # weight_decay
__C.SEG.fix_bn = False  # fix bn params
__C.SEG.num_classes = 150  # number of classes
__C.SEG.input_size = 384  # input image size
__C.SEG.disp_iter = 20  # display iteration
__C.SEG.ckpt = 'ckpts'  # path to save checkpoint
__C.SEG.resume_encoder = ''  # path to latest net_encoder checkpoint
__C.SEG.resume_decoder = ''  # path to latest net_decoder checkpoint
__C.SEG.start_epoch = 0  # manual epoch number (useful on resume)
__C.SEG.validate = True  # validate
__C.SEG.evaluate = False  # evaluate
__C.SEG.visualize = False  # visualize

#
# Open Pose options
#
__C.OPOSE = edict()
__C.OPOSE.arch_encoder = 'vgg19s8'  # architecture of net_encoder
__C.OPOSE.arch_decoder = 'cpm6s'  # architecture of net_decoder
__C.OPOSE.weights_encoder = ''  # weights to finetune net_encoder
__C.OPOSE.weights_decoder = ''  # weights to finetune net_decoder
__C.OPOSE.stride = 8
__C.OPOSE.theta = 1.0
__C.OPOSE.sigma = 7.0
__C.OPOSE.data_root = '/mnt/lvmhdd/MSCOCO2017'
__C.OPOSE.info_root = '/home/xiaojie/workspace/pose/Pytorch_Realtime_Multi-Person_Pose_Estimation/data/'
__C.OPOSE.fc_dim = 512  # fc_dim
__C.OPOSE.planes = 128  # fc_dim
__C.OPOSE.batch_size_per_gpu = 10  # input batch size of each gpu
__C.OPOSE.epochs = 100  # number of total epochs to run
__C.OPOSE.base_lr = 0.00004  # encoder base learning rate
__C.OPOSE.lr_schedule = [30, 60]  # decrease learning rate at these epochs
__C.OPOSE.gamma = 0.333  # base_lr is multiplied by gamma on lr_schedule
__C.OPOSE.momentum = 0.9  # momentum
__C.OPOSE.weight_decay = 5e-4  # weight_decay
__C.OPOSE.fix_bn = False  # fix bn params
__C.OPOSE.num_points = 19  # number of classes
__C.OPOSE.num_limbs = 19  # number of classes
__C.OPOSE.input_size = 368  # input image size
__C.OPOSE.rotation = []  # list, randomly rotate the image by angle, etc. [-10, 10]
__C.OPOSE.disp_iter = 10  # display iteration
__C.OPOSE.ckpt = 'ckpts'  # path to save checkpoint
__C.OPOSE.resume = ''  # path to latest net checkpoint
__C.OPOSE.start_epoch = 0  # manual epoch number (useful on resume)
__C.OPOSE.validate = True  # validate
__C.OPOSE.evaluate = False  # evaluate
__C.OPOSE.visualize = False  # visualize

__C.OPOSE.workers= 6
__C.OPOSE.display= 10
__C.OPOSE.best_loss= 12345678.9


            


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
