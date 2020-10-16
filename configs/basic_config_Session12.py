import logging
import torch
import math
import numpy as np
from collections import OrderedDict
import cv2


logger_config = {'log_filename':'logs/Session11_assignment',
                  'level': logging.INFO,
                  'format':'%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                  'datefmt':'%d-%m-%Y:%H:%M:%S'
                }
SEED = 1
cuda = torch.cuda.is_available()
agumentation_package = 'Albumentation'
# mean = np.array([0.4914, 0.4822, 0.4465])
data = {
   #'img_augs':{'color_light':{},'color_medium':{'hue_shift_limit':20, 'sat_shift_limit':50, 'val_shift_limit':50},'CenterCrop':{'height':4,'width':4},'RandomCrop':{'height':4,'width':4},'HorizontalFlip':{}},
  #  'img_augs':{'ShiftScaleRotate':{'shift_limit':0.0625,'scale_limit':0.1,'rotate_limit':[-7,7]} ,'RandomSizedCrop':{'height':32,'width':32,'min_max_height':[28,28]},'HorizontalFlip':{}},
  # 'img_augs':{'rotate':{'rotate_limit':[-7,7]} ,
  #              'cutout':dict(num_holes=1,max_h_size=16,max_w_size=16,fill_value=(0.4914, 0.4822, 0.4465),always_apply=False, p=0.5),
  #              'RandomSizedCrop':{'height':32,'width':32,'min_max_height':[28,28]},                                     
  #              'HorizontalFlip':{}},
   #'img_augs':OrderedDict(
   #        PadIfNeeded =dict(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, value=(0.4914, 0.4822, 0.4465), p=1.0),
   #        oneof_crop=dict(randomcrop=dict(height=32, width=32, p=0.9),centercrop=dict(height=32, width=32, p=0.1),p=1.0),
   #        fliplr=dict(p=0.2),
   #        cutout=dict(num_holes=1,max_h_size=16,max_w_size=16,fill_value=(0.4914, 0.4822, 0.4465),always_apply=False, p=0.1),
   #  ),
   'img_augs':OrderedDict(
            PadIfNeeded =dict(min_height=70, min_width=70, border_mode=cv2.BORDER_CONSTANT, value=(0.4914, 0.4822, 0.4465), p=1.0),
            oneof_crop=dict(randomcrop=dict(height=64, width=64, p=0.9),centercrop=dict(height=64, width=64, p=0.1),p=1.0),
            fliplr=dict(p=0.7),
            cutout=dict(num_holes=1,max_h_size=32,max_w_size=32,fill_value=(0.4914, 0.4822, 0.4465),always_apply=False,p=0.6),
          ),
   'normalize_paras':[(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
   'dataloader_args': dict(shuffle=True, batch_size=128, num_workers=16, pin_memory=True) if cuda else dict(shuffle=True, batch_size=128),
   'data_kind' : {"dataset_type":"open_datasets", "dataset_name": "tiny_imagenet", 'datasets_location':'data/datasets'},
}

# iaa.Fliplr(p = 1.0) # apply horizontal flip

# OneOf([
# 				RandomCrop(height=32, width=32, p=0.8),
# 				CenterCrop(height=32, width=32, p=0.2),
# 			], p=1.0),
# PadIfNeeded(min_height=40, min_width=40, border_mode=BORDER_CONSTANT,
# 					value=mean*255.0, p=1.0),
# 			OneOf([
# 				RandomCrop(height=32, width=32, p=0.8),
# 				CenterCrop(height=32, width=32, p=0.2),
# 			], p=1.0),


# ghost_bn_layer_paras = {
#   'num_splits':2,
#   # 'num_features':0
# }

"""
How to use logger? - Copy paste the below lines where ever logger is needed
from utils import logger_utils
logger = logger_utils.get_logger(__name__)
"""
optimizer_paras = { 
  'sgd':dict(lr=5.34E-02, momentum=0.9, weight_decay=0.0)  # weight_decay is for L2 regularization
  # 'adam':dict()
}


lr_scheduler = {
  # 'stepLR': dict(step_size=150, gamma=0.01, name='stepLR')
  'use_scheduler':True,
  # 'ReduceLROnPlateau': dict(mode='min', factor=0.2, patience=5, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, name='ReduceLROnPlateau')
  'OneCycleLR':dict(max_lr=0.10004247448979592, epochs=50, pct_start=15.0/50.0, steps_per_epoch = math.ceil(77000/128.0), 
                  cycle_momentum=False, div_factor=10, final_div_factor=1, anneal_strategy="linear", name='OneCycleLR'),
}
# 0.02002040714285714 - 88.29 - before applying suggested image aug
# 0.6671251275510204 - bad acc in epoch1 itself - before applying suggested image aug
# 0.1473558381245045 - 84.8 - before applying suggested image aug
# scheduler = OneCycleLR(optimizer, max_lr=best_lr, steps_per_epoch=len(data.train_loader),
#                       epochs=args.epochs, div_factor=10, final_div_factor=1,
#                       pct_start=5/args.epochs, anneal_strategy="linear")




# lr_scheduler_steplr_paras = {
#   'step_size':150,
#   'gamma':0.01
# }

l1_lambda = 0.0 ## For L1 regularization

EPOCHS = 50

# def set_ghost_bn_layer_paras(num_features):
#   ghost_bn_layer_paras['num_features']=num_features
#   return ghost_bn_layer_paras

# class ConvLayer:
#   def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
#     self.in_channels = in_channels
#     self.out_channels = out_channels
#     self.kernel_size = kernel_size
#     self.padding = padding
#     self.bias = bias



# network_config = {
#   'convblock1':{
#       "conv_layer":ConvLayer(in_channels=1, out_channels=8, kernel_size=(3,3), padding=0, bias=False),
#       "relu": True,
#       "ghost_bn": set_ghost_bn_layer_paras(8)
#   },
#   'convblock2':{
#       "conv_layer":ConvLayer(in_channels=8, out_channels=16, kernel_size=(3,3), padding=0, bias=False),
#       "relu": True,
#       "ghost_bn": set_ghost_bn_layer_paras(16)
#   },
#   'pool1':{
#     "stride":2,
#     "kernel_size":(2,2)
#   },
#   'convblock3':{
#       "conv_layer":ConvLayer(in_channels=16, out_channels=16, kernel_size=(3,3), padding=0, bias=False),
#       "relu": True,
#       "ghost_bn": set_ghost_bn_layer_paras(16)
#   },

# }

