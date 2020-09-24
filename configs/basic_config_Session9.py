import logging
import torch

logger_config = {'log_filename':'logs/Session7_assignment',
                  'level': logging.INFO,
                  'format':'%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                  'datefmt':'%d-%m-%Y:%H:%M:%S'
                }
SEED = 1
cuda = torch.cuda.is_available()
agumentation_package = 'Albumentation'
data = {
   #'img_augs':{'color_light':{},'color_medium':{'hue_shift_limit':20, 'sat_shift_limit':50, 'val_shift_limit':50},'CenterCrop':{'height':4,'width':4},'RandomCrop':{'height':4,'width':4},'HorizontalFlip':{}},
   'img_augs':{'ShiftScaleRotate':{'shift_limit':0.0625,'scale_limit':0.1,'rotate_limit':[-7,7]} ,'RandomSizedCrop':{'height':32,'width':32,'min_max_height':[28,28]},'HorizontalFlip':{}},
   'normalize_paras':[(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
   'dataloader_args': dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=4),
   'data_kind' : {"dataset_type":"open_datasets", "dataset_name": "CIFAR10", 'datasets_location':'data/datasets'},
}

ghost_bn_layer_paras = {
  'num_splits':2,
  # 'num_features':0
}

"""
How to use logger? - Copy paste the below lines where ever logger is needed

from utils import logger_utils
logger = logger_utils.get_logger(__name__)
"""
optimizer_paras = {
  'lr':0.1,
  'momentum':0.9,
  'weight_decay':0.0   ## For L2 regularization
}

lr_scheduler_steplr_paras = {
  'step_size':150,
  'gamma':0.01
}

l1_lambda = 0.0 ## For L1 regularization

EPOCHS = 350

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





