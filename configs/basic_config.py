import logging
import torch

def set_ghost_bn_layer_paras(num_features):
  ghost_bn_layer_paras['num_features']=num_features
  return ghost_bn_layer_paras

class ConvLayer:
  def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.padding = padding
    self.bias = bias

logger_config = {'log_filename':'logs/project1_test.txt',
                  'level': logging.DEBUG,
                }
SEED = 1
cuda = torch.cuda.is_available()
data = {
  'img_augs':{'random_rotation':{'angle_range': (-7.0, 7.0), 'fill':(1,)}},
   'normalize_paras':[(0.1307,), (0.3081,)],
   'dataloader_args': dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64),
   'data_kind' : {"dataset_type":"open_datasets", "dataset_name": "mnist", 'datasets_location':'data/datasets'},
}

ghost_bn_layer_paras = {
  'num_splits':2,
  'num_features':0
}

network_config = {
  'convblock1':{
      "conv_layer":ConvLayer(in_channels=1, out_channels=8, kernel_size=(3,3), padding=0, bias=False),
      "relu": True,
      "ghost_bn": set_ghost_bn_layer_paras(8)
  },
  'convblock2':{
      "conv_layer":ConvLayer(in_channels=8, out_channels=16, kernel_size=(3,3), padding=0, bias=False),
      "relu": True,
      "ghost_bn": set_ghost_bn_layer_paras(16)
  },
  'pool1':{
    
  }
  'convblock3':{
      "conv_layer":ConvLayer(in_channels=8, out_channels=16, kernel_size=(3,3), padding=0, bias=False),
      "relu": True,
      "ghost_bn": set_ghost_bn_layer_paras(16)
  },

}





