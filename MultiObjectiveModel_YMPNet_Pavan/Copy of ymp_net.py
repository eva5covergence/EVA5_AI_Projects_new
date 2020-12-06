"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn
import numpy as np

# from torchutils.model.base_model import BaseModel
# from torchutils.model.mde_net.blocks import FeatureFusionBlock, Interpolate, _make_encoder
# from torchutils.model.mde_net.yolo_layer import YOLOLayer
# import torchutils.model.mde_net.maskrcnn as mrcnn

from base_model import BaseModel
from blocks import FeatureFusionBlock, Interpolate, _make_encoder
from models import YOLOLayer
from utils import torch_utils
#from models import get_yolo_layers
#from utils.parse_config import parse_model_cfg
#from models import create_modules


class YMPNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, yolo_props, path=None, features=256, non_negative=True, inference=False):
        """Init.
        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        # print("Loading weights: ", path)

        super(YMPNet, self).__init__()
        
        # self.module_defs = parse_model_cfg(cfg)
        # self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # self.yolo_layers = get_yolo_layers(self)

        use_pretrained = True if path is None else False
        self.inference = inference

        self.pretrained, self.scratch = _make_encoder(features, use_pretrained)

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)
        
        # YOLO head
        conv_output = (int(yolo_props["classes"]) + 5) * int((len(yolo_props["anchors"]) / 3))
        
        # Custom
        self.yolo1_custom_learner = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        # self.yolo1 = YOLOLayer(yolo_props["anchors"][:3],
        #                       nc=int(yolo_props["classes"]),
        #                       img_size=(416, 416),
        #                       yolo_index=0,
        #                       layers=[],
        #                       stride=32)
        
        self.yolo2_learner_s1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),# Custom
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.yolo2_learner_s2 = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),## Missing in yolo weights
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            )
        self.yolo2_learner_s3 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.106.Conv2d.weight torch.Size([128, 384, 1, 1])
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.107.Conv2d.weight torch.Size([256, 128, 3, 3])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.103.Conv2d.weight torch.Size([128, 256, 1, 1])
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.109.Conv2d.weight torch.Size([256, 128, 3, 3])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.108.Conv2d.weight torch.Size([128, 256, 1, 1])
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.111.Conv2d.weight torch.Size([256, 128, 3, 3])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.yolo2_learner = nn.Sequential(
            nn.Conv2d(256, conv_output, kernel_size=1, stride=1, padding=0) ## module_list.112.Conv2d.weight torch.Size([27, 256, 1, 1])
        )
        self.yolo2 = YOLOLayer(yolo_props["anchors"][:3],
                               nc=int(yolo_props["classes"]),
                               img_size=(416, 416),
                               yolo_index=0,
                               layers=[],
                               stride=32)
        
        self.yolo3_learner_s1 = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0, bias=False), ## missing in yolo weights
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.yolo3_learner_s2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False), ## Custom
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )
        self.yolo3_learner_s3 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.94.Conv2d.weight torch.Size([256, 768, 1, 1])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.95.Conv2d.weight torch.Size([512, 256, 3, 3])
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.91.Conv2d.weight torch.Size([256, 512, 1, 1])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.97.Conv2d.weight torch.Size([512, 256, 3, 3])
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False), ## module_list.96.Conv2d.weight torch.Size([256, 512, 1, 1])
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), ## module_list.99.Conv2d.weight torch.Size([512, 256, 3, 3])
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )
        
        
        self.yolo3_learner = nn.Sequential(
            nn.Conv2d(512, conv_output, kernel_size=1, stride=1, padding=0) ## module_list.100.Conv2d.weight torch.Size([27, 512, 1, 1])
        )
        self.yolo3 = YOLOLayer(yolo_props["anchors"][3:6],
                               nc=int(yolo_props["classes"]),
                               img_size=(416, 416),
                               yolo_index=1,
                               layers=[],
                               stride=16)
        
        self.yolo4_learner_s1 = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False), # Custom
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )                      
        self.yolo4_learner = nn.Sequential(
            nn.Conv2d(1024, conv_output, kernel_size=1, stride=1, padding=0) ## module_list.88.Conv2d.weight torch.Size([27, 1024, 1, 1])
        )
        self.yolo4 = YOLOLayer(yolo_props["anchors"][6:],
                               nc=int(yolo_props["classes"]),
                               img_size=(416, 416),
                               yolo_index=2,
                               layers=[],
                               stride=8)
        
        #self.module_defs = parse_model_cfg('cfg/yolov3-custom.cfg')
        #self.module_list, self.routs = create_modules(self.module_defs, 512)
        #self.yolo_layers = get_yolo_layers(self)
        
    def forward_once(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        augment = True
        if not augment:
            return self.forward(x)
        else:  # Augment images (inference and test only) https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward(xi)[1][0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale

            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi

            y = torch.cat(y, 1)
            d = self.forward(x)[0]
            return d, y
            
    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        
        # Depth
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        # print(f"layer_2 - {layer_2.size()}") # torch.Size([2, 512, 52, 52])
        # print(f"layer_3 - {layer_3.size()}") # torch.Size([2, 1024, 26, 26])
        # print(f"layer_4 - {layer_4.size()}") # torch.Size([2, 2048, 13, 13])

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        # print(f"layer_4_rn - {layer_4_rn.size()}")

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # print(f"path_1 - {path_1.size()}")

        depth_out = self.scratch.output_conv(path_1)
        # #depth_out = depth_out.numpy()
        # bits=2
        # depth_min = depth_out.min()
        # depth_max = depth_out.max()
        # max_val = (2**(8*bits))-1
        # if depth_max - depth_min > np.finfo("float").eps:
        #     depth_out = max_val * (depth_out - depth_min) / (depth_max - depth_min)
        # else:
        #     depth_out = torch.zeros_like(depth_out)
        # depth_out = depth_out.type(torch.int16)
        # depth_out.astype("uint16")
        # depth_out = torch.Tensor(depth_out)
        
        # Object Detection
        # print(f"x - {x.shape}")
        out = []
        
        ## Skip connection from input to yolo layers
        
        custom_yolo1_out1 = self.yolo1_custom_learner(x) # skip connection from input layer # 1024 13 13
        
        yolo4_out1 = self.yolo4_learner_s1(layer_4) # 1024 13 13
        # print(f"dbg custom_yolo1_out1 - {custom_yolo1_out1.size()}")
        # print(f"dbg yolo4_out1 - {yolo4_out1.size()}")
        yolo4_out2 = custom_yolo1_out1 + yolo4_out1 # 1024 13 13
        if self.inference:
            try:
                yolo4_out_view, yolo4_out = self.yolo4(self.yolo4_learner(yolo4_out2), out)
            except Exception:
                yolo4_out = self.yolo4(self.yolo4_learner(yolo4_out2), out)
            # y = self.yolo4(self.yolo4_learner(yolo4_out2), out)
            # print(f"yyyyy - {type(y)},len(y)")
            # print(f"yolo4_out_view - {type(yolo4_out_view)}")
            # print(f"yolo4_out_view shape - {yolo4_out_view.shape}")
            # print(f"yolo4_out shape - {yolo4_out.shape}")
        else:
            yolo4_out = self.yolo4(self.yolo4_learner(yolo4_out2), out)
        # print(f"yolo4_out2 - {yolo4_out2.size()}") 
        # print(f"yolo4_out - {yolo4_out[0].size()}") # torch.Size([2, 3, 13, 13, 9])
        
        yolo3_out11 = self.yolo3_learner_s1(yolo4_out2) # 256 26 26
        yolo3_out12 = self.yolo3_learner_s2(layer_3) # 512 26 26
        yolo3_out1 = torch.cat((yolo3_out11,yolo3_out12),1) # 768 26 26
        yolo3_out2 = self.yolo3_learner_s3(yolo3_out1) # 768 26 26-> 256 26 26-> 512 26 26
        if self.inference:
            try:
                yolo3_out_view, yolo3_out = self.yolo3(self.yolo3_learner(yolo3_out2), out)
            except Exception:
                yolo3_out = self.yolo3(self.yolo3_learner(yolo3_out2), out)
        else:
            yolo3_out = self.yolo3(self.yolo3_learner(yolo3_out2), out)
        # yolo3_out = self.yolo3(self.yolo3_learner(yolo3_out2), out)
        # print(f"yolo3_out2 - {yolo3_out2.size()}")
        # print(f"yolo3_out - {yolo3_out[0].size()}") # torch.Size([2, 3, 26, 26, 9])
        
        
        yolo2_out11 = self.yolo2_learner_s1(layer_2) # 256 52 52
        yolo2_out12 = self.yolo2_learner_s2(yolo3_out2) # 128 52 52
        yolo2_out1 = torch.cat((yolo2_out11,yolo2_out12),1) # 384 52 52
        yolo2_out2 = self.yolo2_learner_s3(yolo2_out1) # 256 52 52
        if self.inference:
            try:
                yolo2_out_view, yolo2_out = self.yolo2(self.yolo2_learner(yolo2_out2), out)
            except Exception:
                yolo2_out = self.yolo2(self.yolo2_learner(yolo2_out2), out)
            # x = self.yolo2(self.yolo2_learner(yolo2_out2), out)
            # print(f"xxxxx - {type(x)}, {x.shape}")
            # print(f"yolo2_out_view - {type(yolo2_out_view)}")
            # print(f"yolo2_out_view shape - {yolo2_out_view.shape}")
            # print(f"yolo2_out shape - {yolo2_out.shape}")
        else:
            yolo2_out = self.yolo2(self.yolo2_learner(yolo2_out2), out)
        # yolo2_out = self.yolo2(self.yolo2_learner(yolo2_out2), out)
        # print(f"yolo2_out - {type(yolo2_out)}")
        
        
        # for i,item in enumerate(yolo2_out):
        #     print(f"yolo2_out[{i}] - {item.size()}") # torch.Size([2, 3, 52, 52, 9])
        
        
        # print(f"depth_out - {depth_out.size()}") # torch.Size([2, 1, 416, 416])
        # print(f"yolo2_out length - {len(yolo2_out)}")
        # print(f"yolo3_out length - {len(yolo3_out)}")
        # print(f"yolo4_out length - {len(yolo4_out)}")
        if not self.inference:
            yolo2_out_temp = [torch.unsqueeze(item, dim=0) for item in yolo2_out]
            yolo2_out_final = torch.stack(yolo2_out_temp, dim=1)
            yolo2_out_final = torch.squeeze(yolo2_out_final, dim=0)
            
            yolo3_out_temp = [torch.unsqueeze(item, dim=0) for item in yolo3_out]
            yolo3_out_final = torch.stack(yolo3_out_temp, dim=1)
            yolo3_out_final = torch.squeeze(yolo3_out_final, dim=0)
            
            yolo4_out_temp = [torch.unsqueeze(item, dim=0) for item in yolo4_out]
            yolo4_out_final = torch.stack(yolo4_out_temp, dim=1)
            yolo4_out_final = torch.squeeze(yolo4_out_final, dim=0)
            
            # print(f"yolo2_out_final - {yolo2_out_final.size()}")
            # print(f"yolo3_out_final - {yolo3_out_final.size()}")
            # print(f"yolo4_out_final - {yolo4_out_final.size()}")
            return torch.squeeze(depth_out, dim=1), [yolo4_out_final, yolo3_out_final, yolo2_out_final]
        else:
            # print(f"yolo2_out - {yolo2_out.size()}")
            # print(f"yolo3_out - {yolo3_out.size()}")
            # print(f"yolo4_out - {yolo4_out.size()}")
            # print(f"yolo2_out_view - {yolo2_out_view.size()}")
            # print(f"yolo3_out_view - {yolo3_out_view.size()}")
            # print(f"yolo4_out_view - {yolo4_out_view.size()}")
            img_size = x.shape[-2:]
            nb = x.shape[0]  # batch size
            s = [0.83, 0.67]  # scales
            # inference or test
            # x, p = zip(*(yolo4_out_view, yolo3_out_view, yolo2_out_view))  # inference output, training output
            try:
                x = [yolo4_out_view, yolo3_out_view, yolo2_out_view]
                x = torch.cat(x, 1)  # cat yolo outputs
            except Exception:
                pass
            # print(f"inf_out -  {x.shape}")
            # if augment:  # de-augment results
            # x = torch.split(x, nb, dim=0)
            # x[1][..., :4] /= s[0]  # scale
            # x[1][..., 0] = img_size[1] - x[1][..., 0]  # flip lr
            # x[2][..., :4] /= s[1]  # scale
            # print(f"DBG4 {x.shape}")
            # x = torch.cat(x, 1)
            # return torch.squeeze(depth_out, dim=1), (x, p)
            return torch.squeeze(depth_out, dim=1), [x,(yolo4_out, yolo3_out, yolo2_out)]
            
    
    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)