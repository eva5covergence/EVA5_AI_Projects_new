
import torch

from utils.parse_config import parse_model_cfg
from ymp_net import YMPNet


def load_model(model_path, device, cfg):
    # select device
    print("device: %s" % device)

    # load network
    all_props = parse_model_cfg(cfg)
    yolo_props = []
    for item in all_props:
        if item['type']=='yolo':
            yolo_props.append(item)
    # print(yolo_props[0])
    model = YMPNet(yolo_props[0]).to(device)
    model.inference=True
    
    if model_path.endswith('.pt'):  # pytorch format
    # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        midas_chkpt = torch.load(model_path, map_location=device)
    
        # load model
        try:
            # yolo_chkpt['model'] = {k: v for k, v in yolo_chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            # model.load_state_dict(yolo_chkpt['model'], strict=False)
            # own_state = load_my_state_dict(chkpt, model)
            # model.load_state_dict(own_state, strict=False)
            
            
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            # midas_weighs_dict = {k: v for k, v in midas_chkpt.items() if k in model_dict}
            for k,v in midas_chkpt['model'].items():
                if k in model_dict:
                    model_dict[k]=v
            # 2. overwrite entries in the existing state dict
            # model_dict.update(midas_weighs_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            print("Loaded Model weights successfully")
        
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.midas_weights, opt.cfg, opt.midas_weights)
            raise KeyError(s) from e
    return model

