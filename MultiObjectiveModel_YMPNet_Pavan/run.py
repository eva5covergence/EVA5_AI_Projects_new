"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import midas_utils
import cv2

from torchvision.transforms import Compose
# from midas.midas_net import MidasNet
from ymp_net import YMPNet
from transforms import Resize, NormalizeImage, PrepareForNet

from utils.parse_config import parse_model_cfg

cfg = 'cfg/yolov3-custom.cfg'


def run(input_path, output_path, model_path):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.to(device)
    model.eval()

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input

        img = midas_utils.read_image(img_name)
        img_input = transform({"image": img})["image"]
        # print(f"DBG run img.shape - {img.shape}")
        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            prediction,_ = model(sample)
            # print(f"DBG prediction.shape - {prediction.shape}")
            # print(f"prediction.min() - {prediction.min()}")
            # print(f"prediction.max() - {prediction.max()}")
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
            
        
        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        midas_utils.write_depth(filename, prediction, bits=2)

    print("finished")


if __name__ == "__main__":
    # set paths
    INPUT_PATH = "input"
    OUTPUT_PATH = "output"
    # MODEL_PATH = "model.pt"
    MODEL_PATH = "weights/model-f46da743.pt"

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # compute depth maps
    run(INPUT_PATH, OUTPUT_PATH, MODEL_PATH)
