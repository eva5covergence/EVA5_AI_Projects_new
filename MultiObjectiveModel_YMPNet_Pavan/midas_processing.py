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




def pre_processing(input_path, output_path, device):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
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

    # model.to(device)
    # model.eval()

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    print("start processing")

    for ind, img_name in enumerate(img_names):
        # print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input
        img = midas_utils.read_image(img_name)
        img_input = transform({"image": img})["image"]
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        yield (sample, img, img_name,num_images)
def post_processing(prediction, output_path, img, img_name):
    # create output folder
    os.makedirs(output_path, exist_ok=True)
    prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .detach().numpy()
            )
    # output
    filename = os.path.join(
        output_path, os.path.splitext(os.path.basename(img_name))[0]
    )
    return prediction, filename

