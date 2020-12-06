"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import midas_utils
import cv2



from midas_processing import pre_processing
from midas_processing import post_processing
from utils.model_utils import load_model
cfg = 'cfg/yolov3-custom.cfg'

def write_depth_outs(input_path, output_path, model_path, device):
    print("initialize")
    model = load_model(model_path, device,cfg)
    model.eval()
    image_gen = pre_processing(input_path, output_path, device)
    count=0
    for sample,img, img_name,num_images in image_gen:
        # print(sample.shape, img.shape)
        f=os.path.basename(img_name)
        f=os.path.splitext(f)[0]
        print("  Writing {} ({}/{})".format(output_path+f+".png", count + 1, num_images))
        prediction,(_,_) = model(sample)
        prediction, filename = post_processing(prediction, output_path, img, img_name)
        # print(prediction.shape)
        midas_utils.write_depth(filename, prediction, bits=2)
        count+=1
    print("DONE")









