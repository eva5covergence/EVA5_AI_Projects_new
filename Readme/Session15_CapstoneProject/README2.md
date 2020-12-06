**Data Collection:**

We are solving 3 different problems (Bounding box prediciton, Depthmap, and Planer surfaces detections) through single multi objective model. For object detection we collected images of workers who are wearing vest, hard-hat, mask and boots (3500 images approximately). But this data is not enough to train for depthmap and planer surfaces prediction. So we collected even more data of house interior images from youtube videos. Here the challenge is collecting images should have only interior objects without human beings.

So we filtered the collected images using maskRCNN object predition which trained on coco dataset of 80 classes. If it detects any person object in the image, it will delete that image from collected images.

**Following steps taken to filter out the images:**

- Prepare 5000+ images for Midas and PlanerRCNN:
    - Automated below steps
        - Collected Interior locations related videos youtube links (manual)
        - Installed youtube-dl
        - Installed Detectron2 and dependencies 
        - Downloaded all youtube videos through youtube-dl
        - Extracted the frames for every second
        - Deleted the frames/images which are “having person” or “no objects yet all” by detecting through detectron2 using maskRCNN model
        - Copied all filtered images to target location.      

**Model Building:**

To create multi objective model, we considered 3 different models to merge.

    1) YoloV3
    2) Midas from Intel
    3) PlanerCNN from Nvidia
    
And to solve this kind of complex problem, we need to create encoder-decoder architecture. So we created one-encoder and multi-decoder architecture. Above 3 model network architectures has different backbones except Midas and PlanerCNN has common backbone ResNeXt101 & YoloV3 is having DarkNet-53 as backbone.

**Our high level design approach:**

    A) Encoder - ResNeXt101
    B) Created 3 decoders from corresponding layers of 
        1) YOLOv3 excluding darknet
        2) Refinement & upsampling layers of Midas 
        3) FPN(Feature Pyramid Network), RPN(Region proposal network), Segmentation refinement network & Upsampling layers
        
<p align="center">
  <img src="Images/ympnet1.jpg">
</p>

**Our detailed low level design architecture:**

The inputs and the outputs are of same resolution for depthmaps and surface planes prediction and thus the model has an encoder-decoder architecture, where the model takes one input image  and returns three outputs 

a) Object's Boundingbox & it's class, 
b) Depth Map and 
c) Suface Plane. 

The input image is first processed through encoder block ResNeXt101, which in turn reduces their size to 52x52, 26x26, 13x13 at the last 3 layers. And these layer outputs directly processed through the intermediate layers of yolov3 to get yolov3 final outputs with shapes a) 27x52x52 b) 27x26x26 c) 27x13x13 given the input shapes are 3x416x416. 

And we use the same reduced outputs with shapes 52x52, 26x26 and 13x13 for getting depth and surface plane outputs. To generate depthmap outputs, we pass these outputs through refinement layers of Midas and then upsample using interpolation layers and get the depth map output with shape which is same as input shape 1x416x416. 

To generate surface plane, we pass the same encoder (ResNext101) outputs mentioned above will be passed to FPN (Feature Pyramid network), RPN (Region proposal network), Refinement Network and Upsampling layers of PlanerCNN.

1) Integration of Encoder ResNeXt101 to Decoder-1 (YoloV3)
    
    - Extracted 3 outputs from last 3 blocks of ResNeXt101's last layers from corresponding blocks
    - Pass 3 outputs as input to the YOLO intermediate layers which connects the final output layer of YOLOv3.
    - Yolo produces 3 outputs. Ex: If the input size is 3x416x416 with 4 target classes then outputs of yolov3 of shapes will be a) 27x52x52 b) 27x26x26 c) 27x13x13
    - Re-arrange yolo outputs to a) 3x52x52x9, b) 3x26x26x9, c) 3x13x13x9
    - Filter above outputs through non-max suppression to get the final outputs
    - To filter further, during detection, we can give the inputs a) confidence threshold and b) iou threshold as --conf-thres <value> --iou-thres <value> Ex: --conf-thres 0.1 --iou-thres 0.6
    
2) Integration of Encoder ResNeXt101 to Decoder-2 (Midas)

    - Extracted 3 outputs from last 3 blocks of ResNeXt101's last layers from corresponding blocks
    - Pass above outputs through Midas refinement blocks and refinement outputs.
    - Pass the refinement block outputs through upsampling blocks.
    

3) Integration of Encoder ResNeXt101 to Decoder-3 (PlanerCNN)



