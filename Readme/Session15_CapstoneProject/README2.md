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

1) Integration of Encoder ResNeXt101 to Decoder-1 (YoloV3)
    
    - Extracted 3 outputs from last 3 blocks of ResNeXt101's last layers from corresponding blocks
    - Pass 3 outputs as input to the YOLO intermediate layers which connects the final output layer of YOLOv3.
    
2) Integration of Encoder ResNeXt101 to Decoder-2 (Midas)

3) Integration of Encoder ResNeXt101 to Decoder-3 (PlanerCNN)



