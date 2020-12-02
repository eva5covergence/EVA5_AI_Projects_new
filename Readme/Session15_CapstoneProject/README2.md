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
