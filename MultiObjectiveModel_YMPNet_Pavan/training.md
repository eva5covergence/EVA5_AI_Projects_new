**Training approach:**

1. Trained yolo branch only by freezing all other layers on 64x64 resolution till loss or accuracy metrics get plateaued and observed it got plateaued at 89th epoch. Here I took already trained weights on PPE dataset for 150+ epochs instead of Yolov3 default weights which trained on coco dataset. And midas and planerCNN loss lambda parameters get set to zeros and only yolo loss will be backpropagated to only yolo unfreezed layers and adjust it's weights.
2. With the trained weights from step1, resumed training the yolo branch only by freezing all other layers on 128x128 resolution input image till loss or accuracy metrics get plateaued and observed it got plateaued after around 30 epochs.
3. With the trained weights from step2, resumed training the yolo branch only by freezing all other layers on 256x256 resolution input image till loss or accuracy metrics get plateaued and observed it got plateaued after around 60 epochs.
4. With the trained weights from step3, resumed training the yolo branch only by freezing all other layers on 448x448 resolution input image till loss or accuracy metrics get plateaued and observed it got plateaued after around 15 epochs.
5. With the trained weights from step4, resumed training the yolo branch only by freezing all other layers on 512x512 resolution input image till loss or accuracy metrics get plateaued and observed it got plateaued after around 6 epochs.
6. With the trained weights from step5, resumed training on midas network by freezing the yolo branch with small learning rate 0.0001 as it's already having good weights as loaded with midas pre-trained weights. And it got trained for 11 epochs and observed good results.
7. Trained planerCNN separately by converting our dataset into the Scannet format and trained for 30 epochs.
8. Eventually we got the below predictions.

### Predictions:

**Input:**

<p align="center">
  <img src="Images/TestImage.jpg">
</p>

**Bounding box and class predicitons:**

<p align="center">
  <img src="Images/PImage47.jpg">
</p>

**Depthmap:**

<p align="center">
  <img src="Images/PImage47.png">
</p>

**Surfacemap:**

<p align="center">
  <img src="Images/PImage47_segmentation_final.png">
</p>
