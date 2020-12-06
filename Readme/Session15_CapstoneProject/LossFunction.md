# Loss Function

## Bounding box prediction for object detection

The first stage in building a model was to try out predicting just the bounding box and depth map. The bounding box cordinates were normalized to suit different aspect ratio of the images and 3 anchor boxes where choosen using K mean clustering. The model during training is going to predict the proportion of the bounding boxes and not the absolute bounding box based on the objectness probability. 

### Yolo Loss function
 <p align="center">
  <img src="Images/YoloLoss.jpg">
 </p>

