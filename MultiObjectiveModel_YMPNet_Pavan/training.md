# Training

After finalizing the model architecture, dataset and loss functions, we can now start training the model. We have selected 4 loss functions: **SSIM + (RMSE & Gradient)** and **YoloV3 Loss**. So, we'll train our model with those 4 loss functions.

Each experiment described below had the following common traits
- The model was trained on smaller resolution images first and then gradually the image resolution was increased.
- First Yolo branch was trained by freezing all other layers on 64x64 resolution till loss or accuracy metrics did not varry. Pre-trained weights on PPE dataset for 50 epochs was taken which was trained during Session 13 assignment. Midas loss was set to zeros (lambda = 0) and only yolo loss was backpropagated to only yolo unfreezed layers and adjust it's weights. From the 110 epoch, the loss was stable and the best weights were saved.
- With the best trained weights from above step, the training was resumed on resolution 128 x 128 yolo branch only by freezing all other layers on MIDAS branch for 300 epochs.
- With the best trained weights from above step, the training was performed on subsequent 256x256 and 448x448
- After completely training the yolo branch, MIDAS was unfreezed and yolo branch was freezed. With image size of 448x448 and last best trained weight, MIDAS branch was trained.
- One cycle LR with burnout of 100, batch_size - 8 and lr of 0.01 was used.
- Auto model checkpointing which saved the model weights after every epoch.

The code used for training the model can be found [here](https://github.com/eva5covergence/Ezhirko/blob/main/PPEMultiModel.ipynb)

### Predictions:

**Input:**

<p align="center">
  <img src="Images/Q44.jpg">
</p>

**Bounding box and class predicitons:**

<p align="center">
  <img src="Images/TestImage_BB.jpg">
</p>

**Depthmap:**

<p align="center">
  <img src="Images/TestImageDepth_1.png">
</p>
