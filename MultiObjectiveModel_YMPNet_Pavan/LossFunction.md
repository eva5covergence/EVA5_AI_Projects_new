# Multi objective model Losses:

As our model has 3 objectives, and each objective has it's own loss. For yolov3 and PlanerCNN, we didn't change the loss function, we have used as it is that defined by authors of these networks. But for Midas there no loss or training details provided by author. So we came up with our own loss funciton. In midas, the output is depthmap which is an image and we need to compare with it's ground truth image. There are various techniques to compare the images similarities like Peak to Noise ratio, SSIM, MSE, gradient loss etc... But among them SSIM, Gradient loss, and RMSE worked well for us.

### Bounding box prediction for object detection

Predicting just the bounding box and depth map. The bounding box cordinates were normalized to suit different aspect ratio of the images and 3 anchor boxes where choosen using K mean clustering. The model during training is going to predict the proportion of the bounding boxes and not the absolute bounding box based on the objectness probability. 

### Yolo Loss function
 <p align="center">
  <img src="Images/YoloLoss.jpg">
 </p>
 
 The output image is divided into multipled sections. Each section is a grid which will predict the objectness, probability of the class and its coordinates. The first two equations in the loss equation are responsible to predict the coordinates. Third equation gives the probabiliy of object being present. The fourth equation gives the probability of object not being present. The fifth equation gives the probability of object belonging to specific class or not. For each grid, this equation is run to find the probability of objectness and only if the object is present, the class and the bounding box coordinates are going to be predicted. 
 
 x is the normalized value of center x-coordinate of the bounding box </br>
 y is the normalized value of center y-coordinate of the bounding box </br>
 
 ẋ is the predicted center x-coordinate of the bounding box </br>
 ẏ is the predicted center y-coordinate of the bounding box </br>
 
 w is the normalized value of width of the bounding box </br>
 h is the normalized value of height of the bounding box </br>
 
 ẇ is the predicted width of the bounding box </br>
 ḣ is the predicted height of the bounding box </br>
 
 c is the probability of object being present </br>
 ċ is the predicted probability of object being present </br>
 
 p is the actual class </br>
 ṗ is the predicted class </br>
 
 i is the index of grid in the image </br>
 j is the index of the bounding box </br>
 
Yolov3 instead of sending the class probabilty through Softmax it used Sigmoid function. Non-maximum Suppression technique is used that helps selects the best bounding box among overlapping proposals.


## Depth Loss function

Two loss function SSIM and RMSE are combined to generate the loss for the depth image.

**The Structural Similarity Index (SSIM) Loss**

- SSIM loss will look for similarities within pixels; i.e. if the pixels in the two images line up and or have similar pixel density values.
- **Standardized Values**: SSIM puts everything in a scale of -1 to 1. A score of 1 meant they are very similar and a score of -1 meant they are very different.

**RMSE (Root Mean squared Error) and Gradient loss**

- Compute scale and shift of the pixels of the predicted image
- Calculate the gradient loss with below calculations.
    ```
    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))    
    ```
- Calculate RMSE loss between prediciton and target image
```
sqrt(mse_loss(prediction, target, mask, reduction=self.__reduction))
```
- Take weightage summation of RMSE and Gradient loss
```
    rmse_grad_loss = alpha*grad_loss + rmse_loss # Here we used alpha=0.5
```

**Depth Loss:**

```
    Total_depth_Loss = rmse_grad_loss + SSIM_loss
```



**Sample losses output during the training:**

```
Epoch          gpu_mem             GIoU              obj              cls            total          targets          ImgSize RmseGradMeanLoss    SSIM_meanLoss
   152/299     14.6G             7             2.42             3.91             19.2               25.53               448               0.042            0.152: 100% 346/346 [02:16<00:00,  1.57s/it]
               Class           Images          Targets                P                R          mAP@0.5               F1 RmseGradientLoss         SSIMLoss            DLoss        TotalLoss: 100% 87/87 [00:12<00:00,  1.81it/s]
                 all              692         3.06e+03                0.23                0.46                0.36356                2.736                           16.9              19.636              19.636
```  
 
The code for all these loss functions can be found [here](https://github.com/eva5covergence/EVA5_AI_Projects_new/blob/master/MultiObjectiveModel_YMPNet_Pavan/utils/utils.py).
