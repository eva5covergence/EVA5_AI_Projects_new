## Sesson 15 - Create a network that can perform 3 tasks simultaneously
1.  Predict the boots, PPE, hardhat, and mask if there is an image
2.  Predict the depth map of the image
3.  Predict the Planar Surfaces in the region

Author: S.A.Ezhirko
**********************************************************************************************************************

Some of the biggest challenges in computer vission is to recognise and identify the objects present in the image, predict the depth between the objects in the image and identify the surface planes in a given image. For many years these task were challenging for researchers because the image seen by computer is just a array of pixel values.

![](Images/Image1.jpg)

After the possibility of creating and executing Deep Neural Networks in modern world, these problems which was challenging are now showing good results. In object detection the computer must identify the object of our interest in the given image and mark the bounding box surrounding the identified image.

<p align="center">
  <img src="Images/Q44.jpg">
  <img src="Images/Q44_BB.jpg">
</p>

In Depth Estimation, the machine has to extract the depth information of the foreground entities from a single image. Example below shows the extracted depth information (on the right) from an image (on the left).

<p align="center">
  <img src="Images/Q44.jpg">
  <img src="Images/Q44_Depth.png">
</p>

In Plane Surface Estimation, the machine has to detect the plane surface information from the depth and mask entities from a single image.
<p align="center">
  <img src="Images/InputImage.jpg">
  <img src="Images/Segmentation_final.png">
</p>

