## Session 14 Assignment - Team Submission
Team Members
1. S.A.Ezhirko
2. Naga Pavan Kumar Kalepu
**********************************************************************************************************************
# **Monocular Depth Estimation and Segmentation** 
To obtain a depth information dataset, we need expensive hardwares like LIDAR's or depth Camera and collect the images. Intel have come up with their model that provides us the depth information for a given image. We will be using their model to generate depth information for our dataset.

![](Images/MonocularDepth.png)     

### **:small_orange_diamond: Preparation of Dataset**
1. The dataset that was collected for training yolo model which contains 3590 images of people wearing Hardhat, Vest, Mask and Boots is being used in this assignment.
2. The ["Depth Estimation Repository"](https://github.com/intel-isl/MiDaS) contains pretrained model weight which we have used to convert our PPE image dataset into Depth map images.
3. The model outputs a grey-scale depth map of the given image. 
4. Open CV color map API was used to view the depth information as a color image. 
5. All the predicted depth images will be stored as grey-scale images in order to reduce the number of channels for the model in next assignment.


