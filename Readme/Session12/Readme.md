## Session 12 - Team Submission
Team Members
1. S.A.Ezhirko
2. Naga Pavan Kumar Kalepu
**********************************************************************************************************************
## **Assignment A - Tiny ImageNet**  [Code](https://github.com/Sushmitha-Katti/EVA-4/blob/master/Session12/S12-AssignmentA/FinalCode.ipynb)


Should train ResNet18  on **Tiny ImageNetData Set** and reach test accuracy of 50%

### **Implementation**
**[Api can be found here](https://github.com/Sushmitha-Katti/PyTNet)**

***No changes are done to previous files. Added [tinyimagenet](https://github.com/Sushmitha-Katti/PyTNet/blob/master/tinyimagenet.py) file for processing of tiny image net data.***

1. Wrote a Code to download, mix train and test , split and convert to the dataset format.
2. Used One Cycle Policy as Scheduler. It yielded better and fast results than others
3. Reached the target accuracy

### **Parameters**

1. Agumentations - Horizontal flip, Padding , Random Crop, Normalisation, Cutout
2. Batch Size - 256
3. Model - Resnet 18 with 200 classes
4. Optimiser - SGD(momentum - 0.9 , weight_decay - 0.0001)
5. Scheduler - One Cycle (  max_lr = 0.02, epochs=30,  pct_start=1/3, anneal_strategy='linear', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=10.0,final_div_factor =10)

### **Results**

1. Best train Accuracy - 74%
2. Best test Accuracy - 57.69%
4. Accuracy Change Graph
