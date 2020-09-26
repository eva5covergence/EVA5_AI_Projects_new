# Session 5 Assignment, Group Submission:
#### 1. S.A.Ezhirko, ezhirko.arulmozhi@gmail.com
#### 2. Naga Pavan Kumar Kalepu, nagapavan.kalepu@gmail.com


# Base Model - S5_v0

#### Target:

1. Get the set-up right
2. Set Transforms
3. Set Data Loader
4. Set Basic Working Code
5. Set Basic Training  & Test Loop

#### Results:

1. Parameters: 292164 
2. Best Training Accuracy: 99.73
3. Best Test Accuracy: 99.08

#### Analysis:

1. Model is heavy as capacity is huge
2. Model is over-fitting

# Basic_skeleton - S5_v1

#### Target:

1. Basic Skeleton of Squeeze and excitation model structure

#### Results:

1. Parameters: 13,832
2. Best Train Accuracy: 99.31
3. Best Test Accuracy: 98.68

#### Analysis:
1. Model is little overfitting
2. We need to make to model lighter by reducing the number of parameters


# Lighter_model - S5_v2

#### Target:

1. Make Model Lighter by replacing Big 8x8 kernel with GAP.

#### Results:

1. Parameters: 7,432
2. Best Train Accuracy: 97.48
3. Best Test Accuracy: 97.71

#### Analysis:
1. GAP helped to reduce the number of parameters to 7432 from 13832. Around 6400 parameters got eliminated.
2. Overfitting got reduced as made model simpler with less number of parameters
3. As network capacity got reduced, it is expected that reduction in accuracies

# Batch_Normalization - S5_v3


#### Target:

1. Add Batch Normalization in each conv block except after the last layer

#### Results:

1. Parameters: 7,612
2. Best Train Accuracy: 99.22
3. Best Test Accuracy: 99.05

#### Analysis:
1. Batch Norm helped to push the first few epochs accuracies significantly as previous model without Batch Norm layers first few epochs accuracies is <70% and Batch Norm helped to push first few accuracies >97%
2. Little overfitting
3. Accuracies need to push further to reach target. Let us use LR and LR Schedulers in the next version to resolve it.

# Speed_up_Learning_rates - S5_v4


#### Target:

1. Increased the LR to 0.1 
2. Set the parameters of StepLR to Step_size = 7, Gamma = 0.1

#### Results:

1. Parameters: 7,612
2. Best Train Accuracy: 99.58
3. Best Test Accuracy: 99.35

#### Analysis:
1. Increasing the LR significantly pushed the accuracies further compare to previous model.
2. Little overfitting as most of the epochs train accuracies higher than test accuracies. Let us apply Image augmentation in the next version to regularize the model

# Image Augmentation - S5_v5


#### Target:

1. Added Image augmentation - Random rotation of in the range of -7 to 7 degrees and fill with 1

#### Results:

1. Parameters: 7,612
2. Best Train Accuracy: 99.32
3. Best Test Accuracy: 99.40

#### Analysis:
1. Image augmentation regularized the model as overfitting issue got resolved.
2. Hit the target accuracy 99.40 but yet to make 99.40 consistently atleast in the last few epochs. Let's fine-tune LR or LR Scheduler a bit to get the results consistently in the last few epochs

# FineTune_LR_scheduler - S5_v6

#### Target:

1. FineTune LR scheduler. Set LR=0.1 as before but updated StepSize = 12 and Gamma = 0.2

#### Results:

1. Parameters: 7,612
2. Best Train Accuracy: 99.41
3. Best Test Accuracy: 99.49

#### Analysis:
1. To get best combination values StepSize = 12 and Gamma =0.2, we tried many trails of these two values.
2. The intuition behind above values is, we observed the accuracy is gradually increasing till around 10 epochs and getting stall from there. So we would like to update LR around 10-12 epochs.
3. We tried with StepSize and Gamma combinations - (10, 0.1), (11, 0.1), (12, 0.1) But didn't help to get the target accuracy consistently at last few epochs.
4. So we thought to increase the speed a little bit after 10-12 epochs by updating gamma = 0.2 and tried these StepSize and Gamma combinations - (10, 0.2), (11, 0.2), (12, 0.2) And finaally Stepsize=12, Gamma=0.2 gave best consistency of >=99.4% in the last 3 epochs and hit maximum of 99.49% with less than 8000 parameters



