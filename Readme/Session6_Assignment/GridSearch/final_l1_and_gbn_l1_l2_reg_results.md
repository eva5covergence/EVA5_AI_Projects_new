**Best BN l1 regularisation parameter:**

For L1 lambda parameter 3.994568295536243e-05

**Best GBN L1 and L2:**

For L1 lambda parameter 8.283167683030542e-05, For L2 lambda parameter 9.098971072011508e-05

Final L1 Regularization results:
--------------------------------
```
===================> L1 - Results of Coarse/finer grid search in various ranges - [[0, 0.0001], [0, 0.001], [0, 0.01], [0, 0.1]]<===================

INFO:root:L1 reg parameter: 6.591861839385862e-05, L2 reg parameter: 0, Train_acc: 99.36833333333334, Test_acc: 99.38
INFO:root:L1 reg parameter: 0.0006379964557023101, L2 reg parameter: 0, Train_acc: 98.41166666666666, Test_acc: 98.55
INFO:root:L1 reg parameter: 0.005385266624081285, L2 reg parameter: 0, Train_acc: 86.74333333333334, Test_acc: 88.7
INFO:root:L1 reg parameter: 0.025199167140147163, L2 reg parameter: 0, Train_acc: 16.595, Test_acc: 16.72
For L1 lambda parameter 3.994568295536243e-05
```

**Below 3 values are best among all experiments:**

```
- For L1 lambda parameter 4.9431672827224384e-05 Best train Accuracy 99.36666666666666% and Best Test Accuracy 99.36% at Epoch 2 <== highest - good one as no overfit
- For L1 lambda parameter 2.6027274107961796e-05 Best train Accuracy 99.42666666666666% and Best Test Accuracy 99.37% at Epoch 4 <== highest - very little overfit
- For L1 lambda parameter 6.591861839385862e-05, For L2 lambda parameter 0, Best train Accuracy 99.36833333333334% and Best Test Accuracy 99.38% at Epoch 13
```
 

