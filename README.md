# AI Projects framework
  
#### AI Project template structure

```
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── analysis/
├── check_points/
│   └── Session7_assignment_vgg.h5
├── configs/
│   ├── basic_config.py
│   └── basic_config_Session6.py
├── data/
│   ├── base_data_utils.py
│   ├── data_loaders/
│   │   └── base_data_loader.py
│   ├── data_transforms/
│   │   └── base_data_transforms.py
│   └── datasets/
│       └── MNIST/
│           ├── processed/
│           │   ├── test.pt
│           │   └── training.pt
│           └── raw/
│               ├── t10k-images-idx3-ubyte
│               ├── t10k-images-idx3-ubyte.gz
│               ├── t10k-labels-idx1-ubyte
│               ├── t10k-labels-idx1-ubyte.gz
│               ├── train-images-idx3-ubyte
│               ├── train-images-idx3-ubyte.gz
│               ├── train-labels-idx1-ubyte
│               └── train-labels-idx1-ubyte.gz
├── dev.env
├── docs/
├── draw_project_dir_tree.py
├── LICENSE
├── logs/
│   └── project1_test.txt
├── main.py
├── models/
│   ├── activation_functions/
│   ├── base_network_utils.py
│   ├── evaluator.py
│   ├── learning_rates/
│   ├── losses/
│   ├── model_builder.py
│   ├── networks/
│   │   ├── cifar10_dialation_dsc.py
│   │   ├── cifar10_dialation_dsc_vgg.py
│   │   ├── custom_layers/
│   │   │   └── ghost_batch_norm.py
│   │   ├── mnist_ghost_bn_se.py
│   │   └── mnist_normal_bn_se.py
│   ├── optimizers/
│   └── trainer.py
├── orchestrators/
│   ├── __init__.py
│   ├── base_orchestrator.py
│   ├── session6_assignment .ipynb
│   ├── Session7_assignment_V2.ipynb
│   ├── Session7_assignment_V3.ipynb
│   └── Session7_assignment_vgg_final.ipynb
├── README.md
├── requirements.txt
├── setup.py
├── unit_tests/
└── utils/
    ├── logger_utils.py
    ├── misc_utils.py
    └── visualization_utils.py

```

#### Execution flow 


Whenever we need to create a new deep learning project, you need to update/create following files.

1) orchestrators/<orchestrator.py> - Orchestrator is the file which is ipynb file which interact with all other files in the project and make things done - Create/update required for specific project
2) basic_config.py - This is a configuration file which has most of the configuration values needed for the entire project - Create/update required for specific project
3) models/networks/<network_architecture>.py - This is the file where you need to define the DNN (ANN/CNN/RNN/LSTMs/GRUs/GANs) architecture - Create/update required for specific project
4) logs/* - This is where logs of the project are stored.
5) checkpoints/* - This is where models can be saved.
6) And other file_names in above project directory tree are self explanatory based on their file name



#### Deep Learning Framework used:

```
PyTorch
```

