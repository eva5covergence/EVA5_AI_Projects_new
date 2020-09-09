# EVA5_AI_Projects
  
#### Project template structure

```
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── __init__.py
├── analysis/
│   └── __init__.py
├── check_points/
│   └── __init__.py
├── configs/
│   └── basic_config.py
├── data/
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
│   └── __init__.py
├── draw_project_dir_tree.py
├── LICENSE
├── logs/
│   └── project1_test.txt
├── main.ipynb
├── main.py
├── models/
│   ├── activation_functions/
│   ├── evaluator.py
│   ├── learning_rates/
│   ├── losses/
│   ├── model_builder.py
│   ├── networks/
│   │   ├── custom_layers/
│   │   │   └── ghost_batch_norm.py
│   │   ├── mnist_ghost_bn_se.py
│   │   └── mnist_normal_bn_se.py
│   ├── optimizers/
│   └── trainer.py
├── orchestrators/
│   ├── __init__.py
│   └── session6_assignment.py
├── README.md
├── test_log.txt
├── unit_tests/
└── utils/
    ├── logger_utils.py
    ├── misc_utils.py
    └── visualization_utils.py
```

#### Execution flow 

```
 - main.ipynb -> main.py -> <orchestrator>.py
 ```

#### Run below line in main.ipynb to execute corresponding orchestrator project

```
%run main.py <orchestrator_name>

Ex: %run main.py session6_assignment
```


