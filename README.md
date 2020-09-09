# EVA5_AI_Projects
  
#### Project template structure

```
├── .ipynb_checkpoints/
│   └── main-checkpoint.ipynb
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── __init__.py
├── analysis/
│   └── __init__.py
├── check_points/
│   └── __init__.py
├── configs/
│   ├── __init__.py
│   ├── __pycache__/
│   │   ├── __init__.cpython-37.pyc
│   │   └── basic_config.cpython-37.pyc
│   └── basic_config.py
├── data/
│   ├── __init__.py
│   ├── __pycache__/
│   │   └── __init__.cpython-37.pyc
│   ├── data_loaders/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   └── base_data_loader.cpython-37.pyc
│   │   └── base_data_loader.py
│   ├── data_transforms/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   └── base_data_transforms.cpython-37.pyc
│   │   └── base_data_transforms.py
│   └── datasets/
│       ├── __init__.py
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
│   ├── __init__.py
│   ├── __pycache__/
│   │   ├── __init__.cpython-37.pyc
│   │   ├── evaluator.cpython-37.pyc
│   │   ├── model_builder.cpython-37.pyc
│   │   └── trainer.cpython-37.pyc
│   ├── activation_functions/
│   │   └── __init__.py
│   ├── evaluator.py
│   ├── learning_rates/
│   │   └── __init__.py
│   ├── losses/
│   │   └── __init__.py
│   ├── model_builder.py
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   ├── mnist_ghost_bn_se.cpython-37.pyc
│   │   │   └── mnist_normal_bn_se.cpython-37.pyc
│   │   ├── custom_layers/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   │   ├── __init__.cpython-37.pyc
│   │   │   │   └── ghost_batch_norm.cpython-37.pyc
│   │   │   └── ghost_batch_norm.py
│   │   ├── mnist_ghost_bn_se.py
│   │   └── mnist_normal_bn_se.py
│   ├── optimizers/
│   │   └── __init__.py
│   └── trainer.py
├── orchestrators/
│   ├── __init__.py
│   └── session6_assignment.py
├── README.md
├── test_log.txt
├── unit_tests/
│   └── __init__.py
└── utils/
    ├── __init__.py
    ├── __pycache__/
    │   ├── __init__.cpython-37.pyc
    │   ├── logger_utils.cpython-37.pyc
    │   ├── misc_utils.cpython-37.pyc
    │   └── visualization_utils.cpython-37.pyc
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


