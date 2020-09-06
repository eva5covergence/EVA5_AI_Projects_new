# EVA5_AI_Projects
  
#### Project template structure

```
├── orchestrators
|  └── mnist.py
|  └── example.py
|  └── base.py
|
├── configs
|  └── base_config.py
|  └── mnist_exp_0.py
|  └── example_exp_0.py
|
├── data
|  └── datasets
|  |  └── mnist.py
|  |  └── example.py
|  |
|  └── data_loaders
|  |  └── base.py
|  |  └── mnist_data_loader.py
|  |
|  ├── data_transforms
|     └── base.py
|
├── models
|  └── networks
|  |  └── custom_layers
|  |  |  └── ghost_batch_norm.py
|  |  |
|  |  └── vgg16_model.py
|  |  └── s_e_model.py
|  |  └── example.py
|  |
|  └── optimizers
|  |  └── sgd.py
|  |  └── adam.py
|  |
|  └── learning_rates
|  |  └── lr_schedulers.py
|  |
|  └── activation_functions
|  |  └── relu.py
|  |  └── sigmoid.py
|  |  └── tanh.py
|  |
|  └── losses
|  |  └── categorical_entropy.py
|  |  └── negative_log_likelihood.py
|  |  └── mse.py
|  |  └── mae.py
|  └── train.py
|  └── evaluate.py
|  └── predict.py
|  └── get_network.py
|
├── check_points
|  └── mnist_exps
|     └── mnist_exp0_model_epoch25.h5
|
├── docs
|  └── index.html
|  └── demo
|     └── mnist.ipynb
|
├── logs
|  └── mnist_exp0.log
|  └── mnist_exp1.log
|
├── utils
|  └── logger.py
|  └── visualisations.py
|  └── documentation.py
|  └── miscellaneous.py
|
├── analysis
|  └── misclassified_data.py
|
├── unit_tests
|  └── <same as entire project structure which holds corresponding unit tests>
|
├── main.py
|
├── projects
|  └── project1
|     └── project1.ipynb
|  └── project2
|     └── project2.ipynb
```
#### Execution flow 
 - projects/<project_name>.ipynb -> main.py -> orchestrator


