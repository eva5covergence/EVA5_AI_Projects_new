# from configs import basic_config
import torch.optim as optim

def get_sgd(model_paras, lr, momentum, weight_decay):
    return optim.SGD(model_paras, lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_adam(model_paras, lr, weight_decay):
    return optim.Adam(model_paras, lr=lr, weight_decay=weight_decay)

def get_rmsprop():
    pass