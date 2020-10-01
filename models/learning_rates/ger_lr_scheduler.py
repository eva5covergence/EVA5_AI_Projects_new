from configs import basic_config

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_steplr_scheduler(optimizer, **lr_scheduler_paras):
    return StepLR(optimizer, step_size=lr_scheduler_paras['step_size'], gamma=lr_scheduler_paras['gamma'])

def get_reducelronplateau_scheduler(optimizer, **lr_scheduler_paras):
    return ReduceLROnPlateau(optimizer, 
                            mode=lr_scheduler_paras['mode'],
                            factor=lr_scheduler_paras['factor'],
                            patience=lr_scheduler_paras['patience'],
                            threshold=lr_scheduler_paras['threshold'],
                            threshold_mode=lr_scheduler_paras['threshold_mode'],
                            cooldown=lr_scheduler_paras['cooldown'],
                            min_lr=lr_scheduler_paras['min_lr'],
                            eps=1e-08, 
                            verbose=False)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']