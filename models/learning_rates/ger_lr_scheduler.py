from configs import basic_config

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR

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
def get_one_cycle_lr(optimizer, **lr_scheduler_paras):
    return OneCycleLR(
                    optimizer,
                    max_lr=lr_scheduler_paras['max_lr'],
                    epochs=lr_scheduler_paras['epochs'],
                    pct_start=lr_scheduler_paras['pct_start'],
                    steps_per_epoch = lr_scheduler_paras['steps_per_epoch'],
                    cycle_momentum=lr_scheduler_paras['cycle_momentum'],
                    div_factor=lr_scheduler_paras['div_factor'],
                    final_div_factor=lr_scheduler_paras['final_div_factor'],
                    anneal_strategy=lr_scheduler_paras['anneal_strategy'],
                    )

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
