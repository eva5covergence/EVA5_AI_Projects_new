import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from configs import basic_config
from models.trainer import train
from models.evaluator import test

from models.optimizers import get_optimizer
from models.learning_rates import ger_lr_scheduler

from utils import logger_utils

logger = logger_utils.get_logger(__name__)

# lr = basic_config.optimizer_paras['lr']
# momentum = basic_config.optimizer_paras['momentum']
# step_size = basic_config.lr_scheduler_steplr_paras['step_size']
# gamma = basic_config.lr_scheduler_steplr_paras['gamma']
# weight_decay = basic_config.optimizer_paras['weight_decay']
# l1_lambda = basic_config.l1_lambda

# Read optimizer parameters from configuration
def get_optimizer_paras():
  for k,v in basic_config.optimizer_paras.items():
    lr = v.get('lr')
    momentum = v.get('momentum',0)
    weight_decay = v.get('weight_decay')
  return lr, momentum, weight_decay

# Read lr_scheduler parameters from configuration
def get_lr_scheduler_paras():
  use_scheduler = basic_config.lr_scheduler['use_scheduler']
  if use_scheduler:
    if 'ReduceLROnPlateau' in basic_config.lr_scheduler.keys():
      return basic_config.lr_scheduler['ReduceLROnPlateau']
    elif 'stepLR' in basic_config.lr_scheduler.keys():
      return basic_config.lr_scheduler['stepLR']
    elif 'OneCycleLR' in basic_config.lr_scheduler.keys():
      return basic_config.lr_scheduler['OneCycleLR']
  else:
    return use_scheduler

def select_optimizer(model_paras, lr, weight_decay, momentum=0):
  optimizer_type = list(basic_config.optimizer_paras.keys())[0]
  if optimizer_type=='sgd':
    optimizer = get_optimizer.get_sgd(model_paras, lr, momentum, weight_decay)
  elif optimizer_type=='adam':
    optimizer = get_optimizer.get_adam(model_paras, lr, weight_decay)
  return optimizer

def select_lr_scheduler(optimizer, **lr_scheduler_paras):
  if lr_scheduler_paras['name']=='stepLR':
    return ger_lr_scheduler.get_steplr_scheduler(optimizer, **lr_scheduler_paras)
  elif lr_scheduler_paras['name']=='ReduceLROnPlateau':
    return ger_lr_scheduler.get_reducelronplateau_scheduler(optimizer, **lr_scheduler_paras)
  elif lr_scheduler_paras['name']=='OneCycleLR':
    return ger_lr_scheduler.get_one_cycle_lr(optimizer, **lr_scheduler_paras)
  
  

def build_model(EPOCHS, device, train_loader, test_loader, **kwargs):
  train_acc = []
  train_losses = []
  test_acc = []
  test_losses = []
  learning_rates = []
  best_test_accuracy = 0
  iteration=0
  scheduler = None
  best_model = None
  model = kwargs.get('model')
  logger.info(str(kwargs.get('l1_lambda', 0)) + ' ' + str(kwargs.get('l2_lambda', 0)))
  lr, momentum, weight_decay = get_optimizer_paras()
  logger.info(f"Optimizer paras: optimizer={list(basic_config.optimizer_paras.keys())[0]}, lr={lr},momentum={momentum}, weight_decay={weight_decay}")
  # selects the optimizer which is defined in configs
  optimizer = select_optimizer(model.parameters(), lr, weight_decay, momentum) if momentum else select_optimizer(model.parameters(), lr, weight_decay)
  lr_scheduler_paras = get_lr_scheduler_paras()
  if lr_scheduler_paras:
    logger.info(f"LR Scheduler paras: {lr_scheduler_paras}")
    scheduler = select_lr_scheduler(optimizer, **lr_scheduler_paras) ## Selects the scheduler which is defined in configs
  l1_lambda = kwargs.get('l1_lambda', 0)
  for epoch in range(1,EPOCHS+1):
    logger.info(f"[EPOCH:{epoch}]")
    logger.info(f"\nCurrent LR: {ger_lr_scheduler.get_lr(optimizer)}\n")
    learning_rates.append(ger_lr_scheduler.get_lr(optimizer))
    if scheduler and lr_scheduler_paras['name']=='OneCycleLR':
      train_acc, train_losses, scheduler, optimizer, iteration = train(model, device, train_loader, optimizer, l1_lambda, train_acc, train_losses, epoch, scheduler,iteration)
    else:
      train_acc, train_losses = train(model, device, train_loader, optimizer, l1_lambda, train_acc, train_losses, epoch)
    test_acc, test_losses = test(model, device, test_loader, test_acc, test_losses)
    if scheduler:
      if lr_scheduler_paras['name']=='ReduceLROnPlateau':
        scheduler.step(test_losses[-1])
      elif lr_scheduler_paras['name']=='stepLR':
        scheduler.step() 
    if test_acc[-1] > best_test_accuracy:
      best_test_accuracy = test_acc[-1]
      best_model = model
    logger.info(f"best_test_accuracy {best_test_accuracy}")
    #print(f"best_test_accuracy {best_test_accuracy}")
  return train_acc, train_losses, test_acc, test_losses, best_model,learning_rates
