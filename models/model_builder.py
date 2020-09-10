import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from configs import basic_config
from models.trainer import train
from models.evaluator import test

from utils import logger_utils
logger = logger_utils.get_logger(__name__)

lr = basic_config.optimizer_paras['lr']
momentum = basic_config.optimizer_paras['momentum']
step_size = basic_config.lr_scheduler_steplr_paras['step_size']
gamma = basic_config.lr_scheduler_steplr_paras['gamma']
# weight_decay = basic_config.optimizer_paras['weight_decay']
# l1_lambda = basic_config.l1_lambda

def build_model(EPOCHS, device, train_loader, test_loader, **kwargs):
  train_acc = []
  train_losses = []
  test_acc = []
  test_losses = []
  best_test_accuracy = 0
  best_model = None
  model = kwargs.get('model')
  logger.info(str(kwargs.get('l1_lambda', 0)) + ' ' + str(kwargs.get('l2_lambda', 0)))
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, 
                        weight_decay=kwargs.get('l2_lambda', 0))
  scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
  l1_lambda = kwargs.get('l1_lambda', 0)
  for epoch in range(EPOCHS):
    logger.info(f"[EPOCH:{epoch}]")
    train_acc, train_losses = train(model, device, train_loader, optimizer, l1_lambda, train_acc, train_losses)
    scheduler.step()
    test_acc, test_losses = test(model, device, test_loader, test_acc, test_losses)
    if test_acc[-1] > best_test_accuracy:
      best_test_accuracy = test_acc[-1]
      best_model = model
    logger.info(f"best_test_accuracy {best_test_accuracy}")
    #print(f"best_test_accuracy {best_test_accuracy}")
  return train_acc, train_losses, test_acc, test_losses, best_model
