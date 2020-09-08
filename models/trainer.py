import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import logger_utils

logger = logger_utils.get_logger(__name__)

def get_current_train_acc(model, train_loader, device):
  model.eval()
  train_loss = 0
  correct = 0
  with torch.no_grad():
      for data, target in train_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct += pred.eq(target.view_as(pred)).sum().item()
  train_loss /= len(train_loader.dataset)

  logger.info('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      train_loss, correct, len(train_loader.dataset),
      100. * correct / len(train_loader.dataset)))
  
  train_acc = 100. * correct / len(train_loader.dataset)
  return train_acc, train_loss

def train(model, device, train_loader, optimizer, lambda_l1=0, train_acc=[], train_losses=[]):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    #train_losses.append(loss)

    # L1 regularisation

    l1 = 0
    for p in model.parameters():
      l1 += p.abs().sum()
    loss += lambda_l1 * l1

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Current_train_batch_accuracy={100*correct/processed:0.2f}')
  current_train_acc, current_train_loss = get_current_train_acc(model, train_loader)
  train_acc.append(current_train_acc)
  train_losses.append(current_train_loss)
  return train_acc, train_losses