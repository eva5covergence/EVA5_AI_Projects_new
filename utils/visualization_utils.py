import matplotlib.pyplot as plt
import numpy as np

from utils import logger_utils
logger = logger_utils.get_logger(__name__)

def plot_multigraph(lst_jobs,lst_jobsLegends,title, figsize=(10,8)):
  logger.info("\n**** Started Plotting multigraph ****\n")
  plt.figure(figsize=figsize)
  plt.suptitle(title)
  total_jobs = len(lst_jobs)
  total_legends = len(lst_jobsLegends)
  if(total_jobs != total_legends):
    print('The Total Jobs and Legends count are not matching. Cannot plot')
  else:
    count = 0
    while(count < total_jobs):
      plt.plot(lst_jobs[count], label = lst_jobsLegends[count])
      count += 1
    plt.legend()
    plt.show()
    # plt.clf()
  logger.info("\n**** Ended Plotting multigraph ****\n")
  
def plot_LR_graph(lst_loss,lst_lr,title, figsize=(10,8)):
  logger.info("\n**** Started Plotting Graph ****\n")
  plt.figure(figsize=figsize)
  plt.suptitle(title)
  plt.plot(lst_lr, lst_loss)
  plt.show()
  # plt.clf()
  logger.info("\n**** Ended Plotting multigraph ****\n")

def plot_misclassified_images(model, device, test_loader, num_of_images = 25, figsize=(12,12)):
  logger.info("\n**** Started plot_misclassified_images ****\n")
  plt.figure(figsize=figsize)
  plt.suptitle('Misclassifications')
  num_images = 0
  #print(len(test_loader))
  for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      #print(data.shape)
      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      pred_num = pred.cpu().numpy()
      labels = target.cpu().numpy()
      #print('gues',pred_num[0][0])
      #print('lab',labels[0])
      index = 0
      for label, predict in zip(labels, pred_num):
        if label != predict[0]:
          #print('actual:',label)
          #print('pred',predict[0])
          num_images += 1
          p = plt.subplot((num_of_images/5),5,num_images)
          p.imshow(data[index].cpu().numpy().squeeze(),cmap='gray_r')
          p.set_xticks(()); p.set_yticks(()) # remove ticks
          p.set_title(f'Pred: {predict[0]}, Actual: {label}')
        index +=1
        if num_images == num_of_images:
          break
      if num_images == num_of_images:
          break
  logger.info("\n**** Ended plot_misclassified_images ****\n")

def get_clr(iter, lr_min, lr_max, step_size):
  cycle = np.floor(1 + iter / (2 * step_size))
  x = np.abs( iter/step_size - 2 * cycle + 1 )
  lr_t = lr_min + (lr_max - lr_min) * (1 - x)
  return lr_t

def visualize_clr(num_cycles, step_size, lr_min, lr_max):
  total_iters = step_size * 2 * num_cycles
  x = np.linspace(0, total_iters, 1000)
  y = np.array([get_clr(iter, lr_min, lr_max, step_size) for iter in x])
  plt.figure(figsize=(12,6))
  plt.plot(x, y)
  plt.xlabel('Iterations')
  plt.ylabel('Learning Rate')
  plt.grid(b=None)
  plt.show()

if __name__ == "__main__":
    pass
  # 2 x step_size = 1 cycle
#   num_cycles = 3
#   step_size = 20
#   lr_min = 0.1
#   lr_max = 1.0
#   visualize_clr(num_cycles, step_size, lr_min, lr_max)
