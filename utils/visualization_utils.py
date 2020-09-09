import matplotlib.pyplot as plt

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