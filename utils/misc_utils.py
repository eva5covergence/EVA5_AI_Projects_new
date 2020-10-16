import torch
import sys
from types import ModuleType
from configs import basic_config
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from utils.logger_utils import get_logger

def set_manual_seed(seed):
    logger = get_logger(__name__)
    cuda = is_cuda()
    logger.info(f"CUDA Available? {cuda}")
    torch.cuda.manual_seed(seed) if cuda else torch.manual_seed(seed)


def is_cuda():
    return torch.cuda.is_available()

def get_device_type():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  return device
  
def current_config():
    for item in dir(basic_config):
      if (not item.startswith("_")) and (not isinstance(getattr(basic_config, item), ModuleType)):
        print(f"{item} - {getattr(basic_config, item)}")

def plot_kmeans_clusters_analysis(data, max_clusters=10):
  wcss = []
  for i in range(1, max_clusters+1):
      kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
      kmeans.fit(data)
      wcss.append(kmeans.inertia_)
  plt.plot(range(1, max_clusters+1), wcss)
  plt.title('Elbow Method')
  plt.xlabel('Number of clusters')
  plt.ylabel('WCSS') # Within cluster sum of squares
  plt.show()
  
def scatter_plot(x_data,y_data,title,x_label,y_label):
    plt.title(title)
    plt.scatter(x_data,y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    plt.clf()

if __name__ == "__main__":
    set_manual_seed(20)
    print(is_cuda())
