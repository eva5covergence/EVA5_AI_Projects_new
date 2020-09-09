import time

from data.data_loaders.base_data_loader import BaseDataLoader
from utils.misc_utils import get_device_type
from models.networks.mnist_ghost_bn_se import GhostNet
from models.networks.mnist_normal_bn_se import Net
from configs import basic_config
from models.model_builder import build_model
from utils.visualization_utils import plot_multigraph, plot_misclassified_images


from utils import logger_utils
logger = logger_utils.get_logger(__name__)

def get_base_model(is_gbn: bool = False): 
  device = get_device_type()
  return Net().to(device) if is_gbn else GhostNet().to(device)

def get_data_loaders():
    train_loader = BaseDataLoader(for_training=True).get_data_loader() # Internally it transforms as well w.r.to configs.basic_config
    test_loader = BaseDataLoader(for_training=False).get_data_loader() # Internally it transforms as well w.r.to configs.basic_config
    return train_loader, test_loader

def start_training(EPOCHS, device, train_loader, test_loader, **models_dict):
    results = {}
    for model_type in models_dict:
        print("Model: ", model_type)
        train_accs, train_losses, test_acc, test_losses, best_model = build_model(EPOCHS, device, train_loader, test_loader, **models_dict[model_type])
        results[model_type] = [train_accs, train_losses, test_acc, test_losses, best_model]
        time.sleep(10)
    return results

def plot_results(lst_plottingJobs, lst_plottingLegends, title):
    plot_multigraph(lst_plottingJobs,lst_plottingLegends,title)

## Configuration

EPOCHS = basic_config.EPOCHS
device = get_device_type()
results = {}

models_dict = {'l1_BN': {'model': get_base_model(), 'l1_lambda':3.994568295536243e-05},
          'l2_BN': {'model': get_base_model(), 'l2_lambda':0.0002871},
          'l1_l2_BN': {'model': get_base_model(), 'l1_lambda':1.4700778484806588e-05, 'l2_lambda':1.4212922008994122e-05},
          'GBN': {'model': get_base_model(is_gbn = True), 'l1_lambda':0, 'l2_lambda':0},
          'l1_l2_GBN': {'model': get_base_model(is_gbn = True), 'l1_lambda':8.283167683030542e-05, 'l2_lambda':9.098971072011508e-05},
          }

lst_plottingJobs_val_acc = []
lst_plottingLegends_val_acc = []
lst_plottingJobs_loss = []
lst_plottingLegends_loss = []

## Training

train_loader, test_loader = get_data_loaders()
results = start_training(EPOCHS, device, train_loader, test_loader, **models_dict)

## Plot the results

for model_type in results:
  lst_plottingJobs_val_acc.append(results[model_type][2])
  lst_plottingLegends_val_acc.append(model_type)
  lst_plottingJobs_loss.append(results[model_type][3])
  lst_plottingLegends_loss.append(model_type)

plot_results(lst_plottingJobs_val_acc,lst_plottingLegends_val_acc,title="Validation accuracy for all jobs")
plot_results(lst_plottingJobs_loss,lst_plottingLegends_loss,title="Loss curves for all jobs")

plot_misclassified_images(results['GBN'][4], device, test_loader, num_of_images = 25)



