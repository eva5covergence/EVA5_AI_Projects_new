from torchvision import transforms
from torchvision import datasets

from utils import logger_utils
from data.data_loaders.base_data_loader import BaseDataLoader

logger = logger_utils.get_logger(__name__)


def get_data_loaders(dataset_name=None):
  logger.info("\n**** Started Loading data ****\n")
  train_loader = BaseDataLoader(for_training=True,dataset_name=dataset_name).get_data_loader() # Internally it transforms as well w.r.to configs.basic_config
  test_loader = BaseDataLoader(for_training=False,dataset_name=dataset_name).get_data_loader() # Internally it transforms as well w.r.to configs.basic_config
  logger.info("\n**** Ended Loading data ****\n")
  return train_loader, test_loader
  
def get_data_stats(dataset_name=None, data_set_kind=None, datasets_location='./data'):
    # simple transform
    simple_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       ])
    
    if data_set_kind=='open_datasets':
        if dataset_name=="mnist":
            exp = datasets.MNIST(datasets_location, train=True, download=True, transform=simple_transforms)
        elif dataset_name=="cifar10":
            exp = datasets.CIFAR10(datasets_location, train=True, download=True, transform=simple_transforms)
        exp_data = exp.data
        exp_data = exp.transform(exp_data.numpy())
        
    logger.info('[Train]')
    logger.info(f' - Numpy Shape: {exp.train_data.cpu().numpy().shape}')
    logger.info(f' - Tensor Shape: {exp.train_data.size()}')
    logger.info(f' - min: {torch.min(exp_data)}')
    logger.info(f' - max: {torch.max(exp_data)}')
    logger.info(f' - mean: {torch.mean(exp_data)}')
    logger.info(f' - std: {torch.std(exp_data)}')
    logger.info(f' - var: {torch.var(exp_data)}')