from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import pandas as pd
from PIL import Image
import math

from utils import logger_utils
from data.data_loaders.base_data_loader import BaseDataLoader
from data.data_loaders.tiny_imagenet_data_loader import TinyImageNetDataLoader
from data.data_transforms.base_data_transforms import UnNormalize
from configs import basic_config

logger = logger_utils.get_logger(__name__)


def get_data_loaders(dataset_name=None):
  logger.info("\n**** Started Loading data ****\n")
  train_loader = BaseDataLoader(for_training=True,dataset_name=dataset_name).get_data_loader() # Internally it transforms as well w.r.to configs.basic_config
  test_loader = BaseDataLoader(for_training=False,dataset_name=dataset_name).get_data_loader() # Internally it transforms as well w.r.to configs.basic_config
  logger.info("\n**** Ended Loading data ****\n")
  return train_loader, test_loader

def get_imagenet_data_loaders(dataset_name=None,train_split=70):
  logger.info("\n**** Started Loading TinyImageNet data ****\n")
  train_loader,test_loader,classes = TinyImageNetDataLoader(dataset_name=dataset_name,train_split=train_split).get_data_loader() # Internally it transforms as well w.r.to configs.basic_config
  logger.info("\n**** Ended Loading TinyImageNet data ****\n")
  return train_loader, test_loader, classes
  
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
    
def sample_data(data_loader, classes):
  dataiter = iter(data_loader)
  images, labels = dataiter.next()
  # show images
  unnorm_image_grid = UnNormalize(*basic_config.data['normalize_paras'])
  unnorm_image_grid = unnorm_image_grid(torchvision.utils.make_grid(images))
  plt.figure(figsize=(10,10))
  plt.imshow(np.transpose(unnorm_image_grid, (1, 2, 0)))
  # print labels
  logger.info(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def imshow(img,c ):
    img = (img * 0.2) + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(7,7))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
    plt.title(c)

def show_train_data(dataset, classes):

	# get some random training images

  dataiter = iter(dataset)
  images, labels = dataiter.next()
  for i in range(10):
    index = [j for j in range(len(labels)) if labels[j] == i]
    imshow(torchvision.utils.make_grid(images[index[0:5]],nrow=5,padding=2,scale_each=True),classes[i])
    
def process_bbox_data(jsonfile_path, images_dir_path, data_format='json'):
  if data_format=='json':
    bbox_data = pd.read_json(jsonfile_path)
    # bbox_data.transpose().head()
    bbox_data = bbox_data.transpose().reset_index()[['filename','size', 'regions']]
    # bbox_data.shape
    # bbox_data.head()
    final_data = pd.DataFrame(columns=['img_name','img_width','img_height','object_name','x','y','cx','cy',
                                      'bb_width','bb_height','cx_s_img','cy_s_img','bb_width_s_img','bb_height_s_img'])
    for row in bbox_data.iterrows():
        img_name=row[1]['filename']
        for object_info in row[1]['regions']:
            object_name = object_info['region_attributes']['name']
            x = float(object_info['shape_attributes']['x'])
            y = float(object_info['shape_attributes']['y'])
            bb_width = float(object_info['shape_attributes']['width'])
            bb_height = float(object_info['shape_attributes']['height'])
            cx = x+math.floor(bb_width/2)
            cy = y+math.floor(bb_height/2)
            im = Image.open(images_dir_path+img_name)
            img_width, img_height = [float(dim) for dim in im.size]
            cx_s_img = cx/img_width
            cy_s_img = cy/img_height
            bb_width_s_img = bb_width/img_width
            bb_height_s_img = bb_height/img_height
            final_data = final_data.append(dict(img_name=img_name,img_width=img_width,
                                                img_height=img_height,object_name=object_name,
                                                x=x,y=y,cx=cx,cy=cy,bb_width=bb_width,
                                                bb_height=bb_height,cx_s_img=cx_s_img, cy_s_img=cy_s_img,
                                                bb_width_s_img=bb_width_s_img,bb_height_s_img=bb_height_s_img), 
                                                ignore_index=True)
    logger.info(f"\n\n**** Object class counts in all the images ****\n\n {final_data['object_name'].value_counts()}\n\n")
    return final_data
