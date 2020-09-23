
from utils import logger_utils
from data.data_transforms.base_data_transforms import UnNormalize
from models.gradcam.utils import visualize_cam
from models.gradcam.gradcam import GradCAM
from configs import basic_config

import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid, save_image

logger = logger_utils.get_logger(__name__)

def generate_grad_cam_grid(configs, classes, test_loader, device, model, matched=True):
  for config in configs:
    config['arch'].to(device).eval()
  cams = [
      [cls.from_config(**config) for cls in (GradCAM,)]
      for config in configs
  ]
  images = []
  for gradcam, in cams:
      num_images = 0
      for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        pred_num = pred.cpu().numpy()
        labels = target.cpu().numpy()
        index = 0
        for label, predict in zip(labels, pred_num):
          if matched:
            if label == predict[0]: 
              mask, _ = gradcam(data[index][np.newaxis, :])
              heatmap, result = visualize_cam(mask, data[index][np.newaxis, :])
              images.extend([data[index].cpu(), heatmap,result])
              break
          else:
            if label != predict[0]: 
              mask, _ = gradcam(data[index][np.newaxis, :])
              heatmap, result = visualize_cam(mask, data[index][np.newaxis, :])
              images.extend([data[index].cpu(), heatmap,result])
              break
        break
  grid_image = make_grid(images, nrow=1)
  unnorm_image_grid = UnNormalize(*basic_config.data['normalize_paras'])
  unnorm_image_grid = unnorm_image_grid(torchvision.utils.make_grid(images))
  plt.imshow(np.transpose(unnorm_image_grid, (1, 2, 0)))
  plt.show()
  logger.info(f"Prediction: {classes[predict[0]]}, Actual: {classes[label]}")
