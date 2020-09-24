from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensor
import numpy as np


class BaseDataTransforms:
    """
    This is a class for data tranformations
    """
    def __init__(self, normalize_paras=[(0.0,),(1.0,)], img_augs={}):
        """
        The constructor for initializing the data transformations parameters
        
        Parameters example: 
            normalize_paras=[(0.0,),(1.0,)]
            'img_augs':{'random_rotation':{'angle_range': (-7.0, 7.0), 'fill':(1,)}}
        """
        self.normalize_paras = normalize_paras
        self.img_augs = img_augs

    def tranform_data(self):
        """
        This function is to transform the data using the parameters the initialized in the constructor function.
        """
        self.transforms_list = []
        means, stds =  self.normalize_paras
        self.img_aug_transforms = []

        for img_aug in self.img_augs:
            if img_aug == 'random_rotation':
                angle_range = self.img_augs[img_aug]['angle_range']
                fill = self.img_augs[img_aug]['fill']
                self.img_aug_transforms.append(transforms.RandomRotation(angle_range, fill=fill))
            if img_aug == 'horizontal_flip':
                self.img_aug_transforms.append(transforms.RandomHorizontalFlip())
            if img_aug == 'random_crop':
                size = self.img_augs[img_aug]['size']
                padding = self.img_augs[img_aug]['padding']
                self.img_aug_transforms.append(transforms.RandomCrop(size, padding))
        
        if self.img_aug_transforms:
            self.transforms_list.extend(self.img_aug_transforms)
        self.transforms_list.append(transforms.ToTensor())
        self.transforms_list.append(transforms.Normalize(means, stds))
        self.transforms_result = transforms.Compose(self.transforms_list)
        return self.transforms_result 
    
    def tranform_albumen_augumentation(self):
        """
        This function is to transform the data using the parameters the initialized in the constructor function.
        """
        self.transforms_list = []
        means, stds =  self.normalize_paras
        self.img_aug_transforms = []

        for aug in self.img_augs:
            if aug == 'color_light':
                self.img_aug_transforms.append(A.RandomBrightnessContrast(p=1))
                self.img_aug_transforms.append(A.RandomGamma(p=1))
                self.img_aug_transforms.append(A.CLAHE(p=1))
            if aug == 'color_medium':
                hue_shift = self.img_augs[aug]['hue_shift_limit']
                sat_shift = self.img_augs[aug]['sat_shift_limit']
                val_shift = self.img_augs[aug]['val_shift_limit']
                self.img_aug_transforms.append(A.CLAHE(p=1))
                self.img_aug_transforms.append(A.HueSaturationValue(hue_shift_limit=hue_shift,
                                                                    sat_shift_limit = sat_shift,
                                                                    val_shift_limit = val_shift,p=1))
            if aug == 'color_Large':
                self.img_aug_transforms.append(A.ChannelShuffle(p=1))
                
            if aug == 'CenterCrop':
                height = self.img_augs[aug]['height']
                width = self.img_augs[aug]['width']
                self.img_aug_transforms.append(A.CenterCrop(height,width))
                                             
            if aug == 'RandomSizedCrop':
                height = self.img_augs[aug]['height']
                width = self.img_augs[aug]['width']
                min_max_height = self.img_augs[aug]['min_max_height']
                self.img_aug_transforms.append(A.RandomSizedCrop(min_max_height, height, width))
            
            if aug == 'ShiftScaleRotate':
                shift_limit = self.img_augs[aug]['shift_limit']
                scale_limit = self.img_augs[aug]['scale_limit']
                rotate_limit = self.img_augs[aug]['rotate_limit']
                self.img_aug_transforms.append(A.ShiftScaleRotate(shift_limit, scale_limit, rotate_limit))
                                               
            if aug == 'HorizontalFlip':
                self.img_aug_transforms.append(A.HorizontalFlip(p=0.5))

        if self.img_aug_transforms:
            self.transforms_list.extend(self.img_aug_transforms)
        self.transforms_list.append(A.Normalize(means, stds, max_pixel_value=255.0,p=1.0))
        self.transforms_list.append(ToTensor())
        self.transforms_result = A.Compose(self.transforms_list)
        return lambda img:self.transforms_result(image=np.array(img))["image"] 

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
