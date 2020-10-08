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
            if aug == 'rotate':
                rotate_limit = self.img_augs[aug]['rotate_limit']
                self.img_aug_transforms.append(A.Rotate(limit=rotate_limit))
            if aug == 'cutout':
                num_holes=self.img_augs[aug]['num_holes']
                max_h_size=self.img_augs[aug]['max_h_size']
                max_w_size=self.img_augs[aug]['max_w_size']
                fill_value = self.img_augs[aug]['fill_value']
                always_apply=self.img_augs[aug]['always_apply']
                p = self.img_augs[aug]['p']
                self.img_aug_transforms.append(A.Cutout(num_holes=num_holes, max_h_size=max_h_size, max_w_size=max_w_size, 
                fill_value=fill_value,always_apply=always_apply, p=p))

            if aug == 'PadIfNeeded':
                min_height=self.img_augs[aug]['min_height']
                min_width=self.img_augs[aug]['min_width']
                border_mode=self.img_augs[aug]['border_mode']
                value = self.img_augs[aug]['value']
                p = self.img_augs[aug]['p']
                self.img_aug_transforms.append(A.PadIfNeeded(min_height=min_height, min_width=min_width, 
                                                border_mode=border_mode, value=value, p=p))
            if aug == 'oneof_crop':
                randomcrop=self.img_augs[aug]['randomcrop']
                randomcrop_height=randomcrop['height']
                randomcrop_width=randomcrop['width']
                randomcrop_p=randomcrop['p']
                centercrop=self.img_augs[aug]['centercrop']
                centercrop_height=centercrop['height']
                centercrop_width=centercrop['width']
                centercrop_p=centercrop['p']
                p = self.img_augs[aug]['p']
                self.img_aug_transforms.append(A.OneOf([
 				                            A.RandomCrop(height=randomcrop_height, width=randomcrop_width, p=randomcrop_p),
 				                            A.CenterCrop(height=centercrop_height, width=centercrop_width, p=centercrop_p),], 
                                             p=p))
            if aug=='fliplr':
                p = self.img_augs[aug]['p']
                self.img_aug_transforms.append(A.IAAFliplr(p=p))

                

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
