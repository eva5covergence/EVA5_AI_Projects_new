from torchvision import transforms


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
        
        if self.img_aug_transforms:
            self.transforms_list.extend(self.img_aug_transforms)
        self.transforms_list.append(transforms.ToTensor())
        self.transforms_list.append(transforms.Normalize(means, stds))
        self.transforms_result = transforms.Compose(self.transforms_list)
        return self.transforms_result 
