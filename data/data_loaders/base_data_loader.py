import torch
from torchvision import datasets

from utils.misc_utils import set_manual_seed
from data.data_transforms import base_data_transforms
from configs import basic_config

SEED = basic_config.SEED
normalize_paras = basic_config.data['normalize_paras']
img_augs = basic_config.data['img_augs']
data_loader_args = basic_config.data['dataloader_args']
data_kind = basic_config.data['data_kind']
datasets_location = data_kind['datasets_location']
agumentation_package = basic_config.agumentation_package

class BaseDataLoader:
    def __init__(self, for_training=True, dataset_name='mnist'):
        """
        This constructor is for initilizing the data loader parameters

        Parameters example:
            data_kind = {"dataset_type":"open_datasets", "dataset_name": "mnist", "train":True}
            transform = BaseDataTransforms(normalize_paras=[(0.0,),(1.0,)], img_augs = {'random_rotation':{angle_range: (-7.0, 7.0), fill=(1,)}}).tranform_data()

        """
        self.for_training=for_training

        if self.for_training:
            self.data_transforms = base_data_transforms.BaseDataTransforms(normalize_paras=normalize_paras, img_augs=img_augs)
        else:
            self.data_transforms = base_data_transforms.BaseDataTransforms(normalize_paras=normalize_paras, img_augs={})
        
        if(agumentation_package == 'Albumentation'):
            self.data_transforms = self.data_transforms.tranform_albumen_augumentation()
        else:
            self.data_transforms = self.data_transforms.tranform_data()

        if data_kind['dataset_type']=='open_datasets':
            if dataset_name =='mnist':
                if self.for_training:
                    self.train = datasets.MNIST(datasets_location, train=True, download=True, transform=self.data_transforms)
                else:
                    self.test = datasets.MNIST(datasets_location, train=False, download=True, transform=self.data_transforms)
            if dataset_name =='cifar10':
                if self.for_training:
                    self.train = datasets.CIFAR10(datasets_location, train=True, download=True, transform=self.data_transforms)
                else:
                    self.test = datasets.CIFAR10(datasets_location, train=False, download=True, transform=self.data_transforms)
        set_manual_seed(SEED)

    def get_data_loader(self):
        if self.for_training:
            return torch.utils.data.DataLoader(self.train, **data_loader_args)
        else:
            return torch.utils.data.DataLoader(self.test, **data_loader_args)

if __name__ == "__main__":
    train_loader = BaseDataLoader(for_training=True).get_data_loader()
    test_loader = BaseDataLoader(for_training=False).get_data_loader()
    print(train_loader, test_loader)

