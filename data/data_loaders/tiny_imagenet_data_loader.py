import torch
from torchvision import datasets

from utils.misc_utils import set_manual_seed
from data.data_transforms import base_data_transforms
from data.tiny_imagenet import TinyImageNetDataSet
from configs import basic_config

SEED = basic_config.SEED
normalize_paras = basic_config.data['normalize_paras']
img_augs = basic_config.data['img_augs']
data_loader_args = basic_config.data['dataloader_args']
data_kind = basic_config.data['data_kind']
datasets_location = data_kind['datasets_location']
agumentation_package = basic_config.agumentation_package

class TinyImageNetDataLoader:
    def __init__(self, dataset_name='tiny_imagenet',train_split=70):
        """
        This constructor is for initilizing the data loader parameters

        Parameters example:
            data_kind = {"dataset_type":"open_datasets", "dataset_name": "mnist", "train":True}
            transform = BaseDataTransforms(normalize_paras=[(0.0,),(1.0,)], img_augs = {'random_rotation':{angle_range: (-7.0, 7.0), fill=(1,)}}).tranform_data()

        """
        if(agumentation_package == 'Albumentation'):
            self.data_transforms = base_data_transforms.BaseDataTransforms(normalize_paras=normalize_paras, img_augs=img_augs)
            self.train_transforms = self.data_transforms.tranform_albumen_augumentation()
            self.data_transforms = base_data_transforms.BaseDataTransforms(normalize_paras=normalize_paras, img_augs={})
            self.test_transforms = self.data_transforms.tranform_albumen_augumentation()
        else:
            self.data_transforms = base_data_transforms.BaseDataTransforms(normalize_paras=normalize_paras, img_augs=img_augs)
            self.train_transforms = self.data_transforms.tranform_data()
            self.data_transforms = base_data_transforms.BaseDataTransforms(normalize_paras=normalize_paras, img_augs={})
            self.test_transforms = self.data_transforms.tranform_data()
            
        if dataset_name =='tiny_imagenet':
            self.train, self.test, self.classes = TinyImageNetDataSet(train_split = train_split,test_transforms = self.test_transforms,train_transforms = self.train_transforms)

        set_manual_seed(SEED)

    def get_data_loader(self):
        train_loader = torch.utils.data.DataLoader(self.train, **data_loader_args)
        test_loader = torch.utils.data.DataLoader(self.test, **data_loader_args)
        return train_loader,test_loader,self.classes

if __name__ == "__main__":
    train_loader,test_loader, classes = TinyImageNetDataLoader().get_data_loader()
    print(train_loader, test_loader)

