from typing import Any, Callable, Optional, Sequence, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from PIL import Image
import torchvision.transforms as transforms

"""
A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

"""

class ImageFolderDataModule_GIM(VisionDataModule):
    name = "image_folder"
    dataset_cls = ImageFolder # this is an inbuilt pytorch dataset class for datasets that have episode like folders
    dims = (3, 64, 64)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True, 
        pin_memory: bool = False,
        drop_last: bool = False,
        dataset_size: int = 0,                            # dataset size AUTOMATION
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )
        self.dataset_size = dataset_size
        print("shuffle inside datamodule is: ", shuffle)
    """
    _split_dataset is an inbuilt method from VisionDataModule.
    Check -  https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/datamodules/vision_datamodule.py
    
    """
    @property
    def num_samples(self):
        dataset = self.dataset_cls(self.data_dir) # same as dataset = ImageFolder(data_dir)
        print(dataset)
        return len(self._split_dataset(dataset)) # returns the value 9800 for 10k and 98,000 for 100k image dataset, based on val_split i.e. 0.002
        #return dataset # returns the length of entire dataset i.e 10k

        
    def prepare_data(self, *args, **kwargs):
        dataset = self.dataset_cls(self.data_dir) # same as dataset = ImageFolder(data_dir)
        print(dataset)

    def setup(self, stage=None):
        """
        Creates train, val, and test dataset
        """

        if stage == "fit" or stage is None: # if training
            train_transforms, _ = self.default_transforms() if self.train_transforms is None else self.train_transforms
            _, val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_cls(
                self.data_dir, transform=train_transforms
            )
            dataset_val = self.dataset_cls(
                self.data_dir, transform=val_transforms
            )
            
            '''
            import matplotlib.pyplot as plt

            plt.imshow(my_tensor.numpy()[0], cmap='gray')
            '''

            
            # DATASET SIZE AUTOMATION CONDITION
            
            # if no size passed, then take the entire dataset and split between train and val set
            if self.dataset_size is 0:
                # Split
                print("Note: entire dataset is used for training\n")
                self.dataset_train = self._split_dataset(dataset_train)
                self.dataset_val = self._split_dataset(dataset_val, train=False)
                
            # otherwise, take subset of the data
            else:
            
                # last_index takes care of the val_split difference
                last_index = int(self.dataset_size - (self.dataset_size * self.val_split))
                
                train_indices = [i for i in range(0,last_index)]
                val_indices = [j for j in range(last_index,self.dataset_size)]
                
                self.dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
                self.dataset_val = torch.utils.data.Subset(dataset_val, val_indices)

            
            print("size of dataset_train: ",len(self.dataset_train))
            print("size of dataset_val: ",len(self.dataset_val))


        if stage == "test" or stage is None:
            _, val_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, transform=val_transforms
            )

    # ORIGINAL
    # def default_transforms(self) -> Callable:
    #     # transforms the image to tensor
    #     return T.ToTensor()
    

    def get_transforms(self, eval=False, aug=None):
        trans = []

        if aug["randcrop"] and not eval:
            trans.append(transforms.RandomCrop(aug["randcrop"]))

        if aug["randcrop"] and eval:
            trans.append(transforms.CenterCrop(aug["randcrop"]))

        if aug["flip"] and not eval:
            trans.append(transforms.RandomHorizontalFlip())

        if aug["grayscale"]:
            trans.append(transforms.Grayscale())
            trans.append(transforms.ToTensor())
            trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
        elif aug["mean"]:
            trans.append(transforms.ToTensor())
            trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
        else:
            trans.append(transforms.ToTensor())

        trans = transforms.Compose(trans)
        return trans
    
    #@property
    def default_transforms(self) -> Callable:
        """ Return default data transformation. """


        # define the augmentation types
        aug = {
            "stl10": {
                "randcrop": 64,
                "flip": True,
                "grayscale": True, # HARDCODED!!
                "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
                "std": [0.2683, 0.2610, 0.2687],
                "bw_mean": [0.4120],  # values for train+unsupervised combined
                "bw_std": [0.2570],
            }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
        }

        # define train transforms
        transform_train = transforms.Compose(
            [self.get_transforms(eval=False, aug=aug["stl10"])]
        )

        # define val transforms
        transform_valid = transforms.Compose(
            [self.get_transforms(eval=True, aug=aug["stl10"])]
        )

        return transform_train, transform_valid
