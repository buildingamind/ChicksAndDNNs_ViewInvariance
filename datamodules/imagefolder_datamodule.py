'''
creates a generic dataloader for non-temporal learning
'''

# IMPORTS
from typing import Any, Callable, Optional, Sequence, Union
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from pl_bolts.datamodules.vision_datamodule import VisionDataModule


class ImageFolderDataModule(VisionDataModule):
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
        dataset_size: int = -1,                     
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
            dataset_size: total number of training samples. If set to -1, then take the entire dataset for training
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
        print("[INFO] Shuffle training samples set to :: ", shuffle)

    """
    _split_dataset is an inbuilt method from VisionDataModule.
    Check -  https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/datamodules/vision_datamodule.py
    
    """
    @property
    def num_samples(self):
        dataset = self.dataset_cls(self.data_dir) # same as dataset = ImageFolder(data_dir)
        print(dataset)
        return len(self._split_dataset(dataset)) # returns num of training samples only
        #return dataset # returns the length of entire dataset i.e 10k

        
    def prepare_data(self, *args, **kwargs):
        dataset = self.dataset_cls(self.data_dir) # same as dataset = ImageFolder(data_dir)
        print(dataset)

    def setup(self, stage=None):
        """
        Creates train, val, and test dataset
        """

        if stage == "fit" or stage is None: # if training
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_cls(
                self.data_dir, transform=train_transforms
            )
            dataset_val = self.dataset_cls(
                self.data_dir, transform=val_transforms
            )
        
                        
            # if using the entire dataset
            if self.dataset_size == -1:
                # Split
                print("[INFO] Using the entire dataset for training the model")
                self.dataset_train = self._split_dataset(dataset_train)
                self.dataset_val = self._split_dataset(dataset_val, train=False)
                
            # using a subset of the dataset
            else:
                print("[INFO] Not using the entire dataset for training the model")
                # last_index takes care of the val_split difference
                last_index = int(self.dataset_size - (self.dataset_size * self.val_split))
                
                train_indices = [i for i in range(0,last_index)]
                val_indices = [j for j in range(last_index,self.dataset_size)]
                
                self.dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
                self.dataset_val = torch.utils.data.Subset(dataset_val, val_indices)

            
            print("[INFO] Total Training Samples :: ",len(self.dataset_train))
            print("[INFO] Total Validation Samples :: ",len(self.dataset_val))

        # this stage will never be called while training a self-supervised backbone in this script
        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, transform=test_transforms
            )

    def default_transforms(self) -> Callable:
        # transforms the image to tensor
        return T.ToTensor()
