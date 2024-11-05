import os
from typing import Any, Callable, List, Optional, Sequence, Tuple
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import VisionDataset
import torch
from torch.utils.data import Subset
import random
import pytorch_lightning as pl


class ImagePairs(VisionDataset): # Dataset can also be used as a parent class
    """ Creates temporally ordered pairs of images from sequnces of visual observations.
    This class assumes each bottom-most directory has a sequence of images with file names:

        root/.../0.png
        root/.../1.png
        ...
        root/.../t.png

    where t is the last timestep in the sequence.

    Args:
        root: Root directory path.
        window_size: Size of sliding window for sampling pairs. If the window_size
            is 1, each sample will return a pair of identical images. Otherwise,
            the samples will consist of all temporally ordered pairs of images
            within the sliding time window.
    """
    def __init__(
        self,
        root: str,
        window_size: int = 3,
        shuffle_frames: bool = False,
        shuffle_temporalWindows: bool = False,
        transform: Optional[Callable] = None,
        dataset_size: int = -1,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(root,
                         transform=transform,
                         target_transform=None)

        self.root = root
        self.window_size = window_size
        self.shuffle_frames = shuffle_frames
        self.shuffle_temporalWindows = shuffle_temporalWindows
        self.dataset_size = dataset_size
        self.transform = transform

        self.samples = self._make_pairs()
        #print(self.samples)

    def _make_pairs(self) -> List[Tuple[str, str]]:

        pairs = []

        # Sort images numerically in ascending order.
        '''
        v0.1 - support for images with name - 
        'output_x.png', 'x.png' [DISEMBODIED DATASETS] || 
        'babyName_x.jpeg' [HOMEVIEW DATASET] ||
        'train1_x.png' ||
        '''

        fnames = sorted(
            [d.name for d in os.scandir(self.root) if d.is_file()],
            key=lambda x: (
                int(os.path.splitext(x)[0].split('_')[1].split('.')[0]) if 'train' in x else
                (int(os.path.splitext(x)[0].split('_')[1]) if 'output' in x else 
                (int(os.path.splitext(x)[0].split('_')[1]) if '.jpg' in x else 
                int(os.path.splitext(x)[0].split('/')[-1])))
            )
        )



        if self.dataset_size == -1:
            print("[INFO] Using entire dataset for training")
        else:
            # drop samples from the dataset - 
            print("[INFO] Using {} samples from the dataset ".format(self.dataset_size)) 
            fnames = fnames[:self.dataset_size]

        print("[INFO] Total training samples - ", len(fnames))

        if self.shuffle_frames:
            print("[ALERT] Shuffling frames inside temporal windows")
            random.shuffle(fnames)

        '''Temporal Window Examples - 
        [(1,1),(2,2),(3,3),...] if window_size is 1
        [(1,2),(2,3),(3,4),...] if window_size is 2
        [(1,2,3),(2,3,4),(3,4,5),...] if window_size is 3 --> MOST SUITABLE. BY DEFAULT
        [[1,2,3,4],[2,3,4,5],...] if window_size is 4
        '''

        # PUSH 1 OR 2 IMAGES IN A TEMPORAL WINDOW USING A TUPLE
        if self.window_size < 3:
            # Sample pairs with sliding time window.
            for i in range(len(fnames) - self.window_size):               
                prev_path = os.path.join(self.root, fnames[i])    
                next_path = os.path.join(self.root, fnames[i+self.window_size-1])
                pairs.append((prev_path, next_path))   
   
        # PUSH MORE THAN 2 IMAGES IN A TEMPORAL WINDOW AS A LIST
        else:
        # Sample pairs with sliding time window.
            for i in range(0, len(fnames)-self.window_size+1):
                temp = []
                for j in range(i,i+self.window_size):                
                    path = os.path.join(self.root, fnames[j])
                    temp.append(path)

                pairs.append(temp) 

        if self.shuffle_temporalWindows:
            print("[ALERT] Shuffling temporal windows")
            random.shuffle(pairs)

        #print(pairs)
        return pairs


    def load_and_transform_image(self, path):
        img = Image.open(path)  # PIL format
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __getitem__(self, index: int):

        # IF WINDOW SIZE < 3, RETURN A TUPLE OF FRAMES
        if self.window_size < 3:
            prev_path, next_path = self.samples[index]
            prev_img = self.load_and_transform_image(prev_path)
            next_img = self.load_and_transform_image(next_path)

            return prev_img, next_img, index 
        
        # IF WINDOW SIZE > 2, RETURN A LIST OF FRAMES
        else:
            sample_list = self.samples[index]
        
            # transform samples from list
            for i in range(0,len(sample_list)):
                sample_list[i] = Image.open(sample_list[i])
                if self.transform is not None:
                    sample_list[i] = self.transform(sample_list[i])
            
            # current temporal support for 3 and 4 images per window
            # TODO: dynamically handle returns
            if len(sample_list) == 3:
                return sample_list[0], sample_list[1], sample_list[2], index
            else:
                return sample_list[0], sample_list[1], sample_list[2], sample_list[3], index

    def __len__(self) -> int:
        return len(self.samples)


class ImagePairsDataModule(pl.LightningDataModule):
    name = "image_pairs"
    dataset_cls = ImagePairs # this is alias of pl_bolts.datasets.emnist_dataset.BinaryEMNIST
    #dims = (3, 64, 64)
    #dims = (3, 224, 224)

    #print("[INFO] Image resolution set in DataLoader script :: ", dims)

    def __init__(
        self,
        data_dir: str,
        window_size: int = 3,
        dataset_size:int = -1,
        val_split: float = 0.1,
        gpus: int = 1,
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        shuffle_frames = False,
        shuffle_temporalWindows = False,
        dataloader_shuffle = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        transform: str = None,
        *args: Any,
        **kwargs: Any,
    ):
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
        super().__init__()
        self.data_dir=data_dir
        self.val_split=val_split
        self.num_workers=num_workers
        self.batch_size=batch_size
        self.seed=seed
        self.shuffle_frames = shuffle_frames
        self.shuffle_temporalWindows = shuffle_temporalWindows
        self.dataloader_shuffle = dataloader_shuffle
        self.pin_memory=pin_memory
        self.drop_last=drop_last
        
        self.window_size = window_size
        self.EXTRA_ARGS = {"window_size": window_size}
        self.dataset_size = dataset_size
        self.gpus = gpus
        self.transform = transform

    # PTL function
    def prepare_data(self, ):
        # Here you can download or preapre your data if needed
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            dataset = self.dataset_cls(
                root=self.data_dir,
                window_size=self.window_size,
                shuffle_frames = self.shuffle_frames,
                shuffle_temporalWindows = self.shuffle_temporalWindows,
                dataset_size=self.dataset_size,
                transform=self.transform
            )

            # create train-val split
            dataset_size = len(dataset)
            val_size = int(self.val_split * dataset_size)
            train_size = dataset_size - val_size

            train_indices = list(range(0, train_size))
            val_indices = list(range(train_size, dataset_size))
            

            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)

        return self.train_dataset, self.val_dataset

        
    def train_dataloader(self) -> DataLoader:

        # create a custom sampler for distributed training to maintain the temporal sequence of samples
        if self.gpus > 1:
            print("[INFO] Creating a Custom Sampler for Dataloader in DDP")
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, shuffle=self.dataloader_shuffle
            )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.dataloader_shuffle if self.gpus<1 else False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            sampler=sampler if self.gpus>1 else None
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    
    def default_transforms(self) -> Callable:
        return T.ToTensor()