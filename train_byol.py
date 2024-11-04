from argparse import ArgumentParser
from copy import deepcopy
from typing import Any

import pytorch_lightning as pl
import torch

from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
#from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from datamodules import ImageFolderDataModule, ImagePairsDataModule
from models.archs import resnets
#from  models.archs import resnet_1block
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from models.archs import resnets
from models.archs import resnet_3b
from models.archs import resnet_2b
from models.archs import resnet_1b
from models.archs.resnet_3b import resnet_3blocks
from models.archs.resnet_2b import resnet_2blocks
from models.archs.resnet_1b import resnet_1block



class MLP(nn.Module):

    def __init__(self, input_dim=512, hidden_size=512, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):

    def __init__(self):
        super().__init__()
        
        # Encoder
        self.encoder = self.init_encoder()
        # Projector
        self.projector = MLP()
        # Predictor
        self.predictor = MLP(input_dim=128, hidden_size=128, output_dim=128)

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        h = self.predictor(z)
        return y, z, h

    def init_encoder(self):
        if arch.startswith('resnet'):
            # Resnet34, Resnet18
            if arch == 'resnet34' or arch == 'resnet18':
                resnet = getattr(resnets, arch)
                print("Architecture selected - ", arch)
            # Resnet18 - 3blocks
            elif arch == 'resnet_3blocks':
                resnet = getattr(resnet_3b, arch)
                print("Architecture selected - Resnet18_3Blocks")
            # Resnet18 - 2blocks
            elif arch == 'resnet_2blocks':
                resnet = getattr(resnet_2b, arch)
                print("Architecture selected - Resnet18_2Blocks")
            # Resnet18 - 1block
            elif arch == 'resnet_1block':
                resnet = getattr(resnet_1b, arch)
                print("Architecture selected - Resnet18_1Block")
            encoder = resnet(first_conv=True, maxpool1=True, return_all_feature_maps=False)
        else:
            NotImplementedError("Encoder not implemented.")

        return encoder

class BYOL(pl.LightningModule):
    """
    PyTorch Lightning implementation of `Bootstrap Your Own Latent (BYOL)
    <https://arxiv.org/pdf/2006.07733.pdf>`_

    Paper authors: Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, \
    Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, \
    Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko.

    Model implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_

    .. warning:: Work in progress. This implementation is still being verified.

    TODOs:
        - verify on CIFAR-10
        - verify on STL-10
        - pre-train on imagenet

    Example::

        model = BYOL(num_classes=10)

        dm = CIFAR10DataModule(num_workers=0)
        dm.train_transforms = SimCLRTrainDataTransform(32)
        dm.val_transforms = SimCLREvalDataTransform(32)

        trainer = pl.Trainer()
        trainer.fit(model, datamodule=dm)

    Train::

        trainer = Trainer()
        trainer.fit(model)

    CLI command::

        # cifar10
        python byol_module.py --gpus 1

        # imagenet
        python byol_module.py
            --gpus 8
            --dataset imagenet2012
            --data_dir /path/to/imagenet/
            --meta_dir /path/to/folder/with/meta.bin/
            --batch_size 32
    """

    def __init__(
        self,
        learning_rate: float = 0.2,
        weight_decay: float = 1.5e-6,
        input_height: int = 32,
        batch_size: int = 32,
        num_workers: int = 0,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        **kwargs
    ):
        """
        Args:
            datamodule: The datamodule
            learning_rate: the learning rate
            weight_decay: optimizer weight decay
            input_height: image input height
            batch_size: the batch size
            num_workers: number of workers
            warmup_epochs: num of epochs for scheduler warm up
            max_epochs: max epochs for scheduler
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.online_network = SiameseArm()
        self.target_network = deepcopy(self.online_network)
        self.weight_callback = BYOLMAWeightUpdate()

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        # Add callback for user automatically since it's key to BYOL weight update
        self.weight_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx, dataloader_idx)

    def forward(self, x):
        y, _, _ = self.online_network(x)
        return y

    def shared_step(self, batch, batch_idx):
        #print(len(batch)) # temporal: list of length 3
        #print(batch_idx)
        if len(batch) == 3:
            img_1, img_2, _ = batch
        else:
            (img_1, img_2, _), _ = batch
        
        # Image 1 to image 2 loss
        y1, z1, h1 = self.online_network(img_1)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_2)
        loss_a = -2 * F.cosine_similarity(h1, z2).mean()

        # Image 2 to image 1 loss
        y1, z1, h1 = self.online_network(img_2)
        with torch.no_grad():
            y2, z2, h2 = self.target_network(img_1)
        # L2 normalize
        loss_b = -2 * F.cosine_similarity(h1, z2).mean()

        # Final loss
        total_loss = loss_a + loss_b

        return loss_a, loss_b, total_loss

    def training_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'1_2_loss': loss_a, '2_1_loss': loss_b, 'train_loss': total_loss})

        return total_loss

    def validation_step(self, batch, batch_idx):
        loss_a, loss_b, total_loss = self.shared_step(batch, batch_idx)

        # log results
        self.log_dict({'1_2_val_loss': loss_a, '2_1_val_loss': loss_b, 'val_loss': total_loss})

        return total_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        optimizer = LARSWrapper(optimizer)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=0.5, help="jitter strength")

        # Data
        parser.add_argument("--data_dir", type=str, default=".", help="directory containing dataset")
        parser.add_argument('--num_workers', default=8, type=int)

        # optim
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1.5e-6)
        parser.add_argument('--warmup_epochs', type=float, default=10)

        # Model
        parser.add_argument('--meta_dir', default='.', type=str, help='path to meta.bin for imagenet')

        return parser

# global variable
arch = 'resnet_2blocks'

def cli_main():
    from pl_bolts.models.self_supervised.simclr import (
        SimCLREvalDataTransform, SimCLRTrainDataTransform)

    # global variable to hold the type of architecture
    global arch

    parser = ArgumentParser()
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Max number of epochs to train."
    )
    parser.add_argument(
        "--val_split",
        default=0.01,
        type=float,
        help="Percent (float) of samples to use for the validation split."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=0,
        help="Subset of dataset"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="wandb dashboard project name"
    )
    parser.add_argument(
        "--seed_val",
        type=int,
        default=0,
        help="SEED VALUE"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=['resnet34','resnet18','resnet_3blocks','resnet_2blocks','resnet_1block'],
        help="select architecture"
    )

    # model args
    parser = BYOL.add_model_specific_args(parser)
    args = parser.parse_args()
    args.gpus = 1
    args.lars_wrapper = True

    arch = args.architecture

    """
        IMPORTANT:
            gaussian_blur is commented below and in arguments of this file,
            jitter_strength is commented below and in arguments of this file
            
            comment them out for non-temporal byol models else result will
            differ from original results
    """
 
    dm = ImageFolderDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=False, # changed from True to False becz of empty dataloader error
            val_split=args.val_split,
            dataset_size=args.dataset_size,
        )
    
    print("shuffle is - ", dm.shuffle)

    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=dm.size()[-1],
        #gaussian_blur=args.gaussian_blur,
        #jitter_strength=args.jitter_strength,
    )

    dm.val_transforms = SimCLREvalDataTransform(
        input_height=dm.size()[-1],
        #gaussian_blur=args.gaussian_blur,
        #jitter_strength=args.jitter_strength,
    )

    pl.seed_everything(args.seed_val)

    model = BYOL(**args.__dict__)

    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint]

    logger = TensorBoardLogger("/data/lpandey/LOGS/byol", name=f"{args.exp_name}")

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.max_epochs,
        logger=logger,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
    )
    
    #print(model)
    trainer.fit(model, datamodule=dm)

if __name__ == '__main__':
    cli_main()
