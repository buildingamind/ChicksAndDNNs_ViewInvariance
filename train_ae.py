import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import collections.abc as container_abcs
from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import (resnet18_decoder,
                                                     resnet18_encoder,
                                                     resnet50_decoder,
                                                     resnet50_encoder) # these imports are directly from library

# importing resnet-34, resnet18_3blocks, resnet18_2blocks architecture  - 
from models.archs.ae_component_resnet34 import resnet34_encoder, resnet34_decoder
from models.archs.ae_component_3b import resnet18_encoder_3b, resnet18_decoder_3b
from models.archs.ae_component_2b import resnet18_encoder_2b, resnet18_decoder_2b
from models.archs.ae_component_1b import resnet18_encoder_1b, resnet18_decoder_1b
from pytorch_lightning.callbacks import ModelCheckpoint

from torch import nn as nn
from torch.nn import functional as F

from datamodules import ImageFolderDataModule


from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from train_simclr import SimCLR


class AE(pl.LightningModule):
    """
    Standard AE

    Model is available pretrained on different datasets:

    Example::

        # not pretrained
        ae = AE()

        # pretrained on cifar10
        ae = AE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'ae/ae-cifar10/checkpoints/epoch%3D96.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18', 
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs,
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(AE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height

        valid_encoders = {
            'resnet18': {                       # resnet18 (normal 4 blocks)
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50': {                       # resnet50
                'enc': resnet50_encoder,
                'dec': resnet50_decoder,
            },
            'resnet34': {                       # resnet34
                'enc': resnet34_encoder,
                'dec': resnet34_decoder,
            },
            'resnet18_3b': {                    # resnet18_3blocks
                'enc': resnet18_encoder_3b,
                'dec': resnet18_decoder_3b,
            },
            'resnet18_2b': {                    # resnet18_2blocks
                'enc': resnet18_encoder_2b,
                'dec': resnet18_decoder_2b,
            },
            'resnet18_1b': {                    # resnet18_1block
                'enc': resnet18_encoder_1b,
                'dec': resnet18_decoder_1b,
            },
        }


        if enc_type not in valid_encoders:
            self.encoder = resnet18_encoder(first_conv, maxpool1)
            self.decoder = resnet18_decoder(self.latent_dim, self.input_height, first_conv, maxpool1)
        else:
            self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1)
            self.decoder = valid_encoders[enc_type]['dec'](self.latent_dim, self.input_height, first_conv, maxpool1)


        self.fc = nn.Linear(self.enc_out_dim, self.latent_dim)

        print("Architecture selected: ", enc_type)

    @staticmethod
    def pretrained_weights_available():
        return list(AE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in AE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(AE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        return x_hat

    def step(self, batch, batch_idx):
        x, y = batch
        feats = self.encoder(x)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x, reduction='mean')

        return loss, {"loss": loss}

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet18_3b', 'resnet18_2b', 'resnet18_1b','simclr'])
        parser.add_argument("--first_conv", action='store_false')
        parser.add_argument("--maxpool1", action='store_false')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--latent_dim", type=int, default=256)

        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        return parser


def cli_main(args=None):
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
        default=0,
        type=int,
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

    parser = AE.add_model_specific_args(parser)
    args = parser.parse_args(args)

    dm = ImageFolderDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False, # changed from True to False becz of empty dataloader error
        val_split=args.val_split,
        dataset_size=args.dataset_size,
    )
    args.input_height = dm.size()[-1]

    pl.seed_everything(args.seed_val)

    model = AE(**vars(args))

    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint]


    logger = TensorBoardLogger(save_dir="/data/lpandey/LOGS/ae/", name=f"{args.exp_name}")
    
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.max_epochs,
        logger=logger,
        sync_batchnorm=False,
        callbacks=callbacks,
    )

    #print(model)
    trainer.fit(model, datamodule=dm) 
    


if __name__ == "__main__":
    cli_main()

'''
changes for one shot encoder - 
1. monitoring train_loss instead of val_loss.
2. def validation_step() function is in comments to remove validation step.
'''


