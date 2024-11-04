
# IMPORTS
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import (resnet18_decoder,
                                                     resnet18_encoder,
                                                     resnet50_decoder,
                                                     resnet50_encoder)


from models.archs.ae_component_resnet34 import resnet34_encoder, resnet34_decoder
from models.archs.ae_component_3b import resnet18_encoder_3b, resnet18_decoder_3b
from models.archs.ae_component_2b import resnet18_encoder_2b, resnet18_decoder_2b
from models.archs.ae_component_1b import resnet18_encoder_1b, resnet18_decoder_1b


from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn as nn
from torch.nn import functional as F

from datamodules import ImageFolderDataModule
from pytorch_lightning.loggers import TensorBoardLogger



class VAE(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.

    Model is available pretrained on different datasets:

    Example::

        # not pretrained
        vae = VAE()

        # pretrained on cifar10
        vae = VAE.from_pretrained('cifar10-resnet18')

        # pretrained on stl10
        vae = VAE.from_pretrained('stl10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
        'stl10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-stl10/checkpoints/epoch%3D89.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'resnet18',
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        kl_coeff: float = 0.1,
        latent_dim: int = 256,
        lr: float = 1e-4,
        **kwargs
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
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
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
            'resnet18_3b': {                    # resnet18_3blocks (custom)
                'enc': resnet18_encoder_3b,
                'dec': resnet18_decoder_3b,
            },
            'resnet18_2b': {                    # resnet18_2blocks (custom)
                'enc': resnet18_encoder_2b,
                'dec': resnet18_decoder_2b,
            },
            'resnet18_1b': {                    # resnet18_1block (custom)
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

        self.fc_mu = nn.Linear(self.enc_out_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.enc_out_dim, self.latent_dim)

        print("[INFO] Resnet Backbone Selected :: ", enc_type)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

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

        parser.add_argument("--enc_type", type=str, default='resnet18', choices=['resnet50', 'resnet34', 'resnet18', 'resnet18_3b', 'resnet18_2b', 'resnet18_1b'])
        parser.add_argument("--first_conv", action='store_false')
        parser.add_argument("--maxpool1", action='store_false')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
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
        default=-1,
        type=int,
        help="number of training samples. -1 (default)=entire dataset"
    )
    parser.add_argument(
        "--seed_val",
        type=int,
        default=0,
        help="SEED VALUE"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle training samples"
    )
    parser.add_argument(
        "--print_model",
        action="store_true",
        help="display backbone"
    )

    parser = VAE.add_model_specific_args(parser)
    args = parser.parse_args(args)

    dm = ImageFolderDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        drop_last=False,
        val_split=args.val_split,
        dataset_size=args.dataset_size,
    )
    args.input_height = dm.size()[-1]


    pl.seed_everything(args.seed_val)

    model = VAE(**vars(args))

    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint]

    logger = TensorBoardLogger("/data/lpandey/LOGS/vae/", name=f"{args.exp_name}")
   
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.max_epochs,
        logger=logger,
        sync_batchnorm=False,
        callbacks=callbacks,
    )
    
    if args.print_model:
        print(model)
    
    # train model
    trainer.fit(model, datamodule=dm) 
    

if __name__ == "__main__":
    cli_main()
