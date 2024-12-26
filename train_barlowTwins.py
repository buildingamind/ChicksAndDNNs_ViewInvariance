
# IMPORTS
from argparse import ArgumentParser
from functools import partial
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.models.self_supervised.simclr import (
        SimCLREvalDataTransform, 
        SimCLRTrainDataTransform
        )
from datamodules import ImageFolderDataModule
from models.archs import resnets

from models.archs.resnet_3b import resnet_3blocks
from models.archs.resnet_2b import resnet_2blocks
from models.archs.resnet_1b import resnet_1block




class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # adapted from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag
        

# using lower dimensions here instead of 8192 (as shown in the paper)
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)
        
        
def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)
    
class BarlowTwins(pl.LightningModule):
    def __init__(
        self,
        num_training_samples=0,
        batch_size=512,
        backbone='resnet_2blocks',
        encoder_out_dim=512,
        lambda_coeff=5e-3,
        z_dim=128,
        learning_rate=1e-4,
        warmup_epochs=10,   
        max_epochs=100,
    ):
        super().__init__()

        self.arch = backbone
        self.encoder = self.init_encoder()
        self.projection_head = ProjectionHead(input_dim=encoder_out_dim, hidden_dim=encoder_out_dim, output_dim=z_dim)
        self.loss_fn = BarlowTwinsLoss(batch_size=batch_size, lambda_coeff=lambda_coeff, z_dim=z_dim)

        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        self.num_training_samples = num_training_samples
        self.train_iters_per_epoch = self.num_training_samples // batch_size


    def init_encoder(self):
        if self.arch.startswith('resnet'):
            # Resnet34, Resnet18
            if self.arch == 'resnet34' or self.arch == 'resnet18':
                resnet = getattr(resnets, self.arch)
                print("[INFO] Resnet Backbone Selected :: ", self.arch)
                encoder = resnet(first_conv=True, maxpool1=True, pretrained=False, return_all_feature_maps=False)
                return encoder
            # Resnet18 - 3blocks
            elif self.arch == 'resnet18_3blocks':
                resnet = resnet_3blocks(pretrained=False)
                print("[INFO] Resnet Backbone Selected :: ", self.arch) 
            # Resnet18 - 2blocks
            elif self.arch == 'resnet18_2blocks':
                resnet = resnet_2blocks(pretrained=False)
                print("[INFO] Resnet Backbone Selected :: ", self.arch)
            # Resnet18 - 1block
            elif self.arch == 'resnet18_1block':
                resnet = resnet_1block(pretrained=False)
                print("[INFO] Resnet Backbone Selected :: ", self.arch)

        else:
            NotImplementedError("Encoder not implemented.")

        return resnet
        
    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        (x1, x2, _), _ = batch

        z1 = self.projection_head(self.encoder(x1))
        z2 = self.projection_head(self.encoder(x2))

        return self.loss_fn(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("train_loss", loss.item(), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # Data
        parser.add_argument("--data_dir", type=str, default=".", help="directory containing dataset")
        parser.add_argument('--num_workers', default=8, type=int)
        # optim
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1.5e-6)
        parser.add_argument('--warmup_epochs', type=float, default=10)
        # Model
        parser.add_argument('--meta_dir', default='.', type=str, help='path to meta.bin for imagenet')

        return parser


def cli_main():

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
        "--backbone",
        type=str,
        choices=['resnet34','resnet18','resnet18_3blocks','resnet18_2blocks','resnet18_1block'],
        help="select backbone"
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
    parser.add_argument(
        "--aug",
        type=bool,
        default=True,
        help="apply augmentations to training samples"
    )


    # model args
    parser = BarlowTwins.add_model_specific_args(parser)
    args = parser.parse_args()
    args.gpus = 1
    args.lars_wrapper = True


    dm = ImageFolderDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        drop_last=False,
        val_split=args.val_split,
        dataset_size=args.dataset_size,
    )

    if args.aug is True:
        dm.train_transforms = SimCLRTrainDataTransform(
            input_height=dm.size()[-1],
        )

        dm.val_transforms = SimCLREvalDataTransform(
            input_height=dm.size()[-1],
        )
    
    pl.seed_everything(args.seed_val)
    model = BarlowTwins(
    num_training_samples=args.dataset_size,
    batch_size=args.batch_size,
    backbone=args.backbone,
    )

    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint]

    logger = TensorBoardLogger("/data/lpandey/LOGS/barlowTwins", name=f"{args.exp_name}")
   
    
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=args.max_epochs,
        logger=logger,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
    )
    
    if args.print_model:
        print(model)
    
    # train model
    trainer.fit(model, datamodule=dm) 
    
if __name__ == '__main__':
    cli_main()
