# IMPORTS
from argparse import ArgumentParser
import pytorch_lightning as pl
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform, 
    SimCLRTrainDataTransform
    )
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodules import ImagePairsDataModule
from models.simclr import SimCLR
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision.transforms as T




def create_argparser():
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
        "--image_size",
        default=64,
        type=int,
        help="supported images :: 224X224 and 64X64"
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporally ordered image pairs."
    )
    parser.add_argument(
        "--window_size",
        default=3,
        type=int,
        help="Size of sliding window for sampling temporally ordered image pairs."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--shuffle_frames",
        action="store_true",
        help="shuffle temporal images for training"
    )
    parser.add_argument(
        "--shuffle_temporalWindows",
        action="store_true",
        help="shuffle temporal images for training"
    )
    parser.add_argument(
        "--dataloader_shuffle",
        action="store_true",
        help="shuffle temporal images for training"
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
        choices=['resnet34','resnet18','resnet_3blocks','resnet_2blocks','resnet_1block'],
        help="select architecture"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=-1,
        help="num of training samples to use from dataset. -1 = entire dataset"
    )
    parser.add_argument(
        "--loss_ver",
        type=str,
        choices=['v0','v1'],
        default='v0',
        help="select btw CLTT loss version 0 and loss version 1. Same objectives but different implementations"
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
    
    return parser



def cli_main():
    
    parser = create_argparser()

    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()
    args.gpus = 1
    args.lars_wrapper = True
    args.arch = args.backbone


    if args.temporal:
        dm = ImagePairsDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_frames = args.shuffle_frames,
            shuffle_temporalWindows = args.shuffle_temporalWindows,
            dataloader_shuffle = args.dataloader_shuffle,
            drop_last=False,
            val_split=args.val_split,
            window_size=args.window_size,
            dataset_size=args.dataset_size,
            gpus=args.gpus,
            transform=T.ToTensor(),
        )

    height, width, num_samples = dm.get_info()

    if args.aug is True:
        dm.train_transforms = SimCLRTrainDataTransform(
            input_height=height,
        )
        dm.val_transforms = SimCLREvalDataTransform(
            input_height=height,
        )

    # The SimCLR data transforms are designed to be used with datamodules
    # which return a single image. But ImagePairsDataModule returns
    # a pair of images.
    if isinstance(dm, ImagePairsDataModule):
        dm.train_transforms = dm.train_transforms.train_transform
        dm.val_transforms = dm.val_transforms.train_transform

    pl.seed_everything(args.seed_val)

    args.num_samples = num_samples


    model = SimCLR(**args.__dict__)

    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint]

    logger = TensorBoardLogger("/data/lpandey/LOGS/simclr", name=f"{args.exp_name}")
   
    trainer = pl.Trainer(
        gpus=args.gpus,
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
