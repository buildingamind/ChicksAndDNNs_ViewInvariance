from argparse import ArgumentParser
import ast
import pytorch_lightning as pl
import torch
import os
import torch.nn as nn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # tf warnings set to silent in terminal

from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models import resnet18, resnet34
from models.archs.resnet_3b import resnet_3blocks
from models.archs.resnet_2b import resnet_2blocks
from models.archs.resnet_1b import resnet_1block
from datamodules import InvariantRecognitionDataModule
from models import Evaluator, SimCLR
from train_ae import AE
from train_byol import BYOL
from train_vae import VAE
from train_barlowTwins import BarlowTwins
from pytorch_lightning.loggers import WandbLogger
import csv
import wandb
import pandas as pd
from models.vit_contrastive import Backbone, LitClassifier, ViTConfigExtended 

# table to log data in wandb dashboard
columns = ["IMAGE", "ACTUAL_LABEL", "PREDICTED_LABEL", "PROBABILITY", "CONFIDENCE", "LOSS", "PATH", "VIEWPOINT"]
log_table = wandb.Table(columns)


# pandas dataframe to log data for each fold of k-fold in a csv file
dataFrame = pd.DataFrame(columns = ["ACTUAL_LABEL ", "PREDICTED_LABEL ", "PROBABILITY ", "CONFIDENCE ", "LOSS ", "PATH ", "VIEWPOINT "])


def cli_main():
    parser = create_argparser()
    args = parser.parse_args()
    count = -1
    
    # TODO: fix this hardcoded path
    LOG_DIR = f"/data/lpandey/LOGS/eval/csv/{args.project_name}.csv"

    write_csv_stats(LOG_DIR,
                    [{'MODEL':args.model, ' EXPERIMENT':args.exp_name, ' FOLD':args.identifier, ' TEST_SET':args.data_dir}])

    
    # Run K fold cross-validation.
    for fold in range(args.num_folds):
        count+=1
        cross_validation(args, fold, count)
    
    if args.save_csv:
        # TODO: fix this hardcoded path
        # push result data to a .csv file
        dataFrame.to_csv(f"/home/lpandey/LOGS/eval/dataframes/{args.project_name}", sep=',')


def cross_validation(args, fold_idx, count):

    # TODO: fix this hardcoded path
    LOG_DIR = f"/data/lpandey/LOGS/eval/csv/{args.project_name}.csv"
    
    
    dm = InvariantRecognitionDataModule(
        data_dir=args.data_dir,
        identifier=args.identifier,
        num_folds=args.num_folds,
        val_fold=fold_idx,
        batch_size=128,
        shuffle=True,
    )
    
    # initialize the selected model
    model = init_model(args)
    if args.model in ['vit', 'untrained_vit']:
        feature_dim = 512
    else:
        feature_dim = get_model_output_size(model, dm.dims)
    
    dm.setup()
    
    evaluator = Evaluator(model, 
                          in_features=feature_dim, 
                          max_epochs=args.max_epochs, 
                          log_table=log_table, 
                          dataFrame=dataFrame, 
                          )
    
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss', filename='{epoch}-'+str(count))
    callbacks = [model_checkpoint]

    logger = WandbLogger(save_dir=f"LOGS/eval/{args.model}", name=args.exp_name, project=f"{args.project_name}", log_model="all")

    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=callbacks
    )

    trainer.fit(evaluator, datamodule=dm)
    # returns a list type
    metric_test = trainer.test(datamodule=dm)
    
    if args.save_csv:
        # save metrics (testa_acc, model_info) in .csv file
        write_csv_stats(LOG_DIR, metric_test)
    

# save metrics to .csv file -
def write_csv_stats(csv_path, metric_test):
    # creates a csv file automatically if none exists
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w') as f: # w - write to file
            csv_writer = csv.writer(f)
            csv_writer.writerow(metric_test)

    with open(csv_path, 'a') as f: # a - append to file
        csv_writer = csv.writer(f)
        csv_writer.writerow(metric_test)



def init_model(args):
    if args.model == 'pixels':
        model = nn.Flatten()
    elif args.model == 'simclr':
        model = SimCLR.load_from_checkpoint(args.model_path)
    elif args.model == 'byol':
        model = BYOL.load_from_checkpoint(args.model_path)
        model = model.online_network.encoder
    elif args.model == 'barlowTwins':
        model = BarlowTwins.load_from_checkpoint(args.model_path)
    elif args.model == 'ae':
        model = AE.load_from_checkpoint(args.model_path).encoder
    elif args.model == 'vae':
        model = VAE.load_from_checkpoint(args.model_path).encoder
    elif args.model == 'supervised':
        model = resnet18(pretrained=True)
        model.fc = nn.Identity()
    elif args.model == 'untrained_r18':
        model = resnet18(pretrained=False)
        model.fc = nn.Identity()
    elif args.model == 'untrained_r34':
        model = resnet34(pretrained=False)
        model.fc = nn.Identity()
    elif args.model == 'untrained_r18_3b':
        model = resnet_3blocks(pretrained=False)
        model.fc = nn.Identity()
        print("[INFO] Model Selected :: Untrained ResNet18 : 3 Blocks")
    elif args.model == 'untrained_r18_2b':
        model = resnet_2blocks(pretrained=False)
        model.fc = nn.Identity()
        print("[INFO] Model Selected :: Untrained ResNet18 : 2 Blocks")
    elif args.model == 'untrained_r18_1b':
        model = resnet_1block(pretrained=False)
        model.fc = nn.Identity()
        print("[INFO] Model Selected :: Untrained ResNet18 : 1 Block")
    elif args.model == 'vit':
        model = LitClassifier.load_from_checkpoint(args.model_path).backbone
        model.fc = nn.Identity()
    elif args.model == 'untrained_vit':
        configuration = ViTConfigExtended()
        configuration.image_size = args.image_size
        configuration.patch_size = args.patch_size
        configuration.num_hidden_layers = args.vit_hidden_layers
        configuration.num_attention_heads = args.vit_attention_heads
        # print configuration parameters of ViT
        print("[INFO] Untrained ViT Selected :::: ")
        print('[INFO] Image Size :: ', configuration.image_size)
        print('[INFO] Patch Size :: ', configuration.patch_size)
        print('[INFO] Number Of Attention Heads :: ', configuration.num_attention_heads)
        print('[INFO] Number Of Layers :: ', configuration.num_hidden_layers)
        
        # pass the configuration parameters to get backbone
        backbone = Backbone('vit', configuration)
        model = LitClassifier(backbone).backbone
        model.fc = nn.Identity()


    return model



def get_model_output_size(model, input_size) -> int:
    """ Returns the output activation size of the encoder. """
    with torch.no_grad():
        if isinstance(input_size, int):
            x = model(torch.zeros(1, input_size))
        else:
            x = model(torch.zeros(1, *input_size))
        return x.view(1, -1).size(1)



def create_argparser():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str, help="directory containing training and test sets")
    parser.add_argument("--num_heads", type=int, help="total number of attention heads in the transformer backbone")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--model", type=str, choices=['pixels', 
                                                      'supervised', 
                                                      'simclr', 
                                                      'untrained_r18', 
                                                      'untrained_r34', 
                                                      'untrained_r18_3b', 
                                                      'untrained_r18_2b', 
                                                      'untrained_r18_1b', 
                                                      'untrained_vit', 
                                                      'ae', 'byol', 'vae', 
                                                      'barlowTwins', 
                                                      'vit',
                                                      ]
                                                      )
    parser.add_argument("--model_path", type=str, help="stored model checkpoint")
    parser.add_argument("--max_epochs", default=100, type=int, help="Max number of epochs to train")
    parser.add_argument("--num_folds", default=12, type=int, choices=[12, 6], help="Number of k-folds")
    parser.add_argument("--identifier", type=str, help="6fold, 6sparse, 12sparse, 12fold")
    parser.add_argument("--project_name", type=str, help="project_name") # for wandb dashboard logging
    parser.add_argument('--image_size', default=64, type=int, help='image resolution')
    parser.add_argument('--patch_size', default=8, type=int, help='size of image patches')
    parser.add_argument('--vit_attention_heads', default=3, type=int, help='num of attention heads in each transformer layer')
    parser.add_argument('--vit_hidden_layers', default=3, type=int, help='number of transformer layers')
    
    return parser


if __name__ == "__main__":
    cli_main()

    #if args.wandb_logging:
    # push table to wandb dashboard
    #wandb.log({"table":log_table})
    # finish wandb operation
    wandb.finish()
    
    print("[FINAL CHECK] ALL DONE")
