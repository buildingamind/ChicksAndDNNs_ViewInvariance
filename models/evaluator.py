import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics import Accuracy
from typing import List
from torch.nn import functional as F
from train_ae import AE 
import wandb
import torchvision.transforms as T
from models.common import LinearProbe


class Evaluator(pl.LightningModule):
    """
    Evaluates a self-supervised learning backbone using the standard evaluation protocol of a linear probe.

    Example::

        # pretrained model
        backbone = SimCLR.load_from_checkpoint(PATH, strict=False)

        # dataset + transforms
        dm = ImageFolderDataModule(data_dir='.')

        # finetuner
        evaluator = Evaluator(backbone, in_features=512)

        # train
        trainer = pl.Trainer()
        trainer.fit(evaluator, dm)

        # test
        trainer.test(datamodule=dm)
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        in_features: int = 512,
        max_epochs: int = 100,
        log_table = None,
        dataFrame = None,
        dropout: float = 0.,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = 'cosine',
        decay_epochs: List = [60, 80],
        gamma: float = 0.1,
        final_lr: float = 0.0,
        finetune: bool = False,
        is_videoMAE: bool = False
        
    ):
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
        """
        
        """
        the input image of 64X64 is reduced by the encoder to 512, thats why in_features = 512
        """
        super().__init__()
        if backbone is None: # debug
            raise ValueError("The 'backbone' parameter cannot be None.")
        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.final_lr = final_lr
        self.finetune = finetune
        self.backbone = backbone
        self.log_table = log_table # wandB log table
        self.dataFrame = dataFrame # CSV log table
        self.is_videoMAE = is_videoMAE
        self.linear_probe = LinearProbe(input_dim=in_features, dropout=dropout)
 
        # Determine whether to finetune the weights in backbone.
        # backbone.parameters() gets all the hyper parameters of the backbone.
        for param in self.backbone.parameters():
            param.requires_grad = self.finetune 

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)
       
    
    def load_pretrained(self,url):
        return self.load_from_checkpoint(url, strict=False)
        
    # the weights are frozen before the first epoch.
    def on_train_epoch_start(self) -> None:
        if self.finetune:
            self.backbone.train()
        else:
            self.backbone.eval()


    def training_step(self, batch, batch_idx):
        loss, probs, y = self.shared_step(batch)
        acc = self.train_acc(probs, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, y = self.shared_step(batch)
        self.val_acc(probs, y)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc,prog_bar=True) 

        return loss
        
        

    """
    The test loop is similar to the validation loop.
    The only difference is that the test loop is only called when trainer.test() is used
    """
    
    """
    probs - between 0 and 1 probabilities
    loss - 1D tensor value between [0,1]
    y - actual label (0,1)
    """
    
    def test_step(self, batch, batch_idx):
        loss, probs, y = self.shared_step(batch) # this loss is the mean of each mini-batch
        
        #//////////////////////////////////// DATA LOGGING ///////////////////////////////////
        if not self.is_videoMAE: # no data logging for VideoMAE for now
            imgs, labels, probabilities, indiv_loss, paths = self.custom_forward(batch)

            for img, label, probability, los, path in zip(imgs, labels, probabilities, indiv_loss, paths):
                #print(img.shape)  [3,64,64] - [C,H,W]
                
                # get predicted label and confidence score
                if probability>=0.5:
                    predicted_label = 1
                    confidence = probability
                else:
                    predicted_label = 0
                    confidence = (1-probability)
                    
                # get viewpoint from image path
                viewpoint = path.split('/')[-2][:-2] # returns V10 or V1 as viewpoint
                
                # log in wwandb table
                self.log_table.add_data(wandb.Image(img), label, predicted_label, probability, confidence, los, path, viewpoint)
            
            # insert in dataframe
            # push tensors to cpu and convert tensors to numpy array
            # label_cpu = label.cpu().numpy()
            # confidence_cpu = confidence.cpu().numpy()
            # probability_cpu = probability.cpu().numpy()
            # los_cpu = los.cpu().numpy()
            
            #self.dataFrame.loc[self.dataFrame.shape[0]] = [label_cpu, predicted_label, probability_cpu, confidence_cpu, los_cpu, path, viewpoint]

        #////////////////////////////////////// END OF DATA LOGGING ///////////////////////////////
        
        self.test_acc(probs, y)
        
        self.log('test_loss', loss, sync_dist=True,prog_bar=True)
        self.log('test_acc', self.test_acc,prog_bar=True)
        

        return loss
    
    
    def custom_forward(self,batch):
        # return the list items and use them in the evaluate file
        if len(batch) == 2: # case for videomae model. No support for path yet
            x, y = batch
        else:
            x, y, path = batch
#       Get model prediction
        if self.is_videoMAE:
            outputs = self.backbone(x) # VideoMAE_encoder
            sequence_output = outputs[0] # extract logits

            feats = self.fc_norm(sequence_output.mean(1)) # take mean accross the first dim which is seq_len
            # torch.mean - Apply average pooling to reduce the temporal dimension
            # feats = torch.mean(feats, dim=2, keepdim=True).squeeze(2)  # Shape: from [128, 512, 384] -> [128, 512], where 512 is channels or feature dims, 384 is the temporal dims as per chatGPT
        else:
            feats = self.backbone(x)
            feats = feats.view(feats.size(0), -1) # does not change the shape or size of the tensor
        logits = self.linear_probe(feats).squeeze() # squeeze() removes a dimension, tensor[[]] -> tensor[]
        probs = torch.sigmoid(logits) # transforms the logits to the range [0,1] which is suited better for binary classification.
        
       
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), reduce=False) # this will give loss per sample instead of mean loss of mini-batch
        if len(batch) == 2:
            return x,y,probs,loss
        else:
            return x, y, probs, loss, path

    
    def shared_step(self, batch):
        if len(batch) == 2: # case for videomae model. No support for path yet.
            x, y = batch

        else:
            x, y, path = batch # x- img as as np array or tensors, y-labels in tensors (0,1), path = string

        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1) # does not change the shape or size of the tensor, checked
        logits = self.linear_probe(feats).squeeze() # squeeze() removes a dimension, tensor[[]] -> tensor[]
        probs = torch.sigmoid(logits) # transforms the logits to the range [0,1] which is suited better for binary classification.

        # though probabilites are calculated above, we can use the raw input from the linear layer with the below function

        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        return loss, probs, y


    def configure_optimizers(self):
        # Train entire network if finetune is True.
        if self.finetune:
            params = self.parameters()
        else:
            params = self.linear_probe.parameters()

        optimizer = torch.optim.SGD(
            params,
            lr=self.learning_rate,
            nesterov=self.nesterov,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                self.max_epochs,
                eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]