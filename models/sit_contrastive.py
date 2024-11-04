# TODO: is the attention filter coming out from the last layer or which layer of transformer??
# TODO: get that code from chatgpt in a safe place.



# LIBRARIES
from argparse import ArgumentParser
from typing import Callable, Optional
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Accuracy
from functools import partial
import warnings
# Pytorch modules
import torch
import torch.nn
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Accuracy

from torch.utils.data import DataLoader, random_split

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule

from torchvision import transforms
from datamodules import ImageFolderDataModule
import math

# Extras
from argparse import Namespace

 # ----------------------------------------------------------------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# REGULARIZATION TECHNIQUE FUNCTION
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

# REGULARIZATION TECHNIQUE CLASS
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# SIMCLR PROJECTION LAYER
class Projection(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128, depth=1):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(self.input_dim, self.hidden_dim))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU())

        if depth == 0:
            self.hidden_dim = self.input_dim

        layers.append(nn.Linear(self.hidden_dim, self.output_dim, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)




'''
This mlp is part of the encoder block. Input -> Attention -> Norm -> MLP. This is not the Classifier MLP head.
This is just an MLP block right after the attention block.
'''
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


'''
Note: The key, query and value matrices are generated and parallel linear operations are inside the attention block itself.
'''
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.att_filter_saved = False

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        '''
        @ is a pytorch operator used to multiply matrices of compatible shapes. key matrix is transposed to make it compatible
        for multiplication with query metrix. self.scales in multiplied to prevent gradients becoming very small
        '''
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        '''
        The purpose of applying dropout to the attention scores is to regularize the network during training by randomly setting 
        some of the attention weights to zero. This helps prevent overfitting and improves the generalization
        capability of the model.
        '''
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        '''
        change the dimension of the final matrix using a simple linear layer.
        '''
        x = self.proj(x)
        '''
        apply another dropout layer
        '''
        x = self.proj_drop(x)
        '''
        return final output from layer along with attention filter or matrix.
        during training, we are only dealing with x which is taken care by the below class.
        '''
        # print("filter shape - ", attn.shape)
        # if self.att_filter_saved == False:
        #     torch.save(attn, 'attn.pt')
        #     self.att_filter_saved = True
        return x, attn


'''
this is a transformer block/layer which includes - norm layers, attention heads, mlp
'''
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    '''
    return_attention will be initially set to false. After the model is trained, pass this parameter as true after
    creating the class object and you get the attention filter without further processing.
    '''
    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        #print(attn.shape) # [128-batch, 1-num_heads, 65-path+cls, 65-path+cls]
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# create the patches and flatten them as a vector using Conv2D layer.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# WRAPPER CLASS WITH ALL THE FUNCTIONALITIES
'''
depth-> number of transformer blocks
'''
class VisionTransformer(nn.Module):
    def __init__(self, img_size=[64], patch_size=8, in_chans=3, num_classes=512, embed_dim=768, depth=3,
                 num_heads=3, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

       
        '''
        by using nn.Parameter, this creates a tensor with all zeroes but since this is a parameter now, backpropagation will
        act on this and gradients will be updated for this. The same is true for pos_embed also. This parameter can be set as 
        zeroes or can be set as a random tensor depending upon the problem.
        '''
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        dpr - drop path regularization. Drop path regularization is a technique used in deep neural networks
        to randomly drop (i.e., set to zero) connections between layers during training. This can help 
        prevent overfitting and improve the generalization performance of the model. Drop path regularization
        is similar to dropout regularization, but instead of dropping individual neurons, it drops entire
        paths (i.e., connections between layers) through the network.
        '''
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        '''
        initialize the transformer block
        '''
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head - confirmed
        # the number of classes should be the ouput embedding dims, therefore it should be 512
        self.head = nn.Sequential(*[nn.Linear(2*embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, num_classes)]) if num_classes > 0 else nn.Identity()


        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # print important hyperparameters - 
        print("--------hyperparameters-----------")
        print("patch size - ",  patch_size)
        print("output embedding shape - ", num_classes)
        print("number of transformer blocks - ", depth)
        print("number of attention heads per block - ", num_heads)
        print("--------hyperparameters-----------")

    '''
    initialize weights for layers
    '''
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    #TODO: understand this function
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    '''
    this function concats the positional embedding and the cls_token with the flattened vector of image
    '''
    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    '''
    this is the main forward function of vision transformer
    '''
    def forward(self, x):

        '''
        this x contains:
        1. a vector of patches.
        2. cls_token concatenated to the vector.
        3. pos_embedding concatenated to the vector.
        '''
        #print("before - ",x.shape) # [128, 3-channels , 64-width, 64-height]
        x = self.prepare_tokens(x)
        #print("after - ",x.shape) # [128, 65, 768]
        '''
        this x vector is processed sequentially through each transformer block one by one
        '''

        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        
        '''
        this is where we take the output corresponding to the cls token and pass it through the linear layer to get the final embedding.
        '''
        
        # TODO: change it with the paper 2 vit mlp head which is smaller than this.
        return self.head( torch.cat( (x[:, 0], torch.mean(x[:, 1:], dim=1)), dim=1 ) )

# MAIN CLASS TO BUILD MODEL - 
class ViT(pl.LightningModule):
    def __init__(
            self,
            backbone,
            temporal_mode: str = None,
            learning_rate = 1e-3,
            hidden_mlp = 512,
            feat_dim = 128,
            hidden_depth = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = backbone
        self.temporal_mode = temporal_mode
        #self.learning_rate = learning_rate
        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.hidden_depth = hidden_depth
        self.val_acc = Accuracy(compute_on_step=False)
        self.temperature = 0.5 # default from SIMCLR contrastive model

        self.projection = Projection(    # SimCLR projection head
            input_dim=self.hidden_mlp, # 512
            hidden_dim=self.hidden_mlp, # 512
            output_dim=self.feat_dim, # 128
            depth=self.hidden_depth, #1
        )

    def forward(self, x):
        feats = self.transformer(x)
        # TODO: check if the output tensor shape is 512
        return feats
    
    def shared_step(self, batch):
        # push two images together in a temporal window - 

        if self.temporal_mode == '2images':
            # len(batch) = 3 for temporal model, not confirmed for non-temporal
            if len(batch) == 3:
                img1, img2, _ = batch # (img1, img2, index)
            else:
                # final image in tuple is for online eval
                (img1, img2, _), _ = batch

            # get h representations
            h1 = self.transformer(img1)
            h2 = self.transformer(img2)

            # get z representations
            z1 = self.projection(h1)
            z2 = self.projection(h2)

            loss = self.nt_xent_loss(z1, z2, self.temperature)
        
        # push 2+ images in a temporal window - 
        else:
            #print(type(batch)) #<class 'list'>
            #print(len(batch)) #len = 4 for window_size = 3 because 3 images, and 1 index value
            # window_size = 3
            if len(batch) == 4:
                flag = 0
                img1, img2, img3, _ = batch # (img1, img2, img3, index)
                
            # window_size = 4
            else:
                flag = 1
                img1, img2, img3, img4, _ = batch # (img1, img2, img3, img4, index)
                
            # get h representations
            h1 = self.transformer(img1)
            h2 = self.transformer(img2)
            h3 = self.transformer(img3)
                
            if flag == 1:
                h4 = self.transformer(img4)
                z4 = self.projection(h4)
                    

            # get z representations
            z1 = self.projection(h1)
            z2 = self.projection(h2)
            z3 = self.projection(h3)

            # loss between z1 and other neighboring samples
            l1 = self.nt_xent_loss(z1,z2, self.temperature)
            l2 = self.nt_xent_loss(z1,z3, self.temperature)
            if flag == 1:
                l3 = self.nt_xent_loss(z1,z4, self.temperature)
                    
                # gather losses - 
                loss = (l1+l2+l3)
            else:
                # gather losses - 
                loss = (l1+l2)

        return loss
    
    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        
        out_1_dist = out_1
        out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^1 to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss



    def training_step(self, batch, batch_idx):
        #loss = self.step(batch, batch_idx)
        loss = self.shared_step(batch)
        self.log('train_loss', loss, on_epoch=True, on_step=True) # training_loss
        return loss

    def validation_step(self, batch, batch_idx):
        #loss = self.step(batch, batch_idx)
        loss = self.shared_step(batch)
        
        # TODO: log val_acc
        self.log('val_loss', loss, on_step=False, on_epoch=True) # for val_loss
        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        # transform params
        parser.add_argument("--gaussian_blur", action="store_true", help="add gaussian blur")
        parser.add_argument("--jitter_strength", type=float, default=0.5, help="jitter strength")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")
        parser.add_argument("--temperature", default=0.5, type=float, help="temperature parameter in training loss")
        parser.add_argument("--lars_wrapper", action='store_true', help="apple lars wrapper over optimizer used")
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")

        return parser


# pre-defined architectures of ViT - 

# def vit_tiny(patch_size=16, **kwargs):
#     model = VisionTransformer(
#         patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
#         qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

'''
this is the classification head after the encoder.
It is possible that this is a general head which can be reused with any type of CLS_token.
'''
# class CLSHead(nn.Module):
#     def __init__(self, in_dim, bottleneck_dim, nlayers=3, hidden_dim=4096):
#         super().__init__()
#         nlayers = max(nlayers, 1)
#         if nlayers == 1:
#             self.mlp = nn.Linear(in_dim, bottleneck_dim)
#         else:
#             layers = [nn.Linear(in_dim, hidden_dim)]
#             layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.ReLU(inplace=True))
#             for _ in range(nlayers - 2):
#                 layers.append(nn.Linear(hidden_dim, hidden_dim))
#                 layers.append(nn.BatchNorm1d(hidden_dim))
#                 layers.append(nn.GELU())
#             layers.append(nn.Linear(hidden_dim, bottleneck_dim))
#             layers.append(nn.BatchNorm1d(bottleneck_dim, affine=False))
            
#             self.mlp = nn.Sequential(*layers)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.mlp(x)
#         return x