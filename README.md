<h1 align="center">Parallel development of object recognition in newborn chicks and deep neural networks</h1>


<p align="center"> Lalit Pandey, Donsuk Lee, Samantha M. W. Wood, Justin N. Wood </p>

<p align="center">†S.M.W.W. and J.N.W. contributed equally to this work.</p>


<h2 align="center">Accepted Journal: PLoS Computational Biology, 2024</h2>

<img src='./media/banner.png'>

# Directory Organization

```
ChicksAndDNNs_ViewInvariance
└── datamodules: directory containing python code to set up the datasets and dataloaders to train the model.
│   ├── image_pairs.py - create temporal_window-based dataloader to train the models
│   ├── imagefolder_datamodule.py - a generic data loader script for non-temporal training
│   ├── invariant_recognition.py - script for training and testing the linear classifiers
│
└── models: directory containing python code of different model architecture.
│
└── training_jobs: directory containing bash scripts for training individual models on a single GPU

└── requirements.txt: text file containing all the required libraries for this project

└── train_simclr.py: python script initializing SimCLR-CLTT model, dataloader, and trainer to train the model.

└── train_byol.py: python script initializing BYOL model, dataloader, and trainer to train the model.

└── train_barlowTwins.py: python script initializing Barlow Twins model, dataloader, and trainer to train the model.

└── train_vae.py: python script initializing VAE model, dataloader, and trainer to train the model.

└── train_ae.py: python script initializing AE model, dataloader, and trainer to train the model.

├── media: directory containing images and videos for the readme
```

<br>
