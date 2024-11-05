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

└── requirements_production.txt: text file containing all the required libraries for this project

└── train_simclr.py: python script initializing SimCLR-CLTT model, dataloader, and trainer to train the model.

└── train_byol.py: python script initializing BYOL model, dataloader, and trainer to train the model.

└── train_barlowTwins.py: python script initializing Barlow Twins model, dataloader, and trainer to train the model.

└── train_vae.py: python script initializing VAE model, dataloader, and trainer to train the model.

└── train_ae.py: python script initializing AE model, dataloader, and trainer to train the model.

├── media: directory containing images and videos for the readme
```

<br>


# Environment Setup

### Step 1: Check python version. 
Your python version should be 3.8.10 and pip version should be 23.2.1 to successfully install all the required libraries. If don't have access to python==3.8.10, then create a conda environment as shown in the next step. 
<br><br>
To check your python version - 

```python
python3 --version
```

To check your pip version - 

```python
pip3 --version or pip --version
```

### Step 2: Creating a virtual Environment.
You need a virtual environment to install all the libraries required to run our models. A virutal environment can be created either using 'conda' or 'venv'. If you system has an older version of python then use conda otherwise use venv.

```python
# Option1: create a virtual environment using conda
conda create --name myenv python=3.8.10 
# replace myenv with your desired env name
```

```python
# Option2: create a virtual environment using venv
python3 -m venv myenv 
# replace myenv with your desired env name
```

### Step 3: Activate virtual environment.
After creating your virtual environment, activate it to install libraries inside that env.

<b>Note:</b> If you used option1 to create virtual env, then activate using - 

```
conda activate myenv
```

<b>Note: </b> If you used option2 to create virtual env, then navigate to the dir where the virtual environment is created and activate using - 

```
source myenv/bin/activate
```

### Step 4: Installing libraries.
First, let's make sure that we have the correct pip version installed - 

```python3
python3 -m pip install --upgrade pip==23.2.1
```

Next, a requirement.txt file is present with this repo which has a list of all the required libraries. Install it using - 

```python3
pip3 install -r requirements_production.txt
```

<b>Note:</b> If any of the libraries fail to install, then recheck your python version as shown in step 1.

<br>


# Experiment I

Note to self: each experiment will have its own table of checkpoints, expriment type and a sh file to replicate the experiment.

# Experiment II

# Experiment III

# Experiment IV
