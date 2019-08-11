# DeepECA

Predicting protein contact residues and secondary structures from its amino acid sequence by using Deep Neural Network.

# Requirements

* Python3
* Python3
* numpy
* pandas
* TensorFlow r1.11.0
* HHblits

We tested our models on Ubuntsu 16.04 .

## Setup 
Downloading pre-trained models and datasets 
Pre-trained models (for prediction) and datasets (for training) are needed.
Please download these files from the link below.
<https://drive.google.com/drive/u/1/folders/1_Ys7dZ2P0uF52nvXZkkMeIv0rbqhUMPj>

## HHblits
You need to set up hhblits for generation of MSA.
Please edit the path to hhblits in run_hhblits_local.py appropriately.

# Prediction
To predict by using the multitask model, simply run the following command.
The results will be stored in ./contact_pred and ./ss_pred directory.
`python pipeline.py`

When you want to use your original fasta file, put all your fasta files into ./dataset/fasta .

# Training
If you want to train models originally, set up the database and run the following command.
It takes 2 or 3 days with NVIDIA TITAN GTX GPU.
`python multitask.py`

