# Determination of Nuclear Position by the Arrangement of Actin Filaments

This repository implements the training and testing for our manuscript "From qualitative data to correlation using deep generative networks: Demonstrating the relation of nuclear position with the arrangement of actin filaments". (DOI: 10.1371/journal.pone.027105)

## Framework

<img src="images/Figure1.png" align="center">

Given one Cytoskeleton image, the proposed model is able to predict the nucleus's positions and sizes.

## Results

<img src="images/Figure3.png" align="center">

Example results of our methods for different Cytoskeleton images. 

# Getting started
## Installation
A suitable conda environment named `TCell` can be created and activated with:

```
conda env create -f environment.yaml
conda activate TCell
```

## Datasets

## Training
- Train a model:
```
sh train.sh
```
-  The hyper-parameters information can be found in options

## Testing
- Test a model
```
sh test.sh
```
- Download the pre-trained models and put them under ```checkpoints/``` directory.
- The default setting should be corresponding to each training parameters in ```train.sh```

