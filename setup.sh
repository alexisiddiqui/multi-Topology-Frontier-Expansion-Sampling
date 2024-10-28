#!/bin/bash

# Create new conda environment
conda create -n FES python=3.11 ipykernel -y

# Activate the environment
eval "$(conda shell.bash hook)"
source activate FES # may need to change this to 'source activate' if running on HPC

# Install primary dependencies through conda
conda install -n FES -c conda-forge \
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    mdtraj \
    openmm \
    parmed \
    typing \
    pathlib \
    -y


