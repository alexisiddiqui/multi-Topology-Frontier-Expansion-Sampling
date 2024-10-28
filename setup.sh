#!/bin/bash

# Create new conda environment
conda create -n FES python=3.11 ipykernel -y

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate FES

# Install primary dependencies through conda
conda install -c conda-forge \
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


