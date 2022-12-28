#!/bin/bash

# set the experiment GPU by the first parameters
export CUDA_VISIBLE_DEVICES=$1
# activate your experiment environment
source activate STSSL 

# set the dataset by the second parameters
python main.py --config_filename=configs/$2.yaml
