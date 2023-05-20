#!/bin/bash

### -- set the job Name -- 
#BSUB -J TRAIN-SRGAN-ST[1-1]%1

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --

#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
# -- end of LSF options --

### -- specify queue -- 
#BSUB -q gpua100

### -- ask for number of cores -- 
#BSUB -n 1

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need X GB of memory per core/slot -- 
#BSUB -R "rusage[mem=10GB]"

### -- set walltime limit: hh:mm --
#BSUB -W 24:00

### -- set the email address --
#BSUB -u s204163@student.dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N

nvidia-smi 

export job_index=$((LSB_JOBINDEX-1))

source .env/bin/activate
python3

import os
from config import Config
from train import Train

job_index = os.getenv('jobindex')
print(job_index)
# config = Config()

# Change config variables

train()
# train(config)

# exp_names=("bbgan-sh" "srgan-sh" "gramgan-sh")
# model_names=("bbgan" "srgan" "gramgan")

# num_epochs=2

# job_index=$((LSB_JOBINDEX-1))


# name=${exp_names[$job_index]}
# model=${model_names[$job_index]}


# python train.py -exp_name=$name -model_name=$model -epochs=$num_epochs

# # These steps could be avoided by just saving to the right dir directly tbh
# # Delete the sample directory afterwards
# rm -fr samples/$name
# # Move the results to scratch
# mv /zhome/c9/c/156514/SRGAN-ST/results/$name /work3/s204163/