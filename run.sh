#!/bin/bash
### -- set the job Name -- 
#BSUB -J Train-SRGAN-ST[1-2]%2

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o train_srganst_%J.out
#BSUB -e train_srganst_%J.err
# -- end of LSF options --

### -- specify queue -- 
#BSUB -q gpua100

### -- ask for number of cores -- 
#BSUB -n 1

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=5GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB

### -- set walltime limit: hh:mm --
#BSUB -W 23:00

### -- set the email address --
#BSUB -u s204163@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
#BSUB -Ne

nvidia-smi


# Load the cuda module

experiments=("Weight1" "Weight2")

now=$(date +"%Y-%m-%d-%H:%M")

source .env/bin/activate

module load python3/3.10.7
# module load cuda/11.7

python train_srgan.py -exp_name="$experiments[$LSB_JOBINDEX]-$now"
