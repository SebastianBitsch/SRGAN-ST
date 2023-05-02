#!/bin/bash
### -- set the job Name -- 
#BSUB -J Train-SRGAN-ST[1-3]%1

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o train_srganst_%J.out
#BSUB -e train_srganst_%J.err
# -- end of LSF options --

### -- specify queue -- 
#BSUB -q gpua100

### -- ask for number of cores -- 
#BSUB -n 1

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=5GB]"

### -- set walltime limit: hh:mm --
#BSUB -W 23:00

### -- set the email address --
#BSUB -u s204163@student.dtu.dk
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
##BSUB -N

nvidia-smi


# Load the cuda module

now=$(date +"%m-%d-%H")

declare -a exp_names=("srgan-30-epochs" "bbgan-30-epochs" "gramgan-30-epochs")
declare -a model_names=("srgan" "bbgan" "gramgan")

let i=$LSB_JOBINDEX
let i--

source .env/bin/activate

# module load python3/3.10.7
# module load cuda/11.7

name=${exp_names[$i]}-$now
model=${model_names[$i]}

python train_srgan.py -exp_name=$name -model_name=$model

# Delete the sample directory afterwards
rm -fr samples/$name

# Move the results to scratch
mv /zhome/c9/c/156514/SRGAN-ST/results/$name /work3/s204163/