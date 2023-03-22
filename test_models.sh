#!/bin/bash
### -- set the job Name -- 
#BSUB -J Train-SRGAN-ST[1-6]%6

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

let i=$LSB_JOBINDEX
let i--

source .env/bin/activate

# declare -a 
# Get the model path
models_to_test=($(ls /work3/s204163/PW*))

echo $models_to_test

model_weights=/work3/s204163/${models_to_test[i]}/g_best.pth.tar
exp_name=${models_to_test[i]}-TEST

python3 test.py -exp_name=$exp_name -g_model_weights_path=$model_weights

