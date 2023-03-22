#!/bin/bash
### -- set the job Name -- 
#BSUB -J Train-SRGAN-ST[1-6]%1

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

now=$(date +"%Y-%m-%d-%H:%M")

declare -a exp_names=("CW1" "CW2" "CW3" "CW4" "CW5" "CW6")

let i=$LSB_JOBINDEX
let i--

declare -a pixel_weights=(      1.0   1.0   1.0   1.0   1.0   1.0)
declare -a content_weights=(    0.0   0.01  1.0   10.0  100.0 1000.0)
declare -a adversarial_weights=(0.001 0.001 0.001 0.001 0.001 0.001)

source .env/bin/activate

# module load python3/3.10.7
# module load cuda/11.7

name=${exp_names[$i]}-$now

p_weight=${pixel_weights[$i]}
c_weight=${content_weights[$i]}
a_weight=${adversarial_weights[$i]}

python train_srgan.py -exp_name=$name -pixel_weight=$p_weight -content_weight=$c_weight -adversarial_weight=$a_weight





