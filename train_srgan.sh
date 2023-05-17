#!/bin/bash
### -- set the job Name -- 
#BSUB -J Train-SRGAN-ST[1-1]%1

### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o train_gramloss_%J.out
#BSUB -e train_gramloss_%J.err
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


# Load the cuda module

now=$(date +"%m-%d-%H")

declare -a exp_names=("bbgan-new-loss-rewrite4")
declare -a model_names=("bbgan")

let epochs=1

let i=$LSB_JOBINDEX
let i--

source .env/bin/activate

# module load python3/3.10.7
# module load cuda/11.7

name=${exp_names[$i]}-$now
model=${model_names[$i]}

python train_srgan.py -exp_name=$name -model_name=$model -epochs=$epochs

# Delete the sample directory afterwards
rm -fr samples/$name

# Move the results to scratch
mv /zhome/c9/c/156514/SRGAN-ST/results/$name /work3/s204163/