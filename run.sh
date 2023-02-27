#!/bin/bash
### -- set the job Name -- 
#BSUB -J TCAV
### -- logging files -- 
#BSUB -o tcav_%J.out
#BSUB -e tcav_%J.err
### -- specify queue -- 
#BSUB -q hpc

### -- ask for 1 core -- 
#BSUB -n 10 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB

### -- set walltime limit: hh:mm --
#BSUB -W 12:00

### -- send notification to email --
#BSUB -u s204163@dtu.dk
### -- send notification at start -- 
#BSUB -B
### -- send notification at completion -- 
#BSUB -N
