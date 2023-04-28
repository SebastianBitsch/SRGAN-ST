# SRGAN-ST


## Guide:

* Thinlinc [link](https://th3.gbar.dtu.dk/agent)
* Check plads
    - ```getquota_zhome.sh```
    - ```getquota_work1.sh```


* Husk at moduler allerede liggere i .env
* Moduler: [link](https://www.hpc.dtu.dk/?page_id=282)
    - ```module load python3/3.10.7```
    - ```module load cuda/11.7```

* pip install med:
    - ```python3 -m pip install --user ...```

* GPU Nodes: [link](https://www.hpc.dtu.dk/?page_id=2129)
    - a100sh virker i hvert fald

* Launch tensorboard fra ssh i vsc:
    - ```tensorboard --logdir=samples/logs/ --host localhost --port 3000```
    - ```fuser -k 3000/tcp``` slå processen ned hvis den allerede kører

* Kopier filer til og fra hpc [link](https://www.gbar.dtu.dk/index.php/faq/78-home-directory)

* Find ud hvor meget space er brugt på hpc
    - ```getquota_zhome.sh```
    - ```getquota_work3.sh```
    - ```du -h --max-depth=1 --apparent $HOME```

* how to use venv in notebooks [link](https://anbasile.github.io/posts/2017-06-25-jupyter-venv/)
* Kopier filer fra hpc til pc
    - ```scp -r -i /Users/sebastianbitsch/.ssh/gbar s204163@transfer.gbar.dtu.dk:SRGAN-ST/samples/logs /Users/sebastianbitsch/Desktop/```
* Scratch ligger på ```/work3/s204163```

## Sources
https://www.hpc.dtu.dk/?page_id=4061
@misc{DTU_DCC_resource,
    author    = {{DTU Computing Center}},
    title     = {{DTU Computing Center resources}},
    year      = {2022},
    publisher = {Technical University of Denmark},
    doi       = {10.48714/DTU.HPC.0001},
    url       = {https://doi.org/10.48714/DTU.HPC.0001},
}
