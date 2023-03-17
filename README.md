# SRGAN-ST


## Guide:

* https://th3.gbar.dtu.dk/agent
* ssh gbar1 for terminal

* Husk at moduler allerede liggere i .env
* Moduler: https://www.hpc.dtu.dk/?page_id=282
    - module load python3/3.10.7
    - module load cuda/11.7

* pip install med:
    - python3 -m pip install --user ...

* GPU Nodes: https://www.hpc.dtu.dk/?page_id=2129
    - a100sh virker i hvert fald

* Launch tensorboard fra ssh i vsc:
    - tensorboard --logdir=samples/logs/ --host localhost --port 8888

* Kopier filer til og fra hpc
    - https://www.gbar.dtu.dk/index.php/faq/78-home-directory

how to use venv in notebooks
https://anbasile.github.io/posts/2017-06-25-jupyter-venv/
