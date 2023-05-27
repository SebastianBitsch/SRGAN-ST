import os
from config import Config
from train import train
from test import test

from loss import BestBuddyLoss

def get_jobindex(fallback:int = None) -> int:
    """Get the job-index set in bash. This is mostly for array jobs where multiple models are trained in parallel"""
    num = os.getenv('job_index')
    return int(num) if num else fallback


def benchmark_experiment(config: Config) -> Config:
    config.EXP.NAME = "plain-dummy"
    return config

def warmup_experiment(config: Config, index:int) -> Config:
    """ Run a warmup experiment where we test how warming up and label smoothing performs """
    config.EXP.NAME = ['warmup-exp', 'labelsmoothing-exp', 'warmup-labelsmoothing-exp'][index]
    config.EXP.LABEL_SMOOTHING = [0, 0.1, 0.1][index]
    config.EXP.N_WARMUP_BATCHES = [5000, 0, 5000][index]
    return config

def srgan_bbgan(config: Config, index:int) -> Config:
    """ stock srgan vs bbgan"""
    config.EXP.NAME = ['plain-srgan-rewrite', 'plain-bbgan-rewrite'][index]
    config.EXP.LABEL_SMOOTHING = 0.1
    config.EXP.N_WARMUP_BATCHES = 5000
    if index == 1:
        config.add_g_criterion("BestBuddy", BestBuddyLoss(), 1.0)
    return config


if __name__ == "__main__":

    # Get job-index from bash, if the job is not an array it will be zero
    job_index = get_jobindex()

    # Edit config based on 
    config = Config()

    # config = warmup_experiment(config, job_index)
    # config.EXP.NAME = "terminal"
    config = srgan_bbgan(config, job_index)
    # config = benchmark_experiment(config)


    print(f"Running job: {job_index}")
    train(config = config)

    # Done mostly just to get some images saved to file
    test(config = config)

    print(f"Finished job: {job_index}")
