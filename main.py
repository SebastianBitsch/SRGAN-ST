import os
from config import Config
from train import train
from warmup_generator import warmup
from validate import test

from loss import BestBuddyLoss, GramLoss, PatchwiseStructureTensorLoss, StructureTensorLoss

def get_jobindex(fallback:int = 0) -> int:
    """Get the job-index set in bash. This is mostly for array jobs where multiple models are trained in parallel"""
    num = os.getenv('job_index')
    return int(num) if num else fallback


def srgan(config: Config) -> Config:
    config.EXP.NAME = "srgan-with-resnet"
    config.MODEL.CONTINUE_FROM_WARMUP = True
    return config

def resnet(config: Config) -> Config:
    config.EXP.NAME = "resnet_recheck"
    return config

def warmup_experiment(config: Config, index:int) -> Config:
    """ Run a warmup experiment where we test how warming up and label smoothing performs """
    config.EXP.NAME = ['warmup-exp', 'labelsmoothing-exp', 'warmup-labelsmoothing-exp'][index]
    config.EXP.LABEL_SMOOTHING = [0, 0.1, 0.1][index]
    config.EXP.N_WARMUP_BATCHES = [5000, 0, 5000][index]
    return config

def srgan_bbgan(config: Config, index:int) -> Config:
    """ stock srgan vs bbgan"""
    config.EXP.NAME = ['stock-srgan-w-pixel', 'stock-bbgan-w-pixel'][index]
    config.MODEL.CONTINUE_FROM_WARMUP = True
    config.MODEL.WARMUP_WEIGHTS = "results/resnet20/g_best.pth"
    if index == 1:
        config.add_g_criterion("BestBuddy", BestBuddyLoss(), 1.0)
    return config


def test_gramloss(config: Config) -> Config:
    config.EXP.NAME = "gram-model"
    config.add_g_criterion("Gram", GramLoss(), 10.0)
    return config


def test_st_losses(config: Config) -> Config:
    config.EXP.NAME = "st-testing"
    config.EXP.N_EPOCHS = 3
    config.add_g_criterion("PatchwiseSTLoss", PatchwiseStructureTensorLoss(sigma=5))
    config.add_g_criterion("STLoss", StructureTensorLoss(sigma=5))
    return config


if __name__ == "__main__":

    # Get job-index from bash, if the job is not an array it will be zero
    job_index = get_jobindex()

    # Edit config based on 
    config = Config()


    print(f"Running job: {job_index}")


    config = test_st_losses(config)
    train(config = config)

    # Done mostly just to get some images saved to file
    test(config = config, save_images = True)

    print(f"Finished job: {job_index}")
