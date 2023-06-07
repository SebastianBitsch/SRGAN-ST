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

def warmup_gan(config: Config, epochs:int = 5) -> Config:
    """ Warmup the generator """
    config.EXP.N_EPOCHS = epochs
    config.EXP.NAME = f"resnet{epochs}"
    return config

def warmup_experiment(config: Config, index:int) -> Config:
    """ Run a warmup experiment where we test how warming up and label smoothing performs """
    config.EXP.NAME = ['warmup-exp', 'labelsmoothing-exp', 'warmup-labelsmoothing-exp'][index]
    config.EXP.LABEL_SMOOTHING = [0, 0.1, 0.1][index]
    config.EXP.N_WARMUP_BATCHES = [5000, 0, 5000][index]
    return config

def srgan_bbgan(config: Config, index:int) -> Config:
    """ stock srgan vs bbgan"""
    config.EXP.NAME = ['stock-srgan-fixed-adv', 'stock-bbgan-lorna-fixed-adv'][index]
    config.MODEL.CONTINUE_FROM_WARMUP = True
    config.MODEL.WARMUP_WEIGHTS = "results/resnet20/g_best.pth"
    # config.MODEL.WARMUP_WEIGHTS = "results/SRResNet-lorna-pretrained.pth.tar"
    if index == 1:
        config.add_g_criterion("BestBuddy", BestBuddyLoss(), 1.0)
    return config


def test_gramloss(config: Config) -> Config:
    config.EXP.NAME = "gram-model"
    config.add_g_criterion("Gram", GramLoss(), 10.0)
    return config


def test_st_losses(config: Config) -> Config:
    config.EXP.NAME = "st-now"
    config.EXP.N_EPOCHS = 3
    config.add_g_criterion("PatchwiseSTLoss", PatchwiseStructureTensorLoss(sigma=5))
    config.add_g_criterion("STLoss", StructureTensorLoss(sigma=5))
    return config


def ablation_study(config: Config, index:int) -> Config:
    config.EXP.NAME = ['ablation-plain', 'ablation-bestbuddy', 'ablation-gram', 'ablation-patchwise-st', 'ablation-st'][index]
    config.MODEL.CONTINUE_FROM_WARMUP = True
    config.MODEL.WARMUP_WEIGHTS = "results/resnet3/g_best.pth"

    config.remove_g_criterion("Pixel")
    if index == 1:
        config.add_g_criterion("BestBuddy", BestBuddyLoss(), 1.0)
    elif index == 2:
        config.add_g_criterion("Gram", GramLoss(), 1.0)
    elif index == 3:
        config.add_g_criterion("PatchwiseST", PatchwiseStructureTensorLoss(), 1.0)
    elif index == 4:
        config.add_g_criterion("ST", StructureTensorLoss(), 1.0)
    
    return config


def debug_test(config):
    config.EXP.NAME = "srgan-run2"
    config.MODEL.CONTINUE_FROM_WARMUP = True
    config.MODEL.WARMUP_WEIGHTS = "results/SRRESNET/g_best.pth.tar"#"results/SRResNet-lorna-pretrained.pth.tar"

    config.remove_g_criterion("Pixel")
    config.remove_g_criterion("Content")
    return config


if __name__ == "__main__":

    # Get job-index from bash, if the job is not an array it will be zero
    job_index = get_jobindex()
    print(f"Running job: {job_index}")

    # Edit config based on 
    config = Config()

    # config = warmup_gan(config, epochs = 5)
    # config = ablation_study(config, job_index)
    config = debug_test(config)
    
    train(config = config)

    test(config = config, save_images = True)

    print(f"Finished job: {job_index}")
