import os

from torch import nn

from config import Config
from loss import BestBuddyLoss, GramLoss, PatchwiseStructureTensorLoss, StructureTensorLoss, ContentLossDiscriminator, ContentLossVGG

from train import train
from warmup import warmup
from validate import test


def get_jobindex(fallback:int = 0) -> int:
    """Get the job-index set in bash. This is mostly for array jobs where multiple models are trained in parallel"""
    num = os.getenv('job_index')
    return int(num) if num else fallback



def warmup_gan(config: Config, epochs:int = 5) -> Config:
    """ Warmup the generator """
    config.EXP.N_EPOCHS = epochs
    config.EXP.NAME = f"resnet{epochs}"
    config.G_CHECKPOINT_INTERVAL = 5
    return config



def assorted_tests(config, i):
    # config.EXP.NAME = ['ablation-cd-plain', 'ablation-cd-bestbuddy', 'ablation-cd-gram', 'ablation-cd-patchwise-st', 'ablation-cd-st'][i]
    config.MODEL.G_CONTINUE_FROM_WARMUP = True
    config.MODEL.G_WARMUP_WEIGHTS = "results/SRResNet-lorna-pretrained.pth"
    config.MODEL.D_CONTINUE_FROM_WARMUP = True
    config.MODEL.D_WARMUP_WEIGHTS = "results/discriminator-lorna-pretrained.pth"
    config.SOLVER.D_UPDATE_INTERVAL = 100
    config.EXP.LABEL_SMOOTHING = 0.1
    config.G_CHECKPOINT_INTERVAL = 5

    # if i == 0: # go
    #     config.EXP.NAME = "srgan-w-pixel-reweight"
    #     config.add_g_criterion("Pixel", nn.MSELoss(), weight=1.0)
    #     config.add_g_criterion("ContentVGG", ContentLossVGG(config), weight = 1.0)
    # if i == 1: # go
        # config.EXP.NAME = "patchwise-st-w-pixel-reweight-try2"
        # config.add_g_criterion("Pixel", nn.MSELoss(), weight=1.0)
        # config.add_g_criterion("ContentVGG", ContentLossVGG(config), weight = 1.0)
        # config.add_g_criterion("PatchwiseST", PatchwiseStructureTensorLoss(), 100.0)
    # if i == 2: # - running
    #     config.EXP.NAME = "st-c-pixel-reweight"
    #     config.add_g_criterion("Pixel", nn.MSELoss(), weight=1.0)
    #     # config.add_g_criterion("ContentDiscriminator", ContentLossDiscriminator(config), weight = 2000.0)
    #     config.add_g_criterion("ContentVGG", ContentLossVGG(config), weight = 1.0)
    #     config.add_g_criterion("ST", StructureTensorLoss(), 1/3)
    # if i == 3: # - running
    #     config.EXP.NAME = "st-c-no-pixel-reweighted"
    #     # config.add_g_criterion("ContentDiscriminator", ContentLossDiscriminator(config), weight = 2000.0)
    #     config.add_g_criterion("ContentVGG", ContentLossVGG(config), weight = 1.0)
    #     config.add_g_criterion("ST", StructureTensorLoss(), 1/3)
    # if i == 4: # - go
    #     config.EXP.NAME = "srgan-double-content"
    #     config.add_g_criterion("Pixel", nn.MSELoss(), weight=1.0)
    #     config.add_g_criterion("ContentVGG", ContentLossVGG(config), weight = 1.0)
    #     config.add_g_criterion("ContentDiscriminator", ContentLossDiscriminator(config), weight = 2000.0)
    # if i == 5: # - go
    #     config.EXP.NAME = "patchwise-st-double-content"
    #     config.add_g_criterion("Pixel", nn.MSELoss(), weight=1.0)
    #     config.add_g_criterion("ContentVGG", ContentLossVGG(config), weight = 1.0)
    #     config.add_g_criterion("ContentDiscriminator", ContentLossDiscriminator(config), weight = 2000.0)
    #     config.add_g_criterion("PatchwiseST", PatchwiseStructureTensorLoss(), 100.0)
    if i == 0: # - go
        config.EXP.NAME = "srgan-double-content-no-pixel"
        config.add_g_criterion("ContentVGG", ContentLossVGG(config), weight = 1.0)
        config.add_g_criterion("ContentDiscriminator", ContentLossDiscriminator(config), weight = 2000.0)
    if i == 1: # - go
        config.EXP.NAME = "patchwise-st-double-content-no-pixel"
        config.add_g_criterion("ContentVGG", ContentLossVGG(config), weight = 1.0)
        config.add_g_criterion("ContentDiscriminator", ContentLossDiscriminator(config), weight = 2000.0)
        config.add_g_criterion("PatchwiseST", PatchwiseStructureTensorLoss(), 100.0)
    if i == 2: # - go
        config.EXP.NAME = "patchwise-st-double-content-no-pixel-a100"
        config.add_g_criterion("ContentVGG", ContentLossVGG(config), weight = 1.0)
        config.add_g_criterion("ContentDiscriminator", ContentLossDiscriminator(config), weight = 2000.0)
        config.add_g_criterion("PatchwiseST", PatchwiseStructureTensorLoss(), 100.0)

    return config


if __name__ == "__main__":

    # Get job-index from bash, if the job is not an array it will be zero
    job_index = get_jobindex()
    print(f"Running job: {job_index}")

    config = Config()

    # config = ablation_study(config, job_index)
    config = assorted_tests(config, job_index)

    # Train and validate the experiment given by the config file
    train(config = config)
    test(config = config, save_images = True)

    print(f"Finished job: {job_index}")
