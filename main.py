import os
from config import Config
from train import train
from warmup import warmup
from validate import test
from loss import BestBuddyLoss, GramLoss, PatchwiseStructureTensorLoss, StructureTensorLoss, ContentLossDiscriminator


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


def srgan_bbgan(config: Config, index:int) -> Config:
    """ stock srgan vs bbgan"""
    config.EXP.NAME = ['stock-srgan-fixed-adv', 'stock-bbgan-lorna-fixed-adv'][index]
    config.MODEL.CONTINUE_FROM_WARMUP = True
    config.MODEL.WARMUP_WEIGHTS = "results/resnet20/g_best.pth"
    # config.MODEL.WARMUP_WEIGHTS = "results/SRResNet-lorna-pretrained.pth.tar"
    if index == 1:
        config.add_g_criterion("BestBuddy", BestBuddyLoss(), 1.0)
    return config



def ablation_study(config: Config, index:int) -> Config:
    config.EXP.NAME = ['ablation-plain', 'ablation-bestbuddy', 'ablation-gram', 'ablation-patchwise-st', 'ablation-st'][index]
    config.MODEL.G_CONTINUE_FROM_WARMUP = True
    config.MODEL.G_WARMUP_WEIGHTS = "results/SRResNet-lorna-pretrained.pth"
    config.SOLVER.D_UPDATE_INTERVAL = 100
    config.EXP.LABEL_SMOOTHING = 0.1
    config.G_CHECKPOINT_INTERVAL = 5

    config.remove_g_criterion("Pixel")

    if index == 0: # srgan
        pass
    if index == 1:
        config.add_g_criterion("BestBuddy", BestBuddyLoss(), 50.0)
    elif index == 2:
        config.add_g_criterion("Gram", GramLoss(), 500.0)
    elif index == 3:
        config.add_g_criterion("PatchwiseST", PatchwiseStructureTensorLoss(), 100.0)
    elif index == 4:
        config.add_g_criterion("ST", StructureTensorLoss(), 10.0)
    
    return config

def best_buddy_test(config, i):
    config.EXP.NAME = ['bb-no-pixel-w-warmup', 'bb-content1', 'bb-w-pixel-content0'][i]
    config.SOLVER.D_UPDATE_INTERVAL = 100
    config.EXP.LABEL_SMOOTHING = 0.1
    config.add_g_criterion("BestBuddy", BestBuddyLoss(), 50.0)
    config.MODEL.G_CONTINUE_FROM_WARMUP = True
    config.MODEL.G_WARMUP_WEIGHTS = "results/SRResNet-lorna-pretrained.pth"#"results/SRRESNET/g_best.pth.tar"

    if i == 0:
        config.remove_g_criterion("Pixel")
    elif i == 1:
        config.remove_g_criterion("ContentVGG")
        config.add_g_criterion("ContentDiscriminator", ContentLossDiscriminator(extraction_layers=config.MODEL.G_LOSS.DISC_FEATURES_LOSS_LAYERS, config=config), 2.0)
    elif i == 2:
        pass
    
    return config


if __name__ == "__main__":

    # Get job-index from bash, if the job is not an array it will be zero
    job_index = get_jobindex()
    print(f"Running job: {job_index}")

    # Edit config based on 
    config = Config()

    config = ablation_study(config, job_index)

    train(config = config)
    test(config = config, save_images = True)

    print(f"Finished job: {job_index}")
