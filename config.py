from torch import nn

from loss import ContentLoss

class dotdict(dict):
    """
    Cheeky helper class that adds dot.notation access to dictionary attributes. 
    Makes the config a bit more readable.
    Stolen from: https://stackoverflow.com/a/23689767/19877091
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __dir__ = dict.keys
    __repr__ = dict.__repr__

class Config():
    EXP = dotdict()
    EXP.USER = "s204163"
    EXP.NAME = "experiment-name"
    EXP.START_EPOCH = 0             # Whether to resume training at some epoch number or start at epoch 0
    EXP.N_EPOCHS = 15               # Number of epochs to train for
    EXP.N_WARMUP_BATCHES = 0        # Number of epochs to warm up the generator before the discriminator starts learning
    EXP.LABEL_SMOOTHING = 0.0       # One-sided label smoothing. The true label will be 1.0 - label_smoothing

    # Logging options
    LOG_TRAIN_PERIOD = 100
    LOG_VALIDATION_PERIOD = 1

    # Data
    DATA = dotdict()
    DATA.TRAIN_GT_IMAGES_DIR = f"/work3/{EXP.USER}/data/ImageNet/train" # Training HR gt images 
    DATA.TEST_SET = 'Set5'                                          # The test set to use; Set5, Set14, BSD100, Urban100
    DATA.TEST_GT_IMAGES_DIR = F"/work3/{EXP.USER}/data/{DATA.TEST_SET}/GTmod12"    # Test HR images
    DATA.TEST_LR_IMAGES_DIR = f"/work3/{EXP.USER}/data/{DATA.TEST_SET}/LRbicx4"    # Test downscaled images
    DATA.TEST_SR_IMAGES_DIR = "results/test"                       # Directory to output the SR images to in test.py
    DATA.SEED = 1312
    DATA.UPSCALE_FACTOR = 4
    DATA.BATCH_SIZE = 16
    DATA.GT_IMAGE_SIZE = 192    # Size of the HR GT images i.e. 192 x 192
    
    # Model
    MODEL = dotdict()
    MODEL.DEVICE = 'cuda:0'
    # Generator
    MODEL.G_IN_CHANNEL = 3     
    MODEL.G_OUT_CHANNEL = 3
    MODEL.G_N_CHANNEL = 64
    MODEL.G_N_RCB = 16
    MODEL.G_LOSS = dotdict()
    MODEL.G_LOSS.VGG19_LAYERS = {       # The layers and weights from VGG19 that are used in the content loss 
        "features.17" : 1/8,
        "features.26" : 1/4,
        "features.35" : 1/2
    }
    # The loss functions used in the generator by default. More can be added after instantiating
    MODEL.G_LOSS.CRITERIONS = {
        "Adversarial"   : nn.BCEWithLogitsLoss(),
        "Content"       : ContentLoss(MODEL.G_LOSS.VGG19_LAYERS, device=MODEL.DEVICE),
        "Pixel"         : nn.MSELoss(),
    }
    # How to weigh the loss functions used in the generator
    MODEL.G_LOSS.CRITERION_WEIGHTS = {  
        "Adversarial"   : 0.005,
        "Content"       : 1.0,
        "Pixel"         : 1.0,
    }
    # A list of criterions the generator should use during warmup
    MODEL.G_LOSS.WARMUP_CRITERIONS = [
        "Pixel",
    ]
    MODEL.D_IN_CHANNEL = 3
    MODEL.D_N_CHANNEL = 32

    # Solver
    SOLVER = dotdict()
    # Discriminator
    SOLVER.D_OPTIMIZER = 'Adam'
    SOLVER.D_BASE_LR = 1e-4
    SOLVER.D_BETA1 = 0.9
    SOLVER.D_BETA2 = 0.999
    SOLVER.D_WEIGHT_DECAY = 0
    SOLVER.D_EPS = 1e-8
    # Generator
    SOLVER.G_OPTIMIZER = 'Adam'
    SOLVER.G_BASE_LR = 1e-4
    SOLVER.G_BETA1 = 0.9
    SOLVER.G_BETA2 = 0.999
    SOLVER.G_WEIGHT_DECAY = 0
    SOLVER.G_EPS = 1e-8
    
    # Scheduler
    SCHEDULER = dotdict()
    SCHEDULER.STEP_SIZE = EXP.N_EPOCHS // 2
    SCHEDULER.GAMMA = 0.1

    def add_g_criterion(self, name: str, value: nn.Module, weight: float) -> None:
        """ Add a crition to the generator """
        self.MODEL.G_LOSS.CRITERIONS[name] = value
        self.MODEL.G_LOSS.CRITERION_WEIGHTS[name] = weight

    def remove_g_criterion(self, name: str) -> None:
        """ Remove a crition from the generator """
        if name in self.MODEL.G_LOSS.CRITERIONS:
            del self.MODEL.G_LOSS.CRITERIONS[name]
            del self.MODEL.G_LOSS.CRITERION_WEIGHTS[name]

    def get_all_params(self) -> str:
        """
        Get a list of all the parameters in the config file. 
        This is useful for logging all the parameters to tensorboard, to know which params were used in a given experiment
        """
        params = [getattr(self,attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        return str(params)