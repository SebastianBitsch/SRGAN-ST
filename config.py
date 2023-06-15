from torch import nn, cuda

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

    DEVICE = "cuda:0" if cuda.is_available() else "cpu"               # Device, should only really be cuda tbh

    EXP = dotdict()
    EXP.USER = "s204163"            # The user id on HPC the scratch directory in on
    EXP.NAME = "experiment-name"    # Name of the experiment, folders etc. will be named by this
    EXP.START_EPOCH = 0             # Whether to resume training at some epoch number or start at epoch 0
    EXP.N_EPOCHS = 40               # Number of epochs to train for
    EXP.LABEL_SMOOTHING = 0.1       # One-sided label smoothing. The true label will be 1.0 - label_smoothing

    # Logging options
    LOG_TRAIN_PERIOD = 100          # How many iterations should be between each print loss statement when training
    LOG_VALIDATION_PERIOD = 1       # How many iterations should be between each psnr/ssim log statement when validating
    D_CHECKPOINT_INTERVAL = 100     # How many epochs should be between every time the discriminator state dict is saved
    G_CHECKPOINT_INTERVAL = 100     # How many epochs should be between every time the generator state dict is saved

    # Data
    DATA = dotdict()
    DATA.TRAIN_GT_IMAGES_DIR = f"/work3/{EXP.USER}/data/train"                      # Location of training HR gt images 
    DATA.TEST_SET = 'Set5'                                                          # The test set to use; Set5, Set14, BSD100, Urban100
    DATA.TEST_GT_IMAGES_DIR = F"/work3/{EXP.USER}/data/{DATA.TEST_SET}/GTmod12"     # Location of test HR images
    DATA.TEST_LR_IMAGES_DIR = f"/work3/{EXP.USER}/data/{DATA.TEST_SET}/LRbicx4"     # Location of test downscaled images
    DATA.TEST_SR_IMAGES_DIR = "results/_test"                                       # Directory to output the SR images to in test.py
    DATA.SEED = 0               # The seed to use for reproducability
    DATA.UPSCALE_FACTOR = 4     # The upscale factor, only really tested for 4
    DATA.BATCH_SIZE = 16        # The batchsize of images to use
    DATA.GT_IMAGE_SIZE = 96     # Size of the HR ground truth images i.e. 192 x 192
    
    # Model
    MODEL = dotdict()
    MODEL.G_CONTINUE_FROM_WARMUP = False        # Should the generator continue from some pretrained weights?
    MODEL.G_WARMUP_WEIGHTS = ""                 # Directory of weights to use if we continue from warmup
    MODEL.D_CONTINUE_FROM_WARMUP = False        # Should the generator continue from some pretrained weights?
    MODEL.D_WARMUP_WEIGHTS = ""                 # Directory of weights to use if we continue from warmup

    # Generator network parameters
    MODEL.G_IN_CHANNEL = 3          # In color channels
    MODEL.G_OUT_CHANNEL = 3         # Out color channels
    MODEL.G_N_CHANNEL = 64          # Num channels
    MODEL.G_N_RCB = 16              # Num blocks
    
    MODEL.G_LOSS = dotdict()
    # The layers and weights from VGG19 that are used in the ContentLossVGG()
    # These are the layers and weights used by GramGAN in their paper
    MODEL.G_LOSS.VGG19_LAYERS = {
        "features.17" : 1/8,
        "features.26" : 1/4,
        "features.35" : 1/2
    }
    # The layers and weights from the discriminator to use in the ContentLossDiscriminator()
    MODEL.G_LOSS.DISC_FEATURES_LOSS_LAYERS = {
        "features.4" : 1/4,
        "features.10" : 1/2,
    }
    # The loss functions used in the generator by default. More can be added after instantiating
    MODEL.G_LOSS.CRITERIONS = {
        "Adversarial"   : nn.BCEWithLogitsLoss(),
    }
    # How to weigh the loss functions used in the generator
    # TODO: Might be nicer with non manual weighting - i.e. all losses contribute exactly equal amounts.
    # See: https://discuss.pytorch.org/t/how-to-normalize-losses-of-different-scale/126995
    MODEL.G_LOSS.CRITERION_WEIGHTS = {
        "Adversarial"   : 0.001,
        "ContentVGG"    : 1.0,
        "ContentDiscriminator" : 2000.0,
        "Pixel"         : 1.0,
        "BestBuddy"     : 50.0,
        "Gram"          : 500.0,
        "PatchwiseST"   : 100.0,
        "ST"            : 1/3
    }
    # Which criterions should the generator use during warmup. Defaults to just pixel loss
    MODEL.G_LOSS.WARMUP_CRITERIONS = {
        "Pixel"         : nn.MSELoss()
    }
    MODEL.G_LOSS.WARMUP_WEIGHTS = {
        "Pixel"         : 1.0
    }
    MODEL.D_IN_CHANNEL = 3
    MODEL.D_OUT_CHANNEL = 1
    MODEL.D_N_CHANNEL = 64

    # Solver
    SOLVER = dotdict()
    # Discriminator
    SOLVER.D_UPDATE_INTERVAL = 100
    SOLVER.D_OPTIMIZER = 'Adam'
    SOLVER.D_BASE_LR = 1e-4
    SOLVER.D_BETA1 = 0.9
    SOLVER.D_BETA2 = 0.999
    SOLVER.D_WEIGHT_DECAY = 0
    SOLVER.D_EPS = 1e-4
    # Generator
    SOLVER.G_OPTIMIZER = 'Adam'
    SOLVER.G_BASE_LR = 1e-4
    SOLVER.G_BETA1 = 0.9
    SOLVER.G_BETA2 = 0.999
    SOLVER.G_WEIGHT_DECAY = 0
    SOLVER.G_EPS = 1e-4
    
    # Scheduler
    SCHEDULER = dotdict()
    SCHEDULER.STEP_SIZE = EXP.N_EPOCHS // 2
    SCHEDULER.GAMMA = 0.5 # or 0.5


    def add_g_criterion(self, name: str, value: nn.Module, weight: float = 1.0) -> None:
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