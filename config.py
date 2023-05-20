class dotdict(dict):
    """
    Cheeky helper class that adds dot.notation access to dictionary attributes. 
    Makes the config a bit nicer.
    Stolen from: https://stackoverflow.com/a/23689767/19877091
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __dir__ = dict.keys

class Config:

    EXP = dotdict()
    EXP.NAME = "placeholder"
    EXP.START_EPOCH = 0         # Whether to resume training at some epoch number or start at epoch 0
    EXP.N_EPOCHS = 5        
    EXP.N_WARMUP_BATCHES = 0

    # Data
    DATA = dotdict()
    DATA.TRAIN_GT_IMAGES_DIR = "/work3/s204163/data/ImageNet/train"
    DATA.TEST_GT_IMAGES_DIR = "/work3/s204163/data/Set5/GTmod12"
    DATA.TEST_LR_IMAGES_DIR = "/work3/s204163/data/Set5/LRbicx4"
    DATA.SEED = 1
    DATA.UPSCALE_FACTOR = 4
    DATA.BATCH_SIZE = 16
    DATA.GT_IMAGE_SIZE = 192
    
    # Model
    MODEL = dotdict()
    MODEL.DEVICE = 'cuda:0'
    # Generator
    MODEL.G_IN_CHANNEL = 3
    MODEL.G_OUT_CHANNEL = 3
    MODEL.G_N_CHANNEL = 64
    MODEL.G_N_RCB = 16
    MODEL.G_LOSS = dotdict()
    MODEL.G_LOSS.VGG19_LAYERS = {
        "features.17" : 1/8,
        "features.26" : 1/4,
        "features.35" : 1/2
    }

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
    
    LOG_TRAIN_PERIOD = 100
    LOG_VALIDATION_PERIOD = 1