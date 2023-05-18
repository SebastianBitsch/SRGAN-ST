class dotdict(dict):
    """
    Cheeky helper class that adds dot.notation access to dictionary attributes. 
    Makes the config a bit nicer.
    Stolen from: https://stackoverflow.com/a/23689767/19877091
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Config:

    DATA = dotdict()
    DATA.SEED = 1



    MODEL = dotdict()
    MODEL.DEVICE = 'cuda'
    MODEL.G.IN_CHANNEL = 3
    MODEL.G.OUT_CHANNEL = 3




config = Config()