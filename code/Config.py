class Config:
    SR = 32000
    N_MFCC = 13
    # Dataset
    ROOT_FOLDER = './'
    # Training
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 5
    LR = 3e-4
    # Others
    SEED = 42