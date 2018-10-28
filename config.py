# Configuration file
import os

BATCH_SIZE = 16
SHAPE = (512, 512, 4)
SHAFFLE_FLAG = True
USE_CACHE_FLAG = False

TRESHOLDS_PATH = os.path.join(
    'data/',
    'tresholds.npy'
)