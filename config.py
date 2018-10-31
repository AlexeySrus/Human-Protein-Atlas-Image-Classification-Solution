# Configuration file
import os
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

BATCH_SIZE = 8
SHAPE = (512, 512, 4)
SHAFFLE_FLAG = True
USE_CACHE_FLAG = False

BASE_MODEL=Xception

TRESHOLDS_PATH = os.path.join(
    'data/',
    'tresholds.npy'
)
