import pandas as pd
import os

# run nohup python src/dowload.py &
# to download images in the background
# to check the disk image run du -sh
# to check the number of images run ls -l | wc -l

DATASET_FOLDER = 'dataset/'
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
sample_test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test.csv'))
sample_test_out = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample_test_out.csv'))
from utils import download_images
download_images(sample_test['image_link'], 'images')
download_images(train['image_link'], 'images')
download_images(test['image_link'], 'images')