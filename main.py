import os
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.save_dataset import save_dataset
import numpy as np
from data_loader.dataloader import SegmentationDataset
from torch.utils.data import DataLoader
import torch

# load kits23 dataset
DATA_DIR = './data/kits23/dataset/'

ORIGINAL_IMAGES = []
MASK_IMAGES = []

CASES = sorted(os.listdir(DATA_DIR))
for c in CASES:
    if c == 'kits23.json':
        continue
    if len(os.listdir(DATA_DIR+c)) == 2:
        ORIGINAL_IMAGES.append(os.path.join(DATA_DIR, c, "imaging.nii.gz"))
        MASK_IMAGES.append(os.path.join(DATA_DIR, c, "segmentation.nii.gz"))

df_data = pd.DataFrame({'image': ORIGINAL_IMAGES, 'label': MASK_IMAGES})
print(f"Number of Data : {len(df_data)}")

# reduced the number of patients to 300 due to lack of memory issue
# split train set / valid set
train_df, valid_df = train_test_split(df_data[:300], test_size=0.2, random_state=42)
print(f"number of train set : {len(train_df)}, number of valid set : {len(valid_df)}")

# save compressed dataset to npz file
# save_dataset(train_df, input='train')
# save_dataset(valid_df, input='valid')

# load saved data
# train set
concat_train_image_npz = np.load("data/train_image_concat.npz")['data'].astype(np.float32)/255.0
concat_train_label_npz = np.load("data/train_label_concat.npz")['data'].astype(np.float32)
concat_train_index_npz = np.load("data/train_index_concat.npz")['data']
# valid set
concat_valid_image_npz = np.load("data/valid_image_concat.npz")['data'].astype(np.float32)/255.0
concat_valid_label_npz = np.load("data/valid_label_concat.npz")['data'].astype(np.float32)
concat_valid_index_npz = np.load("data/valid_index_concat.npz")['data']

train_set = SegmentationDataset(data_image_array=concat_train_image_npz, data_label_array=concat_train_label_npz,
                                data_idx_array=concat_train_index_npz, augmentations=None)
valid_set = SegmentationDataset(data_image_array=concat_valid_image_npz, data_label_array=concat_valid_label_npz,
                                data_idx_array=concat_valid_index_npz, augmentations=None)

# define data loader
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
print(f"total number of batches in train loader : {len(train_loader)}")
print(f"total number of batches in valid loader : {len(valid_loader)}")

