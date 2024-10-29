from torch.utils.data import Dataset
import numpy as np
from utils.preprocess import onehot_encoding

class SegmentationDataset(Dataset):
  def __init__(
      self,
      data_image_array,
      data_label_array,
      data_idx_array,
      augmentations=None
      ):
    self.data_image_array=data_image_array
    self.data_label_array=data_label_array
    self.data_idx_array=data_idx_array
    self.augmentations=augmentations

  def __getitem__(self, index):
    # start index, number of slices
    start, ns = self.data_idx_array[index]
    image=np.expand_dims(self.data_image_array[start+ns//2,:,:], axis=0)
    label=onehot_encoding(self.data_label_array[start+ns//2,:,:],3)

    if self.augmentations:
      sample=self.augmentations(image=image, label=label)
      image, label=sample

    return image, label

  def __len__(self):
    return len(self.data_idx_array)