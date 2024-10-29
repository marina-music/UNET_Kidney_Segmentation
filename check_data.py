from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import wandb

print_config()

root_dir = "C:/Users/gabridal/Documents/CT_kidney_segmentation/"

data_dir = os.path.join(root_dir, "dataset")
print(data_dir)

train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
print(train_images)
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:-2], data_dicts[-2:]

set_determinism(seed=0) # set deterministic seed for reproducibility


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ]
)

check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_dataset = first(check_loader)
image, label = (check_dataset["image"][0][0], check_dataset["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}")
# plot the slice [:, :, 80]
plt.figure("check", (12, 6))
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 80], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 80])
plt.show()


# utility function for generating interactive image mask from components
def wb_mask(bg_img, mask):
    return wandb.Image(bg_img, masks={
    "ground truth" : {"mask_data" : mask, "class_labels" : {0: "background", 1: "mask"} }})

def log_spleen_slices(total_slices=100):
    
    wandb_mask_logs = []
    wandb_img_logs = []

    check_ds = Dataset(data=train_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader) # get the first item of the dataloader

    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    
    for img_slice_no in range(total_slices):
        img = image[:, :, img_slice_no]
        lbl = label[:, :, img_slice_no]
        
        # append the image to wandb_img_list to visualize 
        # the slices interactively in W&B dashboard
        wandb_img_logs.append(wandb.Image(img, caption=f"Slice: {img_slice_no}"))

        # append the image and masks to wandb_mask_logs
        # to see the masks overlayed on the original image
        wandb_mask_logs.append(wb_mask(img, lbl))

    wandb.log({"Image": wandb_img_logs})
    wandb.log({"Segmentation mask": wandb_mask_logs})


# 🐝 init wandb with appropiate project and run name
wandb.init(project="MONAI_Spleen_3D_Segmentation", name="slice_image_exploration")
# 🐝 log images to W&B
log_spleen_slices(total_slices=100)
# 🐝 finish the run
wandb.finish()