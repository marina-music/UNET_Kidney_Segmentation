import numpy as np
import pandas as pd
import nibabel as nib
from matplotlib import pyplot as plt
import torch
import os
#from utils.postprocess import colour_code_segmentation

def print_segmentation_output(dataset, best_model, exp_num, device):
    for i in range(10):
        idx = np.random.randint(0, len(dataset))
        image, label = dataset[idx]

        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

        # Predict test image
        pred_mask = best_model(x_tensor)
        # pred_mask=test_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        # x=np.argmax(pred_mask, axis=0)

        # Convert pred_mask from `CHW` format to `HWC` format
        # 출력 전 이미지 dimension 변경
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        # Convert label from `CHW` format to `HWC` format
        label = np.transpose(label, (1, 2, 0))
        image = np.transpose(image, (1, 2, 0))

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('original')
        plt.imshow(image, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title('ground-truth')
        plt.imshow(colour_code_segmentation(np.argmax(label, axis=2)))

        plt.subplot(1, 3, 3)
        plt.title("prediction")
        plt.imshow(colour_code_segmentation(np.argmax(pred_mask, axis=2)))
        if not os.path.exists(f'result/exp{exp_num}'):
            os.mkdir(f'result/exp{exp_num}')
        plt.savefig(f'result/exp{exp_num}/prediction_{i}.png')
        plt.show()

def save_segmentation_as_nii(dataset, model, output_dir, device):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx in range(len(dataset)):
        image, label = dataset[idx]
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

        # Predict the segmentation
        pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        pred_mask = np.argmax(pred_mask, axis=0)  # Assuming multi-class output



def save_all_segmentation_outputs(dataset, best_model, exp_num, device):
    # Create the directory to save results if it doesn't exist
    result_dir = f'result/exp{exp_num}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Iterate over all images in the dataset
    for idx in range(len(dataset)):
        image= dataset[idx]

        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

        # Predict test image
        pred_mask = best_model(x_tensor)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()

        # Convert pred_mask from `CHW` to `HWC` format
        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_mask = np.argmax(pred_mask, axis=2)

        # Save predicted mask as a .nii.gz file
        output_path = os.path.join(result_dir, f'predicted_segmentation_{idx}.nii.gz')
        nib.save(nib.Nifti1Image(pred_mask, affine=np.eye(4)), output_path)

def print_logs(train_logs_list, valid_logs_list, exp_num, score_name):
    train_logs_df = pd.DataFrame(train_logs_list)
    valid_logs_df = pd.DataFrame(valid_logs_list)
    train_logs_df.transpose()
    plt.figure(figsize=(20, 8))
    plt.plot(train_logs_df.index.tolist(), train_logs_df[score_name].tolist(), lw=3, label='Train')
    plt.plot(valid_logs_df.index.tolist(), valid_logs_df[score_name].tolist(), lw=3, label='Valid')
    plt.xlabel('Epochs', fontsize=21)
    plt.ylabel(f'{score_name} Score', fontsize=21)
    plt.ylim([-0.5, 1.5])
    plt.title(f'{score_name} Score Plot', fontsize=21)
    plt.grid()
    if not os.path.exists(f'result/exp{exp_num}'):
        os.mkdir(f'result/exp{exp_num}')
    plt.savefig(f'./result/exp{exp_num}/{score_name}_score_plot.png')
    plt.show()

    # return best score (%)
    if score_name == 'IoU':
        return round(max(valid_logs_df[score_name].tolist())*100, 2)
    elif score_name == 'Loss':
        return round((1.-min(valid_logs_df[score_name].tolist()))*100, 2)


def inference(valid_set, exp_num, device): #train_logs_list, valid_logs_list, device):
    best_model = torch.load(f'./SavedModel/best_model_exp{exp_num}.pt')
    save_all_segmentation_outputs(valid_set, best_model, exp_num, device)
    #print_segmentation_output(valid_set, best_model, exp_num, device)
    #best_dsc = print_logs(train_logs_list, valid_logs_list, exp_num, score_name='Loss')
    print(f'Saving all segmentation outputs to ./result/exp{exp_num}')
    #print(f"best score {exp_num} : DSC {best_dsc}, IoU {best_iou}")
