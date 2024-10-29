import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import os
from utils.postprocess import colour_code_segmentation

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


def inference(valid_set, exp_num, train_logs_list, valid_logs_list, device):
    best_model = torch.load(f'./SavedModel/best_model_exp{exp_num}.pt')
    print_segmentation_output(valid_set, best_model, exp_num, device)
    best_iou = print_logs(train_logs_list, valid_logs_list, exp_num, score_name='IoU')
    best_dsc = print_logs(train_logs_list, valid_logs_list, exp_num, score_name='Loss')

    print(f"best score {exp_num} : DSC {best_dsc}, IoU {best_iou}")
