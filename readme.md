

## Dataset (KiTS 23)


![Untitled](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/b21dfca1-d6ac-4e26-9c43-e0bc54e3c004) |![Untitled1](https://github.com/hyunji-lee99/CT_kidney_segmentation/assets/58133945/10c29ffd-31fe-4ee9-bd2f-b09c10672a5c)
--- | --- | 
https://kits-challenge.org/kits23/

**Installation**

```
git clone https://github.com/neheller/kits23
cd kits23
pip3 install -e .
```

**Original Class** : background(0), kidney(1), tumor(2), cyst(3)
** Probably will need to changed to background(0), left kidney(1), right kidney(2)**

# Setting Up a Virtual Environment with MONAI and PyTorch on Windows

This guide will walk you through creating a Python virtual environment in Visual Studio Code (VS Code) and installing MONAI and PyTorch for your Windows system.

## Prerequisites
- **Python**: Ensure Python is installed and added to your system PATH.
- **VS Code**: Make sure Visual Studio Code is installed.

## Steps

### 1. Open VS Code
Open VS Code and navigate to your project folder.

### 2. Open the Terminal
Open the terminal in VS Code by navigating to `View > Terminal` or pressing `Ctrl + ``.

### 3. Create a Virtual Environment
Run the following command in the terminal to create a virtual environment:
   ```bash
   python -m venv env
   ```

### 4. Activate the Virtual Environment
Activate the environment by running:
```bash
   .\env\Scripts\activate
```

## Prerequisites

### 1. Python and Libraries:
Install the required libraries listed in requirements.txt, including MONAI, PyTorch, and Weights & Biases (wandb) for tracking experiments.

### 2. Dataset
Place your images and labels in the designated directories as specified in the config file under data.image and data.label. The directory structure should look like this.

```bash
dataset/

├── imagesTr/       # Training images (in .nii format but can be modified)

└── labelsTr/       # Corresponding labels (in .nii format but can be modified)
```

## Configuration file explaination

The config_template.yaml file includes several sections that control different aspects of training:

- wandb: Settings for experiment tracking with Weights & Biases
   - entity_name: Your W&B username.
   - project_name: The name of the project in W&B.
   - experiment_name: Experiment identifier.
- data: Dataset parameters.
  - image: Path to the directory containing training images.
  - label: Path to the directory containing corresponding labels.
  - num_class: Set to 2 for binary segmentation (background and spleen).
  - cache_rate: Defines how much of the dataset to keep in memory (1.0 means fully cached).
  - num_workers: Number of worker threads for data loading.
  - log: Logging settings.

- name: Name of the model.
- save_dir: Directory to save the best model weights.
- loss: Defines the loss function.
   - Uses monai.losses.DiceCELoss with softmax and one-hot encoding for the labels.
- net: Network architecture.
  - Defines a 3D UNet with specified spatial dimensions, input/output channels, and layer parameters.
- train: Training parameters.
  - batch_size: Batch size for training.
  - val_batch_size: Batch size for validation.
  - max_epochs: Maximum number of training epochs.
  - val_interval: Frequency of validation (e.g., every 5 epochs).
  - learning_rate: Initial learning rate for the optimizer.
  - augmentation: A list of transformations applied to the training data.
- valid: Validation parameters.
   - roi_size: Size of the region of interest for sliding window inference.
   - sw_batch_size: Batch size during sliding window inference.
   - augmentation: A list of transformations applied to the validation data.

## Tips for Successful Training (and improve your results)

1. **Tune `learning_rate` and `batch_size`**:
   - Start with the default settings but adjust if needed based on GPU memory and model performance. Lower batch sizes help with memory constraints, while adjusting the learning rate can stabilize training.

2. **Use Data Augmentation**:
   - Apply transformations like rotation, flipping, and scaling to make the model more robust. Experiment with different augmentations to reduce overfitting, especially if the dataset is small.

3. **Ensure a Good Train-Validation Split**:
   - Aim for an 80-20 split and avoid data leakage (e.g., keep scans from the same patient in one set). A representative split helps the model generalize better.

4. **Try K-Fold Cross-Validation**:
   - K-fold cross-validation improves reliability by training on different subsets of data, ensuring consistent performance across the dataset.

5. **Add Validation Metrics**:
   - Besides Dice score, consider sensitivity, specificity, or precision for a more comprehensive view of model performance, especially for imbalanced data.

6. **Monitor Validation Metrics**:
   - Watch for a flat validation metric—it might indicate label issues, insufficient data, or overfitting. Regularly checking metrics helps you catch and fix issues early.
