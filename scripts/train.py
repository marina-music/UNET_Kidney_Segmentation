import os
import glob
import yaml
from tqdm import tqdm
import wandb
import torch
import importlib
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose, AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Dynamically load the model based on config prefix
def load_model_from_config(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class_path = list(config['net'].keys())[0]
    model_kwargs = config['net'][model_class_path].get('kwargs', {})

    if model_class_path.startswith("monai."):
        model_module_path, model_class_name = model_class_path.rsplit(".", 1)
    elif model_class_path.startswith("CustomModels."):
        model_module_path, model_class_name = model_class_path.replace("CustomModels.", "").rsplit(".", 1)
    else:
        model_module_path, model_class_name = model_class_path.rsplit(".", 1)

    # Import the module and class dynamically
    model_module = importlib.import_module(model_module_path)
    model_class = getattr(model_module, model_class_name)
    return model_class(**model_kwargs).to(device)

 # Utility function to dynamically load a transformation class
def load_transform(class_path, args, kwargs):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    transform_class = getattr(module, class_name)
    return transform_class(*args, **kwargs)

def main():
    # Load configuration
    with open("C:\\Users\\Marina.VALVE\\GitHub\\UNET_Kidney_Segmentation\\configs\\config_template.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Ensure W&B API key is set and login if needed
    wandb.login()

    # Generate timestamp for unique experiment naming
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{config['wandb']['experiment_name']}_{timestamp}"

    # Initialize W&B run with settings from config
    run = wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity_name'],
        name=experiment_name
    )

    # Define custom metrics to track
    run.define_metric("train/loss", summary="min")
    run.define_metric("val/dice_metric", summary="max")
    run.define_metric("val/sensitivity", summary="max")

    # Log the configuration dictionary to W&B for experiment tracking
    wandb.config.update(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dynamically load the model based on config prefix
    model = load_model_from_config(config)
    model.to(device)

    # Load loss function from config
    loss_class_path, loss_params = next(iter(config['loss'].items()))
    loss_args = loss_params.get("args", [])
    loss_kwargs = loss_params.get("kwargs", {})
    module_path, class_name = loss_class_path.rsplit(".", 1)
    loss_module = importlib.import_module(module_path)
    loss_class = getattr(loss_module, class_name)
    loss_function = loss_class(*loss_args, **loss_kwargs)

    # Prepare optimizer
    learning_rate = config['train']['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config['train']['max_epochs'], eta_min=1e-9)

    # Initialize metrics and post-processing
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    # Prepare data from configuration
    # Create the data list by pairing image and label files
    image_files = sorted(glob.glob(os.path.join(config['data']['image'], "*.nii")))  # Adjust extension if needed
    label_files = sorted(glob.glob(os.path.join(config['data']['label'], "*.nrrd")))

    assert len(image_files) == len(label_files), "Mismatch between number of images and labels."

    # Prepare the dataset as a list of dictionaries
    data = [{"image": img, "label": lbl} for img, lbl in zip(image_files, label_files)]
    assert len(data) > 0, "Data list is empty. Check image and label paths."

    # Split into training and validation files
    train_files, val_files = data[:-2], data[-2:]

    # Parse and apply transformations from config for training and validation
    train_transforms = Compose([
        load_transform(aug_class, aug_details.get('args', []), aug_details.get('kwargs', {}))
        for aug in config['train']['augmentation']
        for aug_class, aug_details in aug.items()
    ])

    val_transforms = Compose([
        load_transform(aug_class, aug_details.get('args', []), aug_details.get('kwargs', {}))
        for aug in config['valid']['augmentation']
        for aug_class, aug_details in aug.items()
    ])

    # Initialize CacheDataset with training and validation data lists
    train_ds = CacheDataset(
        data=train_files, 
        transform=train_transforms, 
        cache_rate=config['data']['cache_rate'], 
        num_workers=config['data']['num_workers']
    )
    val_ds = CacheDataset(
        data=val_files, 
        transform=val_transforms, 
        cache_rate=0.8, 
        num_workers=config['data']['num_workers']
    )

    # Create DataLoader for training and validation
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers']
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=config['train']['val_batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers']
    )

    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    for epoch in range(config['train']['max_epochs']):
        print(f"Epoch {epoch + 1}/{config['train']['max_epochs']}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            wandb.log({"train/loss_step": loss.item()})
        
        # Average epoch loss and log
        epoch_loss /= step
        wandb.log({"train/loss_epoch": epoch_loss})

        # Validation
        if (epoch + 1) % config['train']['val_interval'] == 0:
            model.eval()
            with torch.no_grad():
                for val_data in tqdm(val_loader):
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = sliding_window_inference(val_inputs, config['valid']['roi_size'], config['valid']['sw_batch_size'], model)
                    val_outputs = post_pred(val_outputs)
                    val_labels = post_label(val_labels)
                    print(f"val_outputs shape: {val_outputs.shape}")
                    print(f"val_labels shape: {val_labels.shape}")

                    dice_metric(y_pred=val_outputs, y=val_labels)

                # Aggregate and log the dice metric
                metric = dice_metric.aggregate().item()
                dice_metric.reset()
                wandb.log({"val/dice_metric": metric})

                # Save best model
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    os.makedirs(config['log']['save_dir'], exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(config['log']['save_dir'], "best_metric_model.pth"))
                    print("Saved new best metric model")

                print(f"Epoch {epoch + 1}, Mean dice: {metric:.4f}, Best dice: {best_metric:.4f} at epoch {best_metric_epoch}")

    # Log best metric to W&B after training
    wandb.log({"best_dice_metric": best_metric, "best_metric_epoch": best_metric_epoch})

# Check for the main module
if __name__ == "__main__":
    main()
