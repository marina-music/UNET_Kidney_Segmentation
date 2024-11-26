import torch
import yaml
import glob
import numpy
import importlib
import nrrd
from monai.transforms import Compose, AsDiscrete
from monai.data import DataLoader, CacheDataset, MetaTensor
from monai.inferers import sliding_window_inference
import nibabel as nib
import matplotlib.pyplot as plt
from train import load_model_from_config

# Load configuration
with open("config_template_inference.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_transform(class_path, args, kwargs):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    transform_class = getattr(module, class_name)
    return transform_class(*args, **kwargs)
    # Handle `LoadImaged` case specifically to exclude the `label` key
    """if class_name == "LoadImaged" and "keys" in kwargs:
        # Exclude 'label' key if it exists
        keys = kwargs["keys"]
        kwargs["keys"] = [key for key in keys if key != "label"]"""




def main():
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_config(config)
    model.load_state_dict(
        torch.load("C:\\Users\\Marina.VALVE\\GitHub\\UNET_Kidney_Segmentation\\weights\\best_metric_model.pth"))
    model.eval()

    # Prepare transforms and data
    val_transforms = Compose([
        load_transform(aug_class, aug_details.get('args', []), aug_details.get('kwargs', {}))
        for aug in config['valid']['augmentation']
        for aug_class, aug_details in aug.items()
    ])
    #print(config['valid']['augmentation'])
    #print(val_transforms)
    data = [{"image": img_path} for img_path in glob.glob("C:\\Users\\Marina.VALVE\\kits23\\7_cases_inference\\images\\*.nii")]
    #print(glob.glob("C:\\Users\\Marina.VALVE\\kits23\\7_cases_inference\\images\\*.nii"))

    # Create Dataset and DataLoader
    inference_ds = CacheDataset(data=data, transform=val_transforms, cache_rate=1.0, num_workers=2)
    inference_loader = DataLoader(inference_ds, batch_size=1, num_workers=2)
    #print(len(inference_ds))

    # Perform inference
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    #batch_counter = 0  # To track output file numbering

    for idx, batch_data in enumerate(inference_loader):
        inputs = batch_data["image"].to(device)

        # Run sliding window inference
        outputs = sliding_window_inference(
            inputs, config['valid']['roi_size'], config['valid']['sw_batch_size'], model
        )

        # Post-process predictions
        #outputs = post_pred(outputs)
        print(f"Output shape after inference: {outputs.shape}")  # Should be [1, num_classes, depth, height, width]

        # Take argmax across class dimension (dim=1)
        output_mask = outputs.argmax(dim=1).cpu().numpy()  # Shape: [1, depth, height, width]
        output_mask = output_mask[0]  # Remove batch dimension
        num_1s = numpy.sum(output_mask)
        # Ensure dtype is np.uint8
        output_mask = output_mask.astype(numpy.uint8)
        nrrd.write(f"C:/Users/Marina.VALVE/kits23/50_cases_inference_output/output_{idx}.nrrd",output_mask
        )
        img = output_mask[:,:,output_mask.shape[2]//2]

        ##print(f"Unique values in output_mask: {numpy.unique(output_mask)}")
        #output_mask = torch.from_numpy(output_mask)
        # Extract affine from the input metadata
        """nib.save(
            nib.Nifti1Image(output_mask, affine=affine),
            f"C:/Users/Marina.VALVE/kits23/7_cases_inference_output/output_{idx}.nii.gz"
        )"""



        """"# Save the mask as a NIfTI file
        nib.save(
            nib.Nifti1Image(output_mask, affine=affine),
            f"C:/Users/Marina.VALVE/kits23/7_cases_inference_output/output_{idx}.nii"
        )
        print(f"Saved: C:/Users/Marina.VALVE/kits23/7_cases_inference_output/output_{idx}.nii")"""



    """for batch_data in inference_loader:
        inputs = batch_data["image"].to(device)
        outputs = sliding_window_inference(inputs, config['valid']['roi_size'], config['valid']['sw_batch_size'], model)
        outputs = post_pred(outputs)
        print(type(outputs), type(outputs[0]) if isinstance(outputs, list) else None)
        print(type(outputs))
        print(outputs.shape)
        # Save predictions
        for i, output in enumerate(outputs):
            output_mask = output.argmax(dim=0).cpu().numpy()
            output_mask = output_mask.astype(numpy.int32)
            affine = numpy.eye(4)
            nib.save(nib.Nifti1Image(output_mask, affine=affine), f"C:/Users/Marina.VALVE/kits23/7_cases_inference_output/output_{i}.nii")"""



if __name__ == '__main__':
    main()


