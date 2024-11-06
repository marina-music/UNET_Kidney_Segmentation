import os

def main():
    data_dir = "C:\\Users\\Marina.VALVE\kits23\\rough_segmentations_50"
    output_dir = "C:\\Users\\Marina.VALVE\\kits23\\50_images_labelled"

    # Loop through each case directory in data_dir
    for case_dir in os.listdir(data_dir):
        case_path = os.path.join(data_dir, case_dir)
        # Ensure it's a directory and contains the "imaging.nii" file
        imaging_file = os.path.join(case_path, "imaging.nii.gz")
        # Construct the new filename with the case number included
        new_filename = f"imaging_{case_dir}.nii.gz"
        dest_path = os.path.join(output_dir, new_filename)
        # Rename (move) the file to the output directory with the new name
        os.rename(imaging_file, dest_path)
        print(f"Moved and renamed {imaging_file} to {dest_path}")

main()