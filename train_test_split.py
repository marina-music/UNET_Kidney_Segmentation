import os
import sklearn
from sklearn.model_selection import train_test_split

def main():
    data_dir = "C:\\Users\\Marina.VALVE\\kits23\\50_ground_truths"
    output_dir = "C:\\Users\\Marina.VALVE\\kits23\\train_test_split_50"


    # Get a list of all .nrrd files in data_dir
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".nrrd")]

    # Split the files into train and test sets (80-20 split)
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    # Move files with "train" and "test" added to the filename
    for file_list, subset in zip([train_files, test_files], ["train", "test"]):
        for filename in file_list:
            # Define original and new file paths
            src_path = os.path.join(data_dir, filename)
            new_filename = f"{filename.split('.')[0]}_{subset}.nrrd"
            dest_path = os.path.join(output_dir, new_filename)

            # Rename (move) the file to the new path
            os.rename(src_path, dest_path)
            print(f"Moved {src_path} to {dest_path}")

main()