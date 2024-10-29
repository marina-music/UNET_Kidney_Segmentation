import nibabel as nib
from utils.preprocess import preprocessing_counting_nonzero_slices
import numpy as np

def save_dataset(df, input):
    data_idx = []
    data_image_npz_array = []
    data_label_npz_array = []

    max_value = 0
    min_value = 1e9
    data_sum_value = 0
    for i in range(len(df)):
        img = nib.load(df['image'].iloc[i]).get_fdata(dtype=np.float32)
        lbl = nib.load(df['label'].iloc[i]).get_fdata(dtype=np.float32)

        # exclude different width and height data
        if img.shape[1] != 512 or img.shape[2] != 512:
            continue

        image, label, n = preprocessing_counting_nonzero_slices(img, lbl)

        data_idx.append((data_sum_value, n))
        max_value = max(max_value, n)
        min_value = min(min_value, n)
        data_sum_value += n

        image = (image * 255).astype(np.uint8)
        label = label.astype(np.uint8)

        data_image_npz_array.append(image)
        data_label_npz_array.append(label)

        concat_image = np.concatenate(data_image_npz_array, axis=0)
        concat_label = np.concatenate(data_label_npz_array, axis=0)

        if i%5==0:
            print(f"saved {i}th image")

    if input == 'train':
        np.savez_compressed("data/train_image_concat.npz", data=concat_image)
        np.savez_compressed("data/train_label_concat.npz", data=concat_label)
        np.savez_compressed("data/train_index_concat.npz", data=data_idx)
    elif input == 'valid':
        np.savez_compressed("data/valid_image_concat.npz", data=concat_image)
        np.savez_compressed("data/valid_label_concat.npz", data=concat_label)
        np.savez_compressed("data/valid_index_concat.npz", data=data_idx)
