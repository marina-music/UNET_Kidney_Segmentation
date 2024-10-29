import numpy as np

def onehot_encoding(label, num_classes):
    shape = label.shape
    onehot = np.zeros((num_classes, shape[0], shape[1]))

    for i in range(num_classes):
        indices = np.where(label == i)
        onehot[i][indices] = 1.0

    return onehot

def window_CT(slice, min=-260, max=340):
  # kidney HU = +20 to +45
  # best visualize = -260 to +340
  sub=abs(min)
  diff=abs(min-max)

  img=slice+sub
  img[img<=0]=0   # min normalization
  img[img>diff]=diff  # max normalization
  img=img/diff

  return img


def preprocessing_counting_nonzero_slices(img, lbl):
    # rotate 180
    img = np.rot90(img, k=2, axes=(1, 2))
    lbl = np.rot90(lbl, k=2, axes=(1, 2))

    # window CT
    img = np.asarray([window_CT(_) for _ in img])

    # 4 label ->  2 label
    # 0: backgroud
    # 1: kidney
    # 2: tumor
    # 3: cyst
    # =>
    # 0: background
    # 1: left kidney
    # 2 : right kidney
    left_lbl = lbl[:, :, :256]
    left_lbl[left_lbl == 2] = 1
    left_lbl[left_lbl == 3] = 1

    right_lbl = lbl[:, :, 256:]
    right_lbl[right_lbl == 1] = 2
    right_lbl[right_lbl == 3] = 2

    lbl[:, :, :256] = left_lbl
    lbl[:, :, 256:] = right_lbl

    # only non-zero slices
    class_counts = np.sum(lbl, axis=(1, 2))
    nof_nonzero = np.count_nonzero(class_counts)

    _img = []
    _lbl = []
    for i, count in enumerate(class_counts):
        if count > 0:
            _img.append(img[i])
            _lbl.append(lbl[i])

    _img = np.asarray(_img)
    _lbl = np.asarray(_lbl)

    return _img, _lbl, nof_nonzero
