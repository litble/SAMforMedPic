import numpy as np
import json
import torch
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from torchvision import transforms
from matplotlib import pyplot as plt

origin_base_dir = '../RawData/'
dealed_base_dir = './data/'

# image reinforce
def adjustMethod1(data_resampled, w_width=400, w_center=40):
    val_min = w_center - (w_width / 2)
    val_max = w_center + (w_width / 2)

    data_adjusted = data_resampled.copy()
    data_adjusted[data_resampled < val_min] = val_min
    data_adjusted[data_resampled > val_max] = val_max

    return data_adjusted

# prework for training
def dataPrework(paths, save_path):
    labels = []
    cnt = 0

    for path in paths:
        img_path = origin_base_dir + path['image']
        label_path = origin_base_dir + path['label']
        img = nib.load(img_path)
        label = nib.load(label_path)
        width, height, channel = img.dataobj.shape

        for slice_index in range(channel):
            image_slice_data = adjustMethod1(img.get_fdata()[:, :, slice_index])
            label_slice_data = label.get_fdata()[:, :, slice_index]
            for i in range(1, 14):
                mask = (label_slice_data == i)
                if(not np.any(mask)):
                    continue
                img_with_mask = np.dstack((image_slice_data, mask.astype(image_slice_data.dtype)))

                cnt += 1
                np.save(save_path + str(cnt)+'.npy', img_with_mask)
                labels.append(i)
    
    np.save(save_path + 'label.npy', np.array(labels))

if __name__ == '__main__':
    with open('../dataset_0.json', 'r') as f:
        data = json.load(f)
        training_path = data['training']
        validation_path = data['validation']

    dataPrework(training_path, dealed_base_dir+'training/')
    dataPrework(validation_path, dealed_base_dir+'validation/')