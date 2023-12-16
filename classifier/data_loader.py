import numpy as np
import json
import torch
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D
from torchvision import transforms
from matplotlib import pyplot as plt

base_dir = '../RawData/'

# number of workers to load data
def get_dataloader_workers():
    return 4

# draw images to varify the training dataset
def show_images(imgs, num_rows, num_cols, labels, scale=1.5): 
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # tensor image
            ax.imshow(img.numpy())
        else:
            # PIL image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(labels[i].item())
    return axes

class CTDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        super(CTDataset, self).__init__()
        self.dir = dir
        self.labels = np.load(self.dir + 'label.npy')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = np.load(self.dir + str(index+1) + '.npy')
        img = img.transpose(2, 0, 1)
        img = torch.Tensor(img)
        #img = img.unsqueeze(0) # add a dimension
        return img, self.labels[index]-1

def getDataLoader(batch_size):
    train_dataset = CTDataset('./data/training/')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=get_dataloader_workers())
    validate_dataset = CTDataset('./data/validation/')
    validate_dataloader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=True,
        num_workers=get_dataloader_workers())
    return train_dataloader, validate_dataloader

if __name__ == '__main__':
    X, y = next(iter(train_dataloader))
    show_images(X.reshape(batch_size, 512, 512), 2, 6, y)
    plt.show()