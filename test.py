from segment_anything import sam_model_registry, SamPredictor
from collections import defaultdict
from utils import adjustMethod1
from classifier.model import AlexNet

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import argparse
import random
import torch
import cv2
import sys
import json

flag = False
 
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
   
def show_points(coords, labels, ax, marker_size=75):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white',
               linewidth=1.25)
 
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
 
def show_label(box, ax, label):
    center_x = (box[0] + box[2]) // 2 - 3
    center_y = (box[1] + box[3]) // 2 + 3
    ax.text(center_x, center_y, str(label), fontsize=14, color='r')

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--img_path",
    type = str,
    default = './RawData/Training/img/img0001.nii.gz',
    help = "path to the image file",
)

parser.add_argument(
    "-l",
    "--label_path",
    type = str,
    default = './RawData/Training/label/label0001.nii.gz',
    help = "path to the label file",
)

parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="assets/",
    help="path to the segmentation folder",
)

parser.add_argument(
    "-a",
    "--all_slice",
    action = 'store_true',
    default = False,
    help = "get all alice",
)

parser.add_argument(
    "-t",
    "--slice_type",
    type = int,
    default = 2,
    help = "how to make the slice",
)

parser.add_argument(
    "-s",
    "--slice_index",
    type = int,
    default = 100,
    help = "slice_index",
)

parser.add_argument(
    "-v",
    "--visible",
    action = 'store_true',
    help = "Whether visualization is needed",
)

parser.add_argument(
    "--device", 
    type=str, 
    default="cuda", 
    help="device"
)

parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="./models/sam_vit_b_01ec64.pth",
    help="path to the trained model",
)

parser.add_argument(
    "--calssifier_path",
    type=str,
    default="./models/AlexNet_24.pth",
    help="path to the classifier model",
)

parser.add_argument(
    "-c",
    "--classify",
    action = 'store_true',
    default = True,
    help = "Whether classifier is needed",
)

args = parser.parse_args()

img = nib.load(args.img_path)
lab = nib.load(args.label_path)

width, height, channel = img.dataobj.shape

sam_checkpoint = args.checkpoint
device = args.device
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# load classifier model
if(args.classify):
    classifier = AlexNet(num_classes = 13, init_weights = True)
    classifier.load_state_dict(torch.load(args.calssifier_path))
    classifier.to(device)

def get_box(points):
    if not points:
        return None, None, None, None 
    
    min_x, min_y = points[0]
    max_x, max_y = points[0]

    for x, y in points:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    min_x = max(min_x - 1, 0)
    min_y = max(min_y - 1, 0)
    max_x = min(max_x + 1, 511)
    max_y = min(max_y + 1, 511)

    return [min_x, min_y, max_x, max_y]

def get_single_point(points): 
    return random.sample(points, 1)

def get_multi_points(points): 
    if len(points) < 3:
        return points
    return random.sample(points, 3)

def get_dice(mask, std_list):
    h, w = mask.shape[-2:]
    my_list = []
    for i in range(h):
        for j in range(w):
            if mask[0][i][j] :
                my_list.append((j,i))
    intersection = len(set(my_list) & set(std_list))
    dice_score = (2.0 * intersection) / (len(my_list) + len(std_list))
    return dice_score

def segment_it(image, label, classify=False):
    predictor.set_image(image)
    label_sets = defaultdict(list)
    width, height = label.shape

    for i in range(width):
        for j in range(height):
            if 1 <= label[i][j] <= 13:
                label_sets[label[i][j]].append((j, i))

    global fig
    global ax

    if args.visible :
        ax[0][0].imshow(label)
        ax[0][1].imshow(image)
        ax[1][0].imshow(image)
        ax[1][1].imshow(image)
        ax[0][0].set_title("std_segmentation")
        ax[0][1].set_title("point_segmentation")
        ax[1][0].set_title("points_segmentation")
        ax[1][1].set_title("box_segmentation")

    def get_dice(mask, std_list):
        h, w = mask.shape[-2:]
        my_list = []
        for i in range(h):
            for j in range(w):
                if mask[0][i][j] :
                    my_list.append((j,i))
        intersection = len(set(my_list) & set(std_list))
        dice_score = (2.0 * intersection) / (len(my_list) + len(std_list))
        return dice_score

    mdice1 = 0
    tot1 = 0
    for num, points in label_sets.items():
        prompt_point = np.array(get_single_point(points))
        prompt_label = np.array([1])
        mask, scores, logits = predictor.predict(
            point_coords = prompt_point,
            point_labels = prompt_label,
            multimask_output = False,
        )
        dice = get_dice(mask, points)
        mdice1 = mdice1 + dice
        tot1  = tot1 + 1

        if(classify):
            mask_1d = mask[0]
            img_with_mask = np.dstack((image[:,:,0], mask_1d.astype(image.dtype)))
            classifier_input = torch.Tensor(img_with_mask.transpose(2, 0, 1))
            classifier_output = classifier(classifier_input.to(device))
            predict_y = torch.max(classifier_output, dim=0)[1]
        
        if(args.visible):
            show_mask(mask, ax[0][1], True)
            show_points(prompt_point, prompt_label, ax[0][1])
            if(classify):
                show_label(np.array(get_box(points)), ax[0][1], int(predict_y)+1)
    if tot1 == 0 :
        return -1, -1, -1
    mdice1 = mdice1 / tot1
    print(f"mdice of single point prompt is {mdice1}")

    mdice2 = 0
    tot2 = 0
    for num, points in label_sets.items():
        prompt_points = np.array(get_multi_points(points))
        prompt_labels = np.array([1] * prompt_points.shape[0])
        mask, scores, logits = predictor.predict(
            point_coords = prompt_points,
            point_labels = prompt_labels,
            multimask_output = False,
        )
        dice = get_dice(mask, points)
        mdice2 = mdice2 + dice
        tot2  = tot2 + 1

        if(classify):
            mask_1d = mask[0]
            img_with_mask = np.dstack((image[:,:,0], mask_1d.astype(image.dtype)))
            classifier_input = torch.Tensor(img_with_mask.transpose(2, 0, 1))
            classifier_output = classifier(classifier_input.to(device))
            predict_y = torch.max(classifier_output, dim=0)[1]

        if(args.visible):
            show_mask(mask, ax[1][0], True)
            show_points(prompt_points,prompt_labels, ax[1][0])
            if(classify):
                show_label(np.array(get_box(points)), ax[1][0], int(predict_y)+1)

    mdice2 = mdice2 / tot2
    print(f"mdice of multiple prompts is {mdice2}")   

    mdice3 = 0
    tot3 = 0
    for num, points in label_sets.items():
        prompt_box = np.array(get_box(points))
        mask, scores, logits = predictor.predict(
            box = prompt_box,
            multimask_output = False,
        )
        dice = get_dice(mask, points)
        mdice3 = mdice3 + dice
        tot3  = tot3 + 1

        if(classify):
            mask_1d = mask[0]
            img_with_mask = np.dstack((image[:,:,0], mask_1d.astype(image.dtype)))
            classifier_input = torch.Tensor(img_with_mask.transpose(2, 0, 1))
            classifier_output = classifier(classifier_input.to(device))
            predict_y = torch.max(classifier_output, dim=0)[1]
            #print(f"True category : {num}, Recognized category: {predict_y+1}")
        
        if(args.visible):
            show_mask(mask, ax[1][1], True)
            show_box(prompt_box, ax[1][1])
            if(classify):
                show_label(prompt_box, ax[0][0], int(num))
                show_label(prompt_box, ax[1][1], int(predict_y)+1)

    mdice3 = mdice3 / tot3
    print(f"mdice of box prompt is {mdice3}")

    if(args.visible):
        plt.axis('on')
        plt.savefig('slice/slice.png') # savefig() should be front of show()
        plt.show()
    
    return mdice1, mdice2, mdice3

def get_slice(data, dim, slice_index):
    if (dim == 0):
        return data.get_fdata()[slice_index, :, :]
    elif (dim == 1):
        return data.get_fdata()[:, slice_index, :]
    elif (dim == 2):
        return data.get_fdata()[:, :, slice_index]

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

if args.all_slice == True :
    totdice1 = 0 
    totdice2 = 0
    totdice3 = 0
    totslice = width + height + channel
    for i in range(width):
        image = adjustMethod1(get_slice(img, 0, i))
        label = get_slice(lab, 0, i)
        output_file_img = 'slice/img.png'
        plt.imsave(output_file_img, image, cmap='gray')
        image = cv2.imread(output_file_img)
        rs1, rs2, rs3 = segment_it(image, label)
        if rs1 == -1 :
            totslice -= 1
        else :
            print(f"({i},{height},{channel}")
            totdice1 += rs1
            totdice2 += rs2
            totdice3 += rs3

    for i in range(height):
        image = adjustMethod1(get_slice(img, 1, i))
        label = get_slice(lab, 1, i)
        output_file_img = 'slice/img.png'
        plt.imsave(output_file_img, image, cmap='gray')
        image = cv2.imread(output_file_img)
        rs1, rs2, rs3 = segment_it(image, label)
        if rs1 == -1 :
            totslice -= 1
        else :
            print(f"({width},{i},{channel}")
            totdice1 += rs1
            totdice2 += rs2
            totdice3 += rs3

    for i in range(channel):
        image = adjustMethod1(get_slice(img, 2, i))
        label = get_slice(lab, 2, i)
        output_file_img = 'slice/img.png'
        plt.imsave(output_file_img, image, cmap='gray')
        image = cv2.imread(output_file_img)
        rs1, rs2, rs3 = segment_it(image, label, classify=args.classify)
        if rs1 == -1 :
            totslice -= 1
        else :
            print(f"({width},{height},{i}")
            totdice1 += rs1
            totdice2 += rs2
            totdice3 += rs3

    mdice1 = totdice1 / totslice
    mdice2 = totdice2 / totslice
    mdice3 = totdice3 / totslice

    print(f"single point prompt mdice is {mdice1}")
    print(f"multiple points prompt mdice is {mdice2}")
    print(f"box prompt mdice is {mdice3}")

elif args.all_slice == False :
    slice_type = args.slice_type
    slice_index = args.slice_index
    image = adjustMethod1(get_slice(img, slice_type, slice_index))
    label = get_slice(lab, slice_type, slice_index)

    output_file_img = 'slice/img.png'
    plt.imsave(output_file_img, image, cmap='gray')
    image = cv2.imread(output_file_img)

    rs1, rs2, rs3 = segment_it(image, label, classify=args.classify)