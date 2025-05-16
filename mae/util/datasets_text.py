# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import sys
sys.path.append(".")
sys.path.append("..")
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
import re
import six
import csv
import pickle
import PIL
from skimage.morphology import label as bwlabel
from skimage.measure import regionprops
from PIL import Image, ImageFont, ImageDraw
import random 
import numpy as np
import codecs
import cv2
import imageio
from scipy import io as sio
import torch.utils.data as data
import matplotlib.pyplot as plt
import string
import lmdb

alphabet = ""
use_customer_dictionary = "chn_cls_list.txt"
with open(use_customer_dictionary, 'rb') as fp:
    CTLABELS = pickle.load(fp)
for c in CTLABELS:
    alphabet += chr(c)
alphabet = alphabet+ "\n "
import torch.nn.functional as F

def get_img_dir_length(env):
    with env.begin(write=False) as txn:
        length = int(txn.get('num-samples'.encode()))
    return length


def polygon_area(poly):
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.

def sort_poly(points):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0]
        py1 = ps[0][1]
        px4 = ps[1][0]
        py4 = ps[1][1]
    else:
        px1 = ps[1][0]
        py1 = ps[1][1]
        px4 = ps[0][0]
        py4 = ps[0][1]
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0]
        py2 = ps[2][1]
        px3 = ps[3][0]
        py3 = ps[3][1]
    else:
        px2 = ps[3][0]
        py2 = ps[3][1]
        px3 = ps[2][0]
        py3 = ps[2][1]

    poly = np.array([[px1, py1], [px2, py2], [px3, py3], [px4, py4]])
    if polygon_area(poly) < 0:
        return poly
    else:
        return poly[(0, 3, 2, 1), :]
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1, delte=6):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / delte)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

class FuseImage(data.Dataset):
    def __init__(self, training=True, data_root='path_to_data', datasets="all_eng_data/all_chn_data"):
        is_training = training
        self.is_training = is_training
        self.images = []
        self.gts = []
        sub_path = datasets.split('/') #os.listdir(data_root)

        self.image_processing = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    #transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        self.glyph_processing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ]
            )
        self.reference_processing = transforms.Compose([
            transforms.RandomHorizontalFlip()
        ])
        count = 0
        self.envs_path = []

        self.first_cases = []
        self.images_pair = []
        self.gt_pair = []
        
        for p in sub_path:
            sub_data_path = os.path.join(data_root, p)
            if "pair" in p:
                label_text = sub_data_path + ".txt"
                with open(label_text) as fb:
                    lines = fb.readlines()
                
                for line in lines:
                    if len(line.strip().split('\t')) != 2:
                        continue
                    img_name, label = line.strip().split('\t')
                    new_img_name = img_name.replace('pair_chn_img_shengpi', p)
                    self.images_pair.append(os.path.join(data_root, new_img_name))
                    self.gt_pair.append(label)
                continue
                

            try:
                env = lmdb.open(str(sub_data_path), readonly=True, lock=False, readahead=False, meminit=False)
                length = get_img_dir_length(env)
                first_case = []
            
                for i in range(length):
                    image_key, label_key = "image-{}".format(str(i + 1).zfill(9)), "label-{}".format(str(i + 1).zfill(9))
                    img_name = str(image_key.encode())[2:-1]
                    if os.path.exists(os.path.join(sub_data_path + "_gts_0201", img_name + '_anno.pkl')):    
                        gt_path = os.path.join(sub_data_path + "_gts_0201", img_name + '_anno.pkl')
                        self.images.append(image_key)
                        self.gts.append(gt_path)
                        self.envs_path.append(str(sub_data_path))
                        if len(first_case) < 10:
                            first_case.append(len(self.images))

            except:
                env = os.listdir(sub_data_path)
                length = len(env)
                for i in range(length):
                    f = env[i]
                    if os.path.exists(os.path.join(sub_data_path + "_gts_0201", f.split('.')[0] + '_anno.pkl')):    
                        self.images.append(os.path.join(sub_data_path, f))
                        self.gts.append(os.path.join(sub_data_path + "_gts_0201", f.split('.')[0]+'_anno.pkl'))
                        self.envs_path.append(None)

        print(len(self.images) + len(self.images_pair))

        
    def convert_to_poly(self, word_boxes):
        new_word_boxes = []
        for word_box in word_boxes:
            temp_box = np.zeros((4,2))
            temp_box[0,0] = word_box[0]
            temp_box[1,0] = word_box[2]
            temp_box[2,0] = word_box[2]
            temp_box[3,0] = word_box[0]

            temp_box[0,1] = word_box[1]
            temp_box[1,1] = word_box[1]
            temp_box[2,1] = word_box[3]
            temp_box[3,1] = word_box[3]
            new_word_boxes.append(temp_box)
        return new_word_boxes

    def __len__(self):
        return len(self.images) + len(self.images_pair)
    



    def vis(self, image):
        image_vis = Image.fromarray(((image / 2 + 0.5) * 255).detach().numpy().transpose(1,2,0).astype("uint8"))
        image_vis.save('outputs/eval_gt/{}.jpg'.format(i))
       

    
    def load_unpair_data_with_char(self, i, repeat_flag=False):
        
        sub_data_path = self.envs_path[i]
        if sub_data_path is not None:
            image_key = self.images[i]
            
            env = lmdb.open(str(sub_data_path), readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin(write=False) as txn:
                imgbuf = txn.get(image_key.encode())  # image
                img_name = str(image_key.encode())[2:-1]#str(image_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
            image_np = np.array(Image.open(buf).convert("RGB"))
            #Image.open(buf).convert("RGB").save("outputs/eval_gt/{}_ori.jpg".format(i))
        else:
            image_np = np.array(Image.open(self.images[i]).convert("RGB"))


        image = self.image_processing(image_np)

                

        
        c, h, w = image.shape
        new_h = 32
        max_w = 640
        new_w = int(new_h / h * w)
        
        if new_w <= max_w:
            new_img = transforms.Resize((new_h, new_w))(image)#cv2.resize(image, (new_h, new_w))
        else:
            new_img = transforms.Resize((new_h, max_w))(image)
            new_w = max_w

        return new_img

    def load_pair_data(self, i):
        img_path = self.images_pair[i]

        image = imageio.imread(img_path)
        image = self.image_processing(image)
        
        c, h, w = image.shape
        new_h = 32
        max_w = 640
        new_w = int(new_h / h * w)
        
        if new_w <= max_w:
            new_img = transforms.Resize((new_h, new_w))(image)#cv2.resize(image, (new_h, new_w))
        else:
            new_img = transforms.Resize((new_h, max_w))(image)
            new_w = max_w
        return new_img

    def __getitem__(self, i):
        
        if i < len(self.images_pair):
            image = self.load_pair_data(i)
            sub_data_path = None
        else:
            i = i - len(self.images_pair)
            image = self.load_unpair_data_with_char(i)

        if image is None:
            idx = random.randint(0, len(self.images)-1)
            return self.__getitem__(idx)
        padded = torch.zeros((3, 32, 640))
        padded[:, :image.shape[1], :image.shape[2]] = image
        return padded
  




def build_dataset(is_train, args):
    #transform = build_transform(is_train, args)

    root = args.data_path
    dataset = FuseImage(True, root, args.datasets)


    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    
    return transforms.Compose(t)
