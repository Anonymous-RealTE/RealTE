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
import torch.nn.functional as F

from accelerate.utils import ProjectConfiguration, set_seed


def get_img_dir_length(env):
    with env.begin(write=False) as txn:
        length = int(txn.get('num-samples'.encode()))
    return length

def norm(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


class LineText(data.Dataset):
    def __init__(self, training=True, data_root='path_to_data', datasets="all_eng_data/all_chn_data", use_fix_width=-1, vis_flag=False, vis_root=""):
        is_training = training
        self.vis_flag = vis_flag
        self.vis_root = vis_root
        self.is_training = is_training
        self.images = []
        self.gts = []
        sub_path = datasets.split('/') #os.listdir(data_root)
        
        self.image_processing = transforms.Compose(
                [

                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ]
            )
        self.glyph_processing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ]
            )
        self.reference_processing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        count = 0
        self.envs_path = []
        self.images = []
        self.gt = []

        self.images_pair = []
        self.gt_pair = []
        self.use_fix_width = use_fix_width
        
        for p in sub_path:
            sub_data_path = os.path.join(data_root, p)
            try:
                env = lmdb.open(str(sub_data_path), readonly=True, lock=False, readahead=False, meminit=False)
                length = get_img_dir_length(env)
                first_case = []
                
                for i in range(length):
                    image_key, label_key = "image-{}".format(str(i + 1).zfill(9)), "label-{}".format(str(i + 1).zfill(9))
                    img_name = str(image_key.encode())[2:-1]
                    self.envs_path.append(str(sub_data_path))
                    self.gt.append(label_key)
                    self.images.append(image_key)
            except:
                # paired data
                i_t_path = os.path.join(sub_data_path, "i_t.txt")

                with open(i_t_path) as fp:
                    lines = fp.readlines()
                 
                for line in lines:
                    name, text = line.strip().split(" ")
                    tgt_path = os.path.join(sub_data_path, "t_f", name)
                    self.images_pair.append(tgt_path)
                    self.gt_pair.append(text)
            

        print(len(self.images), len(self.images_pair))

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
    
    
    def process_input(self, image, text, reference, repeat_flag=False):

        c, h, w = image.shape
        c_ref, h_ref, w_ref = reference.shape
        
        new_h = 32
        max_w = 640
        new_w = int(new_h / h * w)
        if new_w <= 0:
            return None, None, None, None, None, None
        
        random_resized_ratio = 0 #1 #random.random()
        if random_resized_ratio > 0.5:
            new_w = int(random.choice(np.linspace(start = 0.5, stop = 1.5, num = 20)) * new_w)


        if new_w <= max_w:
            new_img = transforms.Resize((new_h, new_w))(image)#cv2.resize(image, (new_h, new_w))
            new_h_ref = 32
            new_w_ref = int(new_h_ref / h_ref * w_ref)
            new_reference = transforms.Resize((new_h_ref, new_w_ref))(reference)
        else:
            new_img = transforms.Resize((new_h, max_w))(image)
            new_w = max_w
            
            new_h_ref = 32
            new_w_ref = int(new_h_ref / h_ref * w_ref)
            new_reference = transforms.Resize((new_h_ref, new_w_ref))(reference)
        
        new_reference = new_reference[:, : ,:640]
        
        if repeat_flag:
            long_img, times = self.repeat_image(max_w, new_img)
            if long_img is not None:
                new_label = "".join(text * times)
                new_img = long_img
                new_h, new_w = new_img.shape[1:]
                h, w = new_h, new_w
                text = new_label
        
        stride = 4

        if 0 in new_img.shape or len(text) == 0:
            return None, None, None, None, None, None
        
        char_glyph = self.get_char_glyph(text, new_img)
        
        if char_glyph is None:
            return None, None, None, None, None, None
        
    
        padded = torch.zeros((3, new_h, max_w)) 
        padded[:, :new_img.shape[1], :new_img.shape[2]] = new_img[:, :new_img.shape[1], :new_img.shape[2]]

        mask_all = torch.zeros((new_h, max_w))
        side = 0
        if not self.is_training:
            side = 0
        
        mask_all[side:new_img.shape[1]-side, side:new_img.shape[2]-side] = 1

        bg_for_inpaint = padded * (1 - mask_all)
        
        mask_all[:new_img.shape[1], :new_img.shape[2]] = 1
       

        padded_reference = torch.zeros((3, new_h, 640))#torch.zeros((3, new_h, max_w))
        padded_reference[:, :new_reference.shape[1], :new_reference.shape[2]] = new_reference
        
        return padded, bg_for_inpaint, padded_reference, mask_all, char_glyph, new_img
    

    def vis(self, image, bg_for_inpaint, reference, mask_all, text, i, char_glyph):
        if self.vis_flag:
            if self.vis_root == "":
                self.vis_root = "vis"
        if not os.path.exists(self.vis_root):
            os.makedirs(self.vis_root)
        image_vis = Image.fromarray(((image / 2 + 0.5) * 255).detach().numpy().transpose(1,2,0).astype("uint8"))

        bg_for_inpaint_vis = Image.fromarray(((bg_for_inpaint / 2 + 0.5) * 255).detach().numpy().transpose(1,2,0).astype("uint8"))
        char_glyph_vis = Image.fromarray(((char_glyph / 2 + 0.5) * 255).detach().numpy().transpose(1,2,0).astype("uint8"))
      
       
        reference_vis = (norm(reference.clone().detach()).numpy() * 255).astype(np.uint8).transpose((1, 2, 0)) 

        reference_vis = np.ascontiguousarray(reference_vis)

        reference_vis = Image.fromarray(reference_vis.astype("uint8"))
      
        new_img = Image.new('RGB', (640, 32 * 4))
        new_img.paste(image_vis, (0, 0))
        new_img.paste(reference_vis, (0, 32))
        new_img.paste(char_glyph_vis, (0, 32 * 2))
        new_img.paste(bg_for_inpaint_vis, (0, 32 * 3))
        
        
        new_img.save('{}/{}_ref.jpg'.format(self.vis_root, i))


    def load_benchmark(self, i):
        img_path = self.images[i]
        image = imageio.imread(img_path, pilmode='RGB')
        gt = open(self.gts[i],'rb')
        all_word_char_boxes, all_words, all_polys = pickle.load(gt)
        return image, all_word_char_boxes, all_words, all_polys 
    

    def repeat_image(self, max_w, image):
        c, img_h, img_w = image.shape
        if img_w > max_w // 2:
            return None, 0
        times = max_w // img_w
        if times > 2:
            times = random.randint(1, times)
        new_img = torch.zeros((c, img_h, times * img_w))#Image.new('RGB', (times * img_w, 32))
        
        for i in range(times):
            #print(new_img[:, :, i * img_w: (i+1) * img_w].shape, image.shape)
            new_img[:, :, i * img_w: (i+1) * img_w] = image
        return new_img, times


    def get_char_glyph(self, text, resized_image):
        path = "dataloader/微软雅黑.ttf"
        font = ImageFont.truetype(path, size=32, encoding="utf-8")
        c, img_h, img_w = resized_image.shape
        
        chars_w, chars_h = font.getsize(text)
     
        if chars_w == 0: # or chars_w / img_w > 1.5 or img_w / chars_w > 1.5:
            return None
        
        temp_glyph = Image.new('RGB', (chars_w, chars_h))

        draw_2 = ImageDraw.Draw(temp_glyph)
        draw_2.text((0, 0), text, font=font, color=(255,0,255))
        glyph = self.glyph_processing(temp_glyph)
        #print(torch.max(glyph), torch.min(glyph))
        glyph = - glyph
        glyph = transforms.Resize((img_h, img_w))(glyph)
        
        padded = torch.zeros((3, 32, 640))
        padded[:, :img_h, :img_w] = glyph
        return padded

    
    def load_unpair_data_with_char(self, i, repeat_flag=False):
        from PIL import ImageOps, ImageChops
        sub_data_path = self.envs_path[i]
        image_key = self.images[i]
        label_key = self.gt[i]
        env = lmdb.open(str(sub_data_path), readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            imgbuf = txn.get(image_key.encode())  # image
            img_name = str(image_key.encode())[2:-1]#str(image_key.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            text = str(txn.get(label_key.encode()))[2:-1]
        image_np = np.array(Image.open(buf).convert("RGB"))

        if self.use_fix_width > 0:
            h, w, c = image_np.shape
            image_np = cv2.resize(image_np, (self.use_fix_width, 32), interpolation=cv2.INTER_CUBIC)

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
        
        reference = self.reference_processing(image_np)
        
        return new_img, reference, text

    def load_pair_data(self, i):
        img_path = self.images_pair[i]
        
        
        ref_path = img_path.replace("t_f", "i_s")
        image_np = imageio.imread(img_path)
        
        label = self.gt_pair[i]
        ref = imageio.imread(ref_path)
        
        if self.use_fix_width > 0:
            h, w, c = image_np.shape
            image_np = cv2.resize(image_np, (self.use_fix_width, 32), interpolation=cv2.INTER_CUBIC)
            ref = cv2.resize(ref, (self.use_fix_width, 32), interpolation=cv2.INTER_CUBIC)
        image = self.image_processing(image_np)

        c, h, w = image.shape

        new_h = 32
        max_w = 640
        new_w = int(new_h / h * w)
        
        if new_w <= max_w:
            new_img = transforms.Resize((new_h, new_w))(image)
        else:
            new_img = transforms.Resize((new_h, max_w))(image)
            new_w = max_w
    
        reference = self.reference_processing(ref)
        
        return new_img, reference, label



    def __getitem__(self, i):
        
        if i < len(self.images):
            is_pair = False
            image, reference_image, label = self.load_unpair_data_with_char(i)
            sub_data_path = None
        else:
            is_pair = True
            i -= len(self.images)
            image, reference_image, label = self.load_pair_data(i)

        if image is None:
            idx = random.randint(0, len(self.images) + len(self.images_pair) -1)
            return self.__getitem__(idx)


        crop_image, bg_for_inpaint, reference, mask_all, char_glyph, resized_image = self.process_input(image, label, reference_image, False)

        if crop_image is None:
            idx = random.randint(0, len(self.images)-1)
            return self.__getitem__(idx)
        
 
        if crop_image is None:
            idx = random.randint(0, len(self.images)-1)
            return self.__getitem__(idx)
        
        if self.vis_flag:
            self.vis(resized_image, bg_for_inpaint, reference, mask_all, label, i, char_glyph)
        

        return crop_image, bg_for_inpaint, mask_all, label, reference, char_glyph, resized_image, is_pair
  
