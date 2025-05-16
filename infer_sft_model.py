import os
import lmdb
import six
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import warnings
warnings.filterwarnings("ignore")
import torch
import argparse
import sys
sys.path.append('.')
sys.path.append('mae')
sys.path.append('xformers')
import cv2
import numpy as np
import random
import codecs
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
from src.model import TextEditor
from accelerate.utils import set_seed
import editdistance as ed
from utils.parser import DefaultParser

vis_height = 32
vis_width = 128

path = "../dataloader/微软雅黑.ttf"
font = ImageFont.truetype(path, size=64, encoding="utf-8")

image_processing = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ]
            )
glyph_processing = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ]
            )
reference_processing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def norm(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def vis(image, reference, char_glyph, mask_all, output):
    
    image = image.cpu()
    reference = reference.cpu()
    char_glyph = char_glyph.cpu()
    image_vis = Image.fromarray(((image / 2 + 0.5) * 255).detach().numpy().transpose(1,2,0).astype("uint8"))  

    ref_vis = Image.fromarray((norm(reference).cpu().detach().numpy() * 255).astype(np.uint8).transpose(1, 2, 0).astype("uint8"))
    
    char_glyph_vis = Image.fromarray(((char_glyph / 2 + 0.5) * 255).detach().numpy().transpose(1,2,0).astype("uint8"))
    mask_all_vis = Image.fromarray(((mask_all * 255).cpu().detach().numpy().astype("uint8")))
    new_im = Image.new('RGB', (vis_width, 5 * vis_height))
    new_im.paste(image_vis, (0, 0))
    new_im.paste(ref_vis, (0, vis_height))
    new_im.paste(char_glyph_vis, (0, 2 * vis_height))
    new_im.paste(mask_all_vis, (0, 3 * vis_height))
    
    output_vis = Image.fromarray((output * 255).astype("uint8"))

    new_im.paste(output_vis, (0, 4 * vis_height))

    return new_im


def get_char_glyph(text, resized_image):
    path = "微软雅黑.ttf"
    font = ImageFont.truetype(path, size=vis_height, encoding="utf-8")
    c, img_h, img_w = resized_image.shape

    chars_w, chars_h = font.getsize(text)
       
    temp_glyph = Image.new('RGB', (chars_w , chars_h))

    draw_2 = ImageDraw.Draw(temp_glyph)
    draw_2.text((0, 0), text, font=font, color=(255,0,255))
    glyph = glyph_processing(temp_glyph)
    glyph = - glyph
    glyph = transforms.Resize((img_h, img_w))(glyph)

    padded = torch.zeros((3, vis_height, vis_width))
    padded[:, :img_h, :glyph.shape[-1]] = glyph

    return padded, int(img_h / chars_h * chars_w)

 
def modify_line(new, image, editor=None, output_path=None, weight_dtype=torch.float16, guidance_scale=7.5, fobj=None, imgbasename=None):

    def prepare_input(input_text, image):

        img = image_processing(image)
        ori_img_h, ori_img_w = img.shape[1:]

        new_h = vis_height
        max_w = vis_width
        
        reference = reference_processing(image)
        ref_new_h = vis_height
        ref_new_w = min(int(ref_new_h / reference.shape[1] * reference.shape[2]), max_w)
        
        new_img = transforms.Resize((ref_new_h, ref_new_w))(img)
        
        reference = transforms.Resize((ref_new_h, ref_new_w))(reference)
        mask_all = torch.zeros((vis_height, vis_width))
        mask_all[:new_img.shape[1], :new_img.shape[2]] = 1

        char_glyph, _ = get_char_glyph(input_text, new_img)

        new_img = transforms.Resize((ref_new_h, ref_new_w))(new_img)

        padded_img = torch.zeros((3, vis_height, vis_width))
        padded_img[:, :, :ref_new_w] = new_img

        padded_ref = torch.zeros((3, 32, 640))
        padded_ref[:, :, :reference.shape[-1]] = reference

        bg_for_inpaint = torch.zeros((3, vis_height, vis_width)) 
        tuples = [new_h, ref_new_w, ori_img_h, ori_img_w]
        return [padded_img, padded_ref, mask_all, bg_for_inpaint, char_glyph, input_text, tuples]
    
    line_img_list = []
    all_words = []
    mid_results = []
    ret = prepare_input(new, image)

    padded_img, padded_ref, mask_all, bg_for_inpaint, char_glyph, input_text, tuples = ret
    padded_img = padded_img.unsqueeze(0).cuda().to(weight_dtype)
    padded_ref = padded_ref.unsqueeze(0).cuda().to(weight_dtype)
    mask_all = mask_all.unsqueeze(0).unsqueeze(0).cuda().to(weight_dtype)
    bg_for_inpaint = bg_for_inpaint.unsqueeze(0).cuda().to(weight_dtype)
    char_glyph_4 = char_glyph.unsqueeze(0).cuda().to(weight_dtype)

    
    image, torch_image = editor.inference_simple(bg_for_inpaint, padded_ref, mask_all, char_glyph_4, height=vis_height, width=vis_width, num_inference_steps=20, guidance_scale=guidance_scale, ratio=0.5)
        
    new_h, new_w, ori_img_h, ori_img_w = tuples

    line_img = vis(bg_for_inpaint[0], padded_ref[0], char_glyph_4[0], mask_all.cpu()[0][0], image[0])

    new_w_ = int(torch.sum(mask_all[0][0][0]))
    cropped_image = Image.fromarray((image[0][:new_h, :new_w_] * 255).astype(np.uint8))
    line_img_list = cropped_image
    mid_results = line_img
    
    return line_img_list, mid_results
        





def main(args, config):
    editor = TextEditor(args.model_path, args.weight_dtype, is_training=False)
    editor.unet.enable_xformers_memory_efficient_attention()

    image_root = args.image_root
    outdir = args.output_path
    template_path = args.template_path
    tags_file = os.path.join(outdir, "generated_image.txt")
    

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(os.path.join(outdir, 'gen_images')):
        os.makedirs(os.path.join(outdir, 'gen_images'))
    if not os.path.exists(os.path.join(outdir, 'intermediate_res')):
        os.makedirs(os.path.join(outdir, 'intermediate_res'))
    if os.path.exists(tags_file):
        os.remove(tags_file) 

    with open(template_path, 'r') as f:
        txt_lines = f.readlines()
    
    text_dict = {}
    for txt_line in txt_lines:
        parts = txt_line.strip().split(' ')
        name, text = parts
        text_dict[name] = text

    
    for imgbasename in os.listdir(image_root):
        image_path = os.path.join(image_root, imgbasename)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image.resize((128, 32)))
        
        new = text_dict[imgbasename]

        res_img, mid_result = modify_line(new, image, editor, outdir, args.weight_dtype, guidance_scale=args.guidance_scale, imgbasename=imgbasename)
            
        lineimgname = os.path.join(outdir, 'gen_images', imgbasename)
        res_img.save(lineimgname)

        midlineimgname = os.path.join(outdir, 'intermediate_res', imgbasename)
        mid_result.save(midlineimgname)
 
        with codecs.open(tags_file, 'a+', encoding='utf-8') as fobj:
            fobj.write('{}/{}\t{}\n'.format("line", os.path.join('{}'.format(imgbasename)), new))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="demo")
    parser.add_argument('--image_root', type=str, default='datasets/evaluation/Tamper-Scene/i_s')
    parser.add_argument('--template_path', type=str, default='datasets/evaluation/Tamper-Scene/i_t.txt')
    parser.add_argument('--output_path', type=str, default='test/eval_res/Tamper_sft/',help='save image')
    parser.add_argument('--model_path', type=str, default='exp/final_ckpt/',help='model path')
    parser.add_argument('--weight_dtype', default=torch.float16)
    parser.add_argument('--guidance_scale', default=7.5)
    parser.add_argument('--seed', default=0)
    args = parser.parse_args()
    set_seed(int(args.seed))
    config = DefaultParser()
    args = parser.parse_args()
    main(args, config)

