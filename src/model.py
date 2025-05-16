import sys
from PIL import Image, ImageFont, ImageDraw
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler, AutoencoderKL
from .unet_2d_condition_local import UNet2DConditionModel_with_patch_local_inpaint
from tqdm.auto import tqdm
import cv2
import os
import random
import numpy as np
from PIL import Image
from torch.autograd import Variable
import inspect
from torch.cuda.amp import autocast as autocast, GradScaler
from skimage.morphology import label as bwlabel
from skimage.measure import regionprops
import matplotlib.pyplot as plt



class TextEditor(nn.Module):
    def __init__(self, model_path, weight_dtype=None, is_training=True):
        super().__init__()

        unet = UNet2DConditionModel_with_patch_local_inpaint.from_pretrained(model_path, subfolder="unet",low_cpu_mem_usage=False)

        scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
        self.weight_dtype = weight_dtype

        self.unet = unet.cuda()
        self.scheduler = scheduler

        ref_ckpt = os.path.join(model_path, "checkpoint-scene-text.pth")
        checkpoint = torch.load(ref_ckpt, map_location='cpu')
        self.unet.referenceEncoder.load_state_dict(checkpoint['model'], strict=False)
        
        if not is_training:
            self.unet = self.unet.to(weight_dtype)


    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )
        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")


    def forward(self, images, bg_for_inpaint, stroke_seg_1, mask_4, char_glyph, ratio=0, optim_step=0):
        
        latents = images.to(self.weight_dtype)
        latent_bg_for_inpaint = bg_for_inpaint.to(latents.device).to(self.weight_dtype)

        choice = random.random()
        stroke_seg_1_no = None
        con_flag = True
        if choice < 0.1:
            stroke_seg_1_no = torch.zeros_like(stroke_seg_1).to(latents.device).to(self.weight_dtype)
            stroke_seg_1 = stroke_seg_1.to(latents.device).to(self.weight_dtype)#None
            con_flag = False
        else:
            stroke_seg_1 = stroke_seg_1.to(latents.device).to(self.weight_dtype)
        
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
     
        memory_2, style, no_style, mask_for_loss = self.unet.get_input_v3(latent_bg_for_inpaint, mask_4.to(self.weight_dtype), stroke_seg_1, stroke_seg_1_no, con_flag, ratio)
        
        mask_for_loss = mask_for_loss.view(bsz, 2, 40)

        mask_for_loss = (
            torch.nn.functional.interpolate(
            mask_for_loss.unsqueeze(1), size=(32, 640)
            ).to(memory_format=torch.contiguous_format)
        )

        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        
        if choice < 0.1:
            style_ = no_style
        else:
            style_ = style
        
        model_pred = self.unet(
            noisy_latents, 
            timesteps, 
            [style_],
            memory=memory_2,
            layout_embeddings=None,
            char_glyph=char_glyph
            )
        model_pred = model_pred.sample

        return model_pred, noise


    def decode_inpaint_latents(self, latents, masked_image, mask):
        torch_image = (latents / 2 + 0.5).clamp(0, 1)
        image = torch_image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image, torch_image


    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        print(images.max())
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images
        
    @torch.no_grad()
    def inference_simple(self, bg_for_inpaint, stroke_seg_1, ori_mask_4, char_glyph=None, guidance_scale=7.5, num_inference_steps=50, height=224, width=224, ratio=0, eta: float = 0.0, generator = None, callback=None, callback_steps=1, output_type="pil"):
        batch_size = bg_for_inpaint.shape[0]
        mask_4 = ori_mask_4
        device = "cuda"
        all_latents = []
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = 3
        latents_shape = [
            batch_size, num_channels_latents,
            height, width
        ]
        latents = torch.randn(latents_shape, dtype=self.weight_dtype).cuda()
        latent_bg_for_inpaint = bg_for_inpaint.cuda()

        stroke_seg_1_no = torch.zeros_like(stroke_seg_1).to(latents.device).to(self.weight_dtype)
        
        b, _, h, w = latent_bg_for_inpaint.shape
        memory_2, style, no_style, mask_for_loss = self.unet.get_input_v3( latent_bg_for_inpaint, mask_4, stroke_seg_1.to(latents.device).to(self.weight_dtype), stroke_seg_1_no, con_flag=False,ratio=ratio)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        batch_size = latent_bg_for_inpaint.shape[0]
        if guidance_scale > 1.0:
            do_classifier_free_guidance = True
        else:
            do_classifier_free_guidance = False

        if do_classifier_free_guidance:
            style = torch.cat([no_style, style])
            memory_2 = torch.cat([memory_2 ,memory_2])
            char_glyph = torch.cat([char_glyph, char_glyph])
        outputs_seg_masks = None


        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latent_model_input = torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if do_classifier_free_guidance:
                    noise_pred = self.unet(
                        latent_model_input, 
                        t, 
                        [style],
                        memory=memory_2,
                        char_glyph=char_glyph,
                        layout_embeddings=None)
                else:
                    noise_pred = self.unet(
                        latent_model_input, 
                        t, 
                        [style],
                        memory=memory_2,
                        char_glyph=char_glyph,
                        layout_embeddings=None)
                noise_pred = noise_pred.sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                all_latents.append(latents)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        
        if output_type == "pil":
            all_decoded = []
            has_nsfw_concept = None
            image, torch_image = self.decode_inpaint_latents(latents, bg_for_inpaint, ori_mask_4)
        return image, torch_image

