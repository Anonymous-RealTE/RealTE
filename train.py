import sys
import os
sys.path.append('.')
sys.path.append('xformers')
sys.path.append('mae')

import re
import six
import PIL
import cv2
import math 
import logging
import argparse
import accelerate
from pathlib import Path
from typing import Optional
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
import datasets
import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
import torch.utils.checkpoint
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import random

from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from dataloader.dataloader import LineText
import transformers
import diffusers
import matplotlib.pyplot as plt
from src.model import TextEditor

from utils.parser import DefaultParser

logger = get_logger(__name__, log_level="INFO")


class MaskMSELoss(nn.Module):
    def __init__(self, alpha=1, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target, mask):

        b, c, h, w = mask.shape
        mask_loss = F.mse_loss(
            input[mask == 1], target[mask == 1], reduction="sum")
        mse_loss_mean = (self.alpha * mask_loss) / (torch.sum(mask==1) + 1e-6) 

        bg_mask_loss = F.mse_loss(
            input[mask == 0], target[mask == 0], reduction="sum")
        bg_mask_loss_mean = (bg_mask_loss * 0.2) / (torch.sum(mask==0) + 1e-6)


        return mse_loss_mean, bg_mask_loss_mean




def main(args):
    set_seed(52)
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        kwargs_handlers=[ddp_kwargs]
    )
    weight_dtype = args.weight_dtype
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    logging.basicConfig( 
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

 
    model_path = args.resume
    model = TextEditor(model_path, weight_dtype)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.enable_xformers_memory_efficient_attention:
        
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.unet.enable_xformers_memory_efficient_attention()
            #model.layout_net.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.unet.save_pretrained(output_dir +"/unet")
                #model.vae.save_pretrained(output_dir +"/vae")
                #torch.save(model.text_encoder.state_dict(), output_dir +"/text_encoder")
                #torch.save(model.discriminator.state_dict(), output_dir +"/discriminator")
                #torch.save(model.pos_head.state_dict(), output_dir + "/pos_head")
                #torch.save(model.cls_head.state_dict(), output_dir + "/cls_head")
                model.scheduler.save_config(output_dir + '/scheduler')
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                
                # pop models so that they are not loaded again
                model = models.pop()
           
                load_model = Modifier_v3_no_glyph(input_dir, weight_dtype)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    
    def collate_fn(batches):
        imgs = []
        bg_for_inpaint = []
        masks = []
        ori_labels = [] 
        reference = []
        char_glyph = []

        # crop_image, bg_for_inpaint, mask_all, label, reference, char_glyph, resized_image
    
        for sample in batches:
            imgs.append(sample[0])
            bg_for_inpaint.append(sample[1])
            masks.append(sample[2])
            ori_labels.append(sample[3])
            reference.append(sample[4])
            char_glyph.append(sample[5])

        imgs = torch.stack(imgs, 0)
        bg_for_inpaint = torch.stack(bg_for_inpaint, 0)
        masks = torch.stack(masks, 0)
        reference = torch.stack(reference, 0)
        char_glyph = torch.stack(char_glyph, 0)
        return imgs, bg_for_inpaint, ori_labels, masks, reference, char_glyph

    optimizer = torch.optim.AdamW(
        list(model.unet.parameters()),# + list(model.cl_net.parameters()),# + list(model.cls_head.parameters()) + list(model.pos_head.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    global_step = 0
    train_dataset = LineText(True, args.data_root, args.datasets)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn
    )
    
    print(type(args.train_batch_size), type(args.gradient_accumulation_steps))
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    # Afterwards we recalculate our number of training epochs
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    args.num_train_epochs = 10 #math.ceil(args.max_train_steps / num_update_steps_per_epoch)



    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
       model, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    #if accelerator.is_main_process:
    #    accelerator.init_trackers("TextEditor", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    loss_func = MaskMSELoss()

    for epoch in range(0, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                images = batch[0].to(weight_dtype).cuda()
                bg_for_inpaint = batch[1].to(weight_dtype)

                ori_texts = batch[2]
                mask = batch[3]
                reference = batch[4].cuda().to(weight_dtype)
                char_glyph = batch[5].cuda().to(weight_dtype)


                mask_4 = mask.unsqueeze(1)
            
                
                model_pred, noise = model(images, bg_for_inpaint, reference, mask_4, char_glyph)
           
                loss_stage_2, loss_stage_2_bg = loss_func(model_pred, noise, mask_4.expand_as(model_pred))
            

                loss = loss_stage_2 + loss_stage_2_bg
                
                accelerator.backward(loss)
		        
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

   
                progress_bar.update(1)
                global_step += 1
                logs = {"step_loss": loss.detach().item(), "mse": loss_stage_2.detach().item()}

   
                progress_bar.set_postfix(**logs)
               
                if global_step % 200 == 0:
                    if accelerator.is_main_process:
                        save_path = args.output_dir
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")


if __name__ == "__main__":
    parser = DefaultParser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    main(args)
