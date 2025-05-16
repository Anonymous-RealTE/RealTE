import argparse 
import torch
class DefaultParser(object):

    def __init__(self):
        parser = argparse.ArgumentParser()
        # Data
        parser.add_argument('--data_root', type=str)
        parser.add_argument('--batch_aug', action='store_true') # True
        parser.add_argument('--chars', type=str, default=' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
        # Training Parameters
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--train_batch_size', type=int, default=4)
        parser.add_argument('--dataloader_num_workers', type=int, default=4)
        parser.add_argument('--steps', type=int, default=500000)
        parser.add_argument('--seed', type=int, default=23)
        parser.add_argument('--resume', type=str, default='')
        parser.add_argument('--output_dir', type=str, default='ckpt')
        parser.add_argument('--checkpoint_freq', type=int, default=200)
        parser.add_argument('--change_text', action='store_true')
        parser.add_argument('--vis_map', action='store_true')
        parser.add_argument('--vis_vae_repro', action='store_true')
        parser.add_argument('--weight_dtype', default=torch.float16)
        parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

        ### 
        parser.add_argument('--logging_dir', default="logs")
        parser.add_argument('--enable_xformers_memory_efficient_attention', action='store_true')
        parser.add_argument('--adam_beta1', default=0.9)
        parser.add_argument('--adam_beta2', default=0.999)
        parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
        parser.add_argument("--adam_epsilon", type=float, default=1e-08)
        parser.add_argument("--gradient_accumulation_steps", default=4)
        parser.add_argument("--max_train_steps", default=500000)
        parser.add_argument("--scale_lr", action='store_true')
        parser.add_argument("--checkpoints_total_limit", type=int,default=2)
        parser.add_argument("--report_to",type=str,default="tensorboard")
        parser.add_argument("--lr_scheduler", type=str, default="constant")
        parser.add_argument("--lr_warmup_steps", type=int, default=500)
        parser.add_argument("--max_grad_norm", default=1.0)
        parser.add_argument("--use_ema",action="store_true")
        parser.add_argument("--local_rank", default=-1)
        parser.add_argument("--with_x0_loss", default=False)
        parser.add_argument("--guidance_scale", default=7.5)
        parser.add_argument("--with_p_loss", default=False)
        parser.add_argument("--with_rec", default=False)
        parser.add_argument("--datasets", default="icdar2013_totaltext_icdar2015_mlt2017")
        parser.add_argument("--ratio", default=0)
        
        parser.add_argument("--rec_model_path", type=str, default="lib/checkpoint_3_acc_0.8222.pth")
        self.parser = parser

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)
    
    def parse_args(self):
        args = self.parser.parse_args()

        return args
        