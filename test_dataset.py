import sys
import os
sys.path.append('.')
from dataloader.dataloader import LineText

vis_root = "vis_training_set"# your path
dataset = LineText(True, "LineData", "train-50k-1", use_fix_width=128, vis_flag=True, vis_root=vis_root)

dataset.__getitem__(0)
dataset.__getitem__(1)
dataset.__getitem__(2)



