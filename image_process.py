import os

import cv2
import torch
import numpy as np
from torchvision.io import read_image
from torchvision.utils import save_image

import imgproc
import model
from config import Config
import torch.nn.functional as F

config = Config()


g_path = f"results/bbgan-label-smoothing-x10-loss/g_best.pth"

# Initialize the super-resolution bsrgan_model
generator = model.Generator(config).to(config.MODEL.DEVICE)
generator.load_state_dict(torch.load(g_path))

# lr_tensor1 = imgproc.preprocess_one_image("/work3/s204163/data/Urban100/LRbicx4/img_062.png", config.MODEL.DEVICE)
f = "/work3/s204163/data/Urban100/GTmod12/img_059.png"
gt_tensor = read_image(f).float().unsqueeze(0) / 255.0
print(gt_tensor.shape)
lr_tensor = F.interpolate(gt_tensor, scale_factor = 1.0 / 4, antialias=True, mode='bicubic')
print(lr_tensor.shape)
gt_tensor = gt_tensor.squeeze()
lr_tensor = lr_tensor.squeeze()
print(gt_tensor.shape)
print(lr_tensor.shape)

# Only reconstruct the Y channel image data.
with torch.no_grad():
    sr_tensor = generator(lr_tensor)

    tensor = sr_tensor.squeeze().float().cpu()
    save_image(tensor, "12345.png")


print(sr_tensor.shape)