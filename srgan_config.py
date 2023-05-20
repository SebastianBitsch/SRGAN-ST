# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random
from torch import nn

from loss import EuclidLoss, BBLoss, GBBLoss, ContentLoss

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0) # ok
torch.manual_seed(0) # ok
np.random.seed(0) # ok
# Use GPU for training by default
# device = torch.device("cuda", 0) # ok
device = "cuda:0"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True # ok
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True # nej

# Model arch config
in_channels = 3
out_channels = 3
channels = 64
num_rcb = 16
# Test upscale factor
upscale_factor = 4
# Current configuration parameter method
mode = None
# Experiment name, easy to save weights and log files
exp_name = None

n_warmup_batches = 0

# Feature extraction layer parameter configuration. See Perceptual loss in GramGAN
feature_model_extractor_nodes = {
    "features.17" : 1/8,
    "features.26" : 1/4,
    "features.35" : 1/2
}

srgan_losses = {
    "AdversarialLoss": nn.BCEWithLogitsLoss(),
    "PixelLoss": nn.MSELoss(),
    "ContentLoss" : ContentLoss(feature_model_extractor_nodes, device=device)
}

bbgan_losses = {
    "AdversarialLoss": nn.BCEWithLogitsLoss(),
    "PixelLoss": nn.MSELoss(),
    "ContentLoss" : ContentLoss(feature_model_extractor_nodes, device=device),
    "BBLoss" : BBLoss()
}

gramgan_losses = {
    "AdversarialLoss": nn.BCEWithLogitsLoss(),
    "PixelLoss": nn.MSELoss(),
    "ContentLoss" : ContentLoss(feature_model_extractor_nodes, device=device),
    "GBBLoss" : GBBLoss()
}

# stgan_losses = {
#     "AdversarialLoss": nn.BCEWithLogitsLoss(),
#     "PixelLoss": nn.MSELoss(),
#     "ContentLoss" : ContentLoss(feature_model_extractor_node, feature_model_normalize_mean, feature_model_normalize_std),
#     "BBLoss" : ()
# }

g_losses = None

loss_weights = {
    "AdversarialLoss": 0.005,
    "PixelLoss": 1.0,
    "ContentLoss" : 1.0,
    "BBLoss" : 10.0,
    "GBBLoss" : 10.0
}

# Whether to save each epoch of trained data, setting to false allows to train for more epochs
save_checkpoints = True

# if mode == "train":
# Dataset address
base_dir = "/work3/s204163"
data_dir = "/data"
train_gt_images_dir = base_dir + data_dir + "/ImageNet/train"

test_gt_images_dir = base_dir + data_dir + "/Set5/GTmod12"
test_lr_images_dir = base_dir + data_dir + f"/Set5/LRbicx{upscale_factor}"

# The size of our GT Images, ideally the same as has been used when slicing images in image_slicer.py
gt_image_size = 48 * 4

batch_size = 16

# Makes parallelized tasks such as loading data faster - setting it to higher throws a warning
num_workers = 1

# The address to load the pretrained model
pretrained_d_model_weights_path = f"" # Not in use
pretrained_g_model_weights_path = f"" # Not in use

# Incremental training and migration training
resume_d_model_weights_path = f"" # Not in use
resume_g_model_weights_path = f"" # Not in use

# Total num epochs (200,000 iters) - is set with cli argument
epochs = None
n_epochs = 2
start_epoch = 0


# Optimizer parameter
model_lr = 1e-4
model_betas = (0.9, 0.999)
model_eps = 1e-8
model_weight_decay = 0.0

# Dynamically adjust the learning rate policy [100,000 | 200,000] - set with cli arg
lr_scheduler_step_size = n_epochs // 2
lr_scheduler_gamma = 0.1

# How many iterations to print the training result
train_print_frequency = 100
valid_print_frequency = 1
train_logging_interval = 100
valid_logging_interval = 1

# if mode == "test":
# Test data address
lr_dir = base_dir + data_dir + f"/Set5/LRbicx{upscale_factor}"
gt_dir = base_dir + data_dir + "/Set5/GTmod12"
sr_dir = f"./results/test/{exp_name}"

# Is set in test.py
g_model_weights_path = None 
