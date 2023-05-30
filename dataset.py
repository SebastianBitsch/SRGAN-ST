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
import os
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image

from bicubic import Bicubic


class TrainImageDataset(Dataset):
    """
    A wrapper class around the torch Dataset class.
    Used for getting GT and LR images for training
    """

    def __init__(self, gt_image_dir: str, upscale_factor: int) -> None:
        super(TrainImageDataset, self).__init__()
        self.image_file_names = [
            os.path.join(gt_image_dir, image_file_name) for image_file_name in absoluteFilePaths(gt_image_dir)
        ]
        self.upscale_factor = upscale_factor
        self.bicubic = Bicubic()

    def __getitem__(self, batch_index: int) -> tuple[Tensor, Tensor]:
        # Read a batch of image data
        im_path = self.image_file_names[batch_index]

        gt_tensor = read_image(im_path).float().unsqueeze(0) / 255.0
        lr_tensor = self.bicubic(gt_tensor, scale = 1.0 / self.upscale_factor)
        gt_tensor = gt_tensor.squeeze()
        lr_tensor = lr_tensor.squeeze()

        return gt_tensor, lr_tensor

    def __len__(self) -> int:
        return len(self.image_file_names)



class TestImageDataset(Dataset):
    """
    A wrapper class around the torch Dataset class.
    Used for getting GT and LR images for training
    """

    def __init__(self, test_gt_images_dir: str, test_lr_images_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.gt_image_file_names = [os.path.join(test_gt_images_dir, x) for x in absoluteFilePaths(test_gt_images_dir) if not x.startswith('.')]
        self.lr_image_file_names = [os.path.join(test_lr_images_dir, x) for x in absoluteFilePaths(test_lr_images_dir) if not x.startswith('.')]

    def __getitem__(self, batch_index: int) -> tuple[Tensor, Tensor]:
        gt_tensor = read_image(self.gt_image_file_names[batch_index]).float() / 255.0
        lr_tensor = read_image(self.lr_image_file_names[batch_index]).float() / 255.0

        return gt_tensor, lr_tensor

    def __len__(self) -> int:
        return len(self.gt_image_file_names)


def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
