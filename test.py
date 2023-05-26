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

import cv2
import torch
from natsort import natsorted

import imgproc
import model
from image_quality_assessment import PSNR, SSIM
from torchvision.io import read_image

from config import Config

def test(config: Config, g_path: str = None) -> None:
    if not g_path:
        g_path = f"results/{config.EXP.NAME}/g_best.pth"

    # Initialize the super-resolution bsrgan_model
    generator = model.Generator(config).to(config.MODEL.DEVICE)
    generator.load_state_dict(torch.load(g_path))

    # Create a folder of super-resolution experiment results
    os.makedirs(os.path.join(config.DATA.TEST_SR_IMAGES_DIR,config.EXP.NAME), exist_ok=True)

    # Start the verification mode of the bsrgan_model.
    generator.eval()

    # Initialize the sharpness evaluation function
    ssim_model = SSIM(config.DATA.UPSCALE_FACTOR, True).to(device=config.MODEL.DEVICE)
    psnr_model = PSNR(config.DATA.UPSCALE_FACTOR, True).to(device=config.MODEL.DEVICE)

    # Initialize IQA metrics
    psnr_avg = 0.0
    ssim_avg = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.DATA.TEST_LR_IMAGES_DIR))
    n_images = len(file_names)

    for im_name in file_names:
        lr_image_path = os.path.join(config.DATA.TEST_LR_IMAGES_DIR, im_name)
        sr_image_path = os.path.join(config.DATA.TEST_SR_IMAGES_DIR, config.EXP.NAME + "1", im_name)
        gt_image_path = os.path.join(config.DATA.TEST_GT_IMAGES_DIR, im_name)

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_tensor = imgproc.preprocess_one_image(lr_image_path, config.MODEL.DEVICE)
        gt_tensor = imgproc.preprocess_one_image(gt_image_path, config.MODEL.DEVICE)
        gt_tensor = read_image(im_path).float().unsqueeze(0) / 255.0
        lr_tensor = F.interpolate(gt_tensor, scale_factor = 1.0 / self.upscale_factor, mode='bicubic', antialias=True)
        gt_tensor = gt_tensor.squeeze()
        lr_tensor = lr_tensor.squeeze()

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = generator(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, True, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        psnr_avg += psnr_model(sr_tensor, gt_tensor).item()
        ssim_avg += ssim_model(sr_tensor, gt_tensor).item()

    psnr_avg /= n_images
    ssim_avg /= n_images

    # Write PSNR and SSIM data to file and terminal
    psnr_label = f"PSNR: {psnr_avg:4.2f} [dB]\n"
    ssim_label = f"SSIM: {ssim_avg:4.4f} [u]"
    print(psnr_label + ssim_label)
    
    out_file = open(f"results/test/{config.EXP.NAME}1/_metrics.txt","w")
    out_file.writelines([psnr_label, ssim_label])
    out_file.close()


if __name__ == "__main__":
    config = Config()
    config.EXP.NAME = "bbgan-label-smoothing-x10-loss"

    test(config = config)