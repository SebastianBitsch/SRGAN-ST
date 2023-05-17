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
import argparse

import cv2
import torch
from natsort import natsorted

import imgproc
import model
import srgan_config
from image_quality_assessment import PSNR, SSIM


def main() -> None:
    # Initialize the super-resolution bsrgan_model
    g_model = model.Generator(
        in_channels=srgan_config.in_channels,
        out_channels=srgan_config.out_channels,
        channels=srgan_config.channels,
        num_rcb=srgan_config.num_rcb,
        upscale_factor=srgan_config.upscale_factor
    )
    g_model = g_model.to(device=srgan_config.device)
    print(f"Successfully built generator.")

    # Load the super-resolution bsrgan_model weights
    checkpoint = torch.load(srgan_config.g_model_weights_path, map_location=lambda storage, _: storage)
    g_model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded generator model weights {os.path.abspath(srgan_config.g_model_weights_path)} successfully.")

    # Create a folder of super-resolution experiment results
    os.makedirs(srgan_config.sr_dir, exist_ok=True)

    # Start the verification mode of the bsrgan_model.
    g_model.eval()

    # Initialize the sharpness evaluation function
    psnr = PSNR(srgan_config.upscale_factor, srgan_config.only_test_y_channel)
    ssim = SSIM(srgan_config.upscale_factor, srgan_config.only_test_y_channel)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=srgan_config.device, non_blocking=True)
    ssim = ssim.to(device=srgan_config.device, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(srgan_config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(srgan_config.lr_dir, file_names[index])
        sr_image_path = os.path.join(srgan_config.sr_dir, file_names[index])
        gt_image_path = os.path.join(srgan_config.gt_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_tensor = imgproc.preprocess_one_image(lr_image_path, srgan_config.device)
        gt_tensor = imgproc.preprocess_one_image(gt_image_path, srgan_config.device)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = g_model(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        psnr_metrics += psnr(sr_tensor, gt_tensor).item()
        ssim_metrics += ssim(sr_tensor, gt_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files


    # Write PSNR and SSIM data to file and terminal
    psnr_label = f"PSNR: {avg_psnr:4.2f} [dB]\n"
    ssim_label = f"SSIM: {avg_ssim:4.4f} [u]"
    print(psnr_label + ssim_label)
    
    out_file = open(f"results/test/{srgan_config.exp_name}/metrics.txt","w")
    out_file.writelines([psnr_label, ssim_label])
    out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'SRGAN-ST',
                    description = 'Does super resolution',
                    epilog = 'Text at the bottom of help bla bla')
    parser.add_argument('-exp_name', type=str, help='The name of the experiment')
    parser.add_argument('-g_weights', type=str, help='The path of the g weights')#, default="./results/pretrained_models/SRGAN_x4-ImageNet-8c4a7569.pth.tar")
    
    args = parser.parse_args()

    srgan_config.exp_name = args.exp_name
    srgan_config.g_model_weights_path = args.g_weights
    srgan_config.mode = "test"

    srgan_config.sr_dir = f"./results/test/{srgan_config.exp_name}"
    
    main()
