import cv2
import os

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from torch.utils.data import DataLoader
from dataset import TestImageDataset
from model import Generator
from utils import bgr2ycbcr, tensor2img, PSNR, SSIM
from bicubic import Bicubic

def test(config: Config, save_images: bool = True, g_path: str = None, concat_w_gt: bool = False):
    """
    Test a generator, if not path to generator is given the generator at current exp-name is used.
    """
    if not g_path:
        g_path = f"results/{config.EXP.NAME}/g_best.pth"
    
    test_datasets = TestImageDataset(config.DATA.TEST_GT_IMAGES_DIR, config.DATA.TEST_LR_IMAGES_DIR)

    test_dataloader = DataLoader(
        dataset = test_datasets,
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        pin_memory = True,
        drop_last = False,
        persistent_workers = True,
    )

    # Initialize generator and load weights
    if config.EXP.NAME == "bicubic":
        generator = Bicubic(device=config.DEVICE).to(config.DEVICE)
    else:
        generator = Generator(config).to(config.DEVICE)
        generator.load_state_dict(torch.load(g_path))

    # Test
    psnr, ssim = _validate(generator=generator, val_loader=test_dataloader, config=config, save_images=save_images, concat_with_gt=concat_w_gt)
    print(f"[Test] [PSNR: {psnr}] [SSIM: {ssim}]")


def _validate(generator, val_loader: DataLoader, config: Config, save_images:bool = False, concat_with_gt:bool = False) -> tuple[float, float]:
    """ Run testing on a generator(or bicubic etc.) with a given dataset """
    with torch.no_grad():
        all_psnr = []
        all_ssim = []

        for idx, (hr_img, lr_img) in enumerate(tqdm(val_loader)):
            lr_img = lr_img.to(config.DEVICE)
            hr_img = hr_img.to(config.DEVICE)

            output = generator(lr_img)

            output = tensor2img(output)
            gt = tensor2img(hr_img)

            # Save SR images
            if save_images:
                path = os.path.join(config.DATA.TEST_SR_IMAGES_DIR, config.EXP.NAME)
                os.makedirs(path, exist_ok=True)
                if concat_with_gt:
                    cv2.imwrite(f"{path}/{idx}.png", np.concatenate([output, gt], axis=1))
                else:
                    cv2.imwrite(f"{path}/{idx}.png", output)

            output = output.astype(np.float32) / 255.0
            gt = gt.astype(np.float32) / 255.0

            output = bgr2ycbcr(output, only_y=True)
            gt = bgr2ycbcr(gt, only_y=True)
            psnr = PSNR(output * 255, gt * 255)
            ssim = SSIM(output * 255, gt * 255)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

        avg_psnr = sum(all_psnr) / len(all_psnr)
        avg_ssim = sum(all_ssim) / len(all_ssim)

    return avg_psnr, avg_ssim


if __name__ == "__main__":
    config = Config()

    # Set the model to test - model should be in /results/ folder, else use gpath parameter for test func
    config.EXP.NAME = "plain-w-pixel"

    # Set the dataset to test on
    config.DATA.TEST_SET = "Urban100"
    config.DATA.TEST_GT_IMAGES_DIR = F"/work3/{config.EXP.USER}/data/{config.DATA.TEST_SET}/GTmod12"
    config.DATA.TEST_LR_IMAGES_DIR = f"/work3/{config.EXP.USER}/data/{config.DATA.TEST_SET}/LRbicx4"

    gpath = "results/SRResNet-lorna-pretrained.pth"
    test(config = config, save_images = True, concat_w_gt = False, g_path=gpath)