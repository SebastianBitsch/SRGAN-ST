import cv2
import os

import numpy as np
import torch

from config import Config
from torch.utils.data import DataLoader
from dataset import TestImageDataset
from model import Generator
from utils import bgr2ycbcr, tensor2img, PSNR, SSIM

def test(config: Config, save_images: bool = True, g_path: str = None):
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

    # Initialize the super-resolution bsrgan_model
    generator = Generator(config).to(config.DEVICE)
    
    # Load weights of generator
    generator.load_state_dict(torch.load(g_path))

    # Test
    psnr, ssim = _validate(generator=generator, val_loader=test_dataloader, config=config, save_images=save_images)
    print(f"[Test] [PSNR: {psnr}] [SSIM: {ssim}]")


def _validate(generator: Generator, val_loader: DataLoader, config: Config, save_images:bool = False) -> tuple[float, float]:
    """ Run testing on a generator with a given dataset """
    with torch.no_grad():
        psnr_l = []
        ssim_l = []

        for idx, (hr_img, lr_img) in enumerate(val_loader):
            lr_img = lr_img.to(config.DEVICE)
            hr_img = hr_img.to(config.DEVICE)

            output = generator(lr_img)

            output = tensor2img(output)
            gt = tensor2img(hr_img)

            # Save SR images
            if save_images:
                path = os.path.join(config.DATA.TEST_SR_IMAGES_DIR, config.EXP.NAME)
                os.makedirs(path, exist_ok=True)
                cv2.imwrite(f"{path}/{idx}.png", np.concatenate([output, gt], axis=1))

            output = output.astype(np.float32) / 255.0
            gt = gt.astype(np.float32) / 255.0

            output = bgr2ycbcr(output, only_y=True)
            gt = bgr2ycbcr(gt, only_y=True)
            psnr = PSNR(output * 255, gt * 255)
            ssim = SSIM(output * 255, gt * 255)
            psnr_l.append(psnr)
            ssim_l.append(ssim)

        avg_psnr = sum(psnr_l) / len(psnr_l)
        avg_ssim = sum(ssim_l) / len(ssim_l)

    return avg_psnr, avg_ssim


if __name__ == "__main__":
    config = Config()
    config.EXP.NAME = "ablation-c1-bestbuddy"
    config.DATA.TEST_SET = "Set5"
    # TODO
    config.DATA.TEST_GT_IMAGES_DIR = F"/work3/{config.EXP.USER}/data/{config.DATA.TEST_SET}/GTmod12"
    config.DATA.TEST_LR_IMAGES_DIR = f"/work3/{config.EXP.USER}/data/{config.DATA.TEST_SET}/LRbicx4"

    test(config = config, save_images = True)