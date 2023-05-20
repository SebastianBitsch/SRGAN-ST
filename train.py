import os

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

import model

from dataset import TrainImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from utils import load_state_dict, save_checkpoint, AverageMeter, ProgressMeter

import srgan_config
from loss import ContentLoss


# House keeping variables
batches_done = 0
best_psnr = 0.0
best_ssim = 0.0


# Define models
discriminator = model.Discriminator().to(srgan_config.device)
generator = model.Generator( # TODO: Pass entire config class instead of this mess
    in_channels=srgan_config.in_channels,
    out_channels=srgan_config.out_channels,
    channels=srgan_config.channels,
    num_rcb=srgan_config.num_rcb,
    upscale_factor=srgan_config.upscale_factor
).to(srgan_config.device)

# Define losses
adversarial_criterion = nn.BCEWithLogitsLoss()
content_criterion = ContentLoss(srgan_config.feature_model_extractor_nodes, device=srgan_config.device)
pixel_criterion = nn.MSELoss() # Maybe use torch.nn.L1Loss().to(device) ?

# Optimizers
d_optimizer = optim.Adam(
    discriminator.parameters(),
    srgan_config.model_lr,
    srgan_config.model_betas,
    srgan_config.model_eps,
    srgan_config.model_weight_decay
)
g_optimizer = optim.Adam(
    generator.parameters(),
    srgan_config.model_lr,
    srgan_config.model_betas,
    srgan_config.model_eps,
    srgan_config.model_weight_decay
)

# Schedulers
d_scheduler = lr_scheduler.StepLR(
    d_optimizer,
    srgan_config.lr_scheduler_step_size,
    srgan_config.lr_scheduler_gamma
)
g_scheduler = lr_scheduler.StepLR(
    g_optimizer,
    srgan_config.lr_scheduler_step_size,
    srgan_config.lr_scheduler_gamma
)

# Dataloaders
# Load train, test and valid datasets
train_datasets = TrainImageDataset(
    srgan_config.train_gt_images_dir,
    srgan_config.upscale_factor,
)
test_datasets = TestImageDataset(
    srgan_config.test_gt_images_dir, 
    srgan_config.test_lr_images_dir
)

# Generator all dataloader
train_dataloader = DataLoader(
    train_datasets,
    batch_size=srgan_config.batch_size,
    shuffle=True,
    num_workers=srgan_config.num_workers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)
test_dataloader = DataLoader(
    test_datasets,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
)

# Create an IQA evaluation model
psnr_model = PSNR(srgan_config.upscale_factor, srgan_config.only_test_y_channel).to(device=srgan_config.device)
ssim_model = SSIM(srgan_config.upscale_factor, srgan_config.only_test_y_channel).to(device=srgan_config.device)

# Tensorboard writer to store train and test info
writer = SummaryWriter(f"samples/logs/{srgan_config.exp_name}")


for epoch in range(srgan_config.start_epoch, srgan_config.n_epochs):

    # ----------------
    #  Train
    # ----------------
    for batch_num, (gt, lr) in enumerate(train_dataloader):

        batches_done = epoch * len(train_dataloader) + batch_num
        
        # Transfer in-memory data to CUDA devices to speed up training
        gt = gt.to(device=srgan_config.device, non_blocking=True)
        lr = lr.to(device=srgan_config.device, non_blocking=True)

        # Set the real sample label to 1, and the false sample label to 0
        batch_size, _, height, width = gt.shape
        real_label = torch.full([batch_size, 1], 1.0, dtype=gt.dtype, device=srgan_config.device)
        fake_label = torch.full([batch_size, 1], 0.0, dtype=gt.dtype, device=srgan_config.device)

        # ----------------
        #  Train Generator
        # ----------------
        g_optimizer.zero_grad()

        sr = generator(lr)

        # Apply losses ? maybe just pixel
        pixel_loss = 1.0 * pixel_criterion(gt, sr)

        if batches_done < srgan_config.n_warmup_batches:
            # TODO: Still log some data etc. and go backwards on some loss?
            continue
    
        # Extract validity predictions from discriminator
        pred_gt = discriminator(gt)
        pred_sr = discriminator(sr)

        adversarial_loss = 0.005 * adversarial_criterion(pred_sr - pred_gt.mean(0, keepdim=True), real_label)
        content_loss = 1.0 * content_criterion(gt, sr)

        g_loss = content_loss + pixel_loss + adversarial_loss # TODO: Add weighting

        g_loss.backward()
        g_optimizer.step()

        # --------------------
        #  Train Discriminator
        # --------------------
        d_optimizer.zero_grad()

        pred_gt = discriminator(gt)
        pred_sr = discriminator(sr)
        
        # Calculate the classification score of the discriminator model for real samples
        loss_real = adversarial_criterion(pred_gt - pred_sr.mean(0, keepdim=True), real_label)
        loss_fake = adversarial_criterion(pred_sr - pred_gt.mean(0, keepdim=True), fake_label)
        
        d_loss = (loss_real + loss_fake) / 2
        d_loss.backward()
        d_optimizer.step()

        # Calculate the score of the discriminator on real samples and fake samples,
        # the score of real samples is close to 1, and the score of fake samples is close to 0
        d_gt_probability = torch.sigmoid_(torch.mean(pred_gt.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(pred_sr.detach()))

        # -------------
        #  Log Progress
        # -------------
        if batches_done % srgan_config.train_logging_interval != 0:
            continue
        
        # Log to TensorBoard
        writer.add_scalar("Train/D_Loss", d_loss.item(), batches_done)
        writer.add_scalar("Train/G_Loss", g_loss.item(), batches_done)
        writer.add_scalar("Train/G_ContentLoss", content_loss.item(), batches_done)
        writer.add_scalar("Train/G_AdversarialLoss", adversarial_loss.item(), batches_done)
        writer.add_scalar("Train/G_PixelLoss", pixel_loss.item(), batches_done)
        writer.add_scalar("Train/D(GT)_Probability", d_gt_probability.item(), batches_done)
        writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), batches_done)

        # Print to terminal / log
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch + 1,
                srgan_config.n_epochs,
                batch_num,
                len(train_dataloader),
                d_loss.item(),
                g_loss.item(),
                content_loss.item(),
                adversarial_loss.item(),
                pixel_loss.item(),
            )
        )


    # ----------------
    #  Validate
    # ----------------
    generator.eval()

    avg_psnr = 0.0
    avg_ssim = 0.0
    with torch.no_grad():
        for batch_num, (gt, lr) in enumerate(test_dataloader):
            
            gt = gt.to(device=srgan_config.device, non_blocking=True)
            lr = lr.to(device=srgan_config.device, non_blocking=True)

            sr = generator(lr)

            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)

            avg_psnr += psnr
            avg_ssim += ssim

            # Print training log information
            if batch_num % srgan_config.valid_print_frequency == 0:
                print(f"[Test: {batch_num+1}/{len(test_dataloader)}] [PSNR: {psnr}] [SSIM: {ssim}]")
            
        print("-----")
        print(f"[AVG PSNR: {avg_psnr / len(test_dataloader)}] [AVG SSIM: {avg_ssim / len(test_dataloader)}]")
        print("-----\n")

 
    # Update learning rate
    d_scheduler.step()
    g_scheduler.step()

    # ----------------
    #  Save best model
    # ----------------
    is_best = best_psnr < psnr and best_ssim < ssim
    is_last = srgan_config.start_epoch + epoch == srgan_config.n_epochs - 1

    results_dir = f"results/{srgan_config.exp_name}"
    os.makedirs(results_dir, exist_ok=True)
    if is_last:
        torch.save(generator.state_dict(), results_dir  + "/g_last.pth")
        torch.save(discriminator.state_dict(), results_dir  + "/d_last.pth")
    if is_best:
        torch.save(generator.state_dict(), results_dir  + "/g_best.pth")
        torch.save(discriminator.state_dict(), results_dir  + "/d_best.pth")
        best_psnr = psnr
        best_ssim = ssim

    #TODO: Maybe run test.py here and save to a grid of output images to see progress.
    