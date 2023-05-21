import os

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model

from dataset import TrainImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM

# from torchmetrics.functional import structural_similarity_index_measure
# from torchmetrics import PeakSignalNoiseRatio

# import srgan_config
from loss import ContentLoss

from config import Config
from utils import init_random_seed

def train(config: Config = None):
    config = Config() if not config else config
    
    # Set seed
    init_random_seed(config.DATA.SEED)

    # House keeping variables
    batches_done = 0
    best_psnr = 0.0
    best_ssim = 0.0

    # Define models
    discriminator = model.Discriminator().to(config.MODEL.DEVICE)
    generator = model.Generator(config).to(config.MODEL.DEVICE)

    # Define losses
    adversarial_criterion = nn.BCEWithLogitsLoss()
    content_criterion = ContentLoss(config.MODEL.G_LOSS.VGG19_LAYERS, device=config.MODEL.DEVICE)
    pixel_criterion = nn.MSELoss() # Maybe use torch.nn.L1Loss().to(device) ?


    # Optimizers
    d_optimizer = optim.Adam(
        params = discriminator.parameters(),
        lr     = config.SOLVER.D_BASE_LR,
        betas  = (config.SOLVER.D_BETA1, config.SOLVER.D_BETA2),
        eps    = config.SOLVER.D_EPS,
        weight_decay = config.SOLVER.D_WEIGHT_DECAY
    )
    g_optimizer = optim.Adam(
        params = generator.parameters(),
        lr     = config.SOLVER.G_BASE_LR,
        betas  = (config.SOLVER.G_BETA1, config.SOLVER.G_BETA2),
        eps    = config.SOLVER.G_EPS,
        weight_decay = config.SOLVER.G_WEIGHT_DECAY
    )

    # Schedulers
    d_scheduler = lr_scheduler.StepLR(
        optimizer = d_optimizer,
        step_size = config.SCHEDULER.STEP_SIZE,
        gamma = config.SCHEDULER.GAMMA
    )
    g_scheduler = lr_scheduler.StepLR(
        optimizer = g_optimizer,
        step_size = config.SCHEDULER.STEP_SIZE,
        gamma = config.SCHEDULER.GAMMA
    )

    # Dataloaders
    # Load train, test and valid datasets
    train_datasets = TrainImageDataset(config.DATA.TRAIN_GT_IMAGES_DIR, config.DATA.UPSCALE_FACTOR)
    test_datasets = TestImageDataset(config.DATA.TEST_GT_IMAGES_DIR, config.DATA.TEST_LR_IMAGES_DIR)

    # Generator all dataloader
    train_dataloader = DataLoader(
        dataset = train_datasets,
        batch_size = config.DATA.BATCH_SIZE,
        shuffle = True,
        num_workers = 1,
        pin_memory = True,
        drop_last = True,
        persistent_workers = True,
    )
    test_dataloader = DataLoader(
        dataset = test_datasets,
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        pin_memory = True,
        drop_last = False,
        persistent_workers = True,
    )

    # Create an IQA evaluation model
    ssim_model = SSIM(config.DATA.UPSCALE_FACTOR, True).to(device=config.MODEL.DEVICE)
    psnr_model = PSNR(config.DATA.UPSCALE_FACTOR, True).to(device=config.MODEL.DEVICE)

    # Tensorboard writer to store train and test info
    writer = SummaryWriter(f"samples/logs/{config.EXP.NAME}")


    for epoch in range(config.EXP.START_EPOCH, config.EXP.N_EPOCHS):
        # ----------------
        #  Train
        # ----------------
        for batch_num, (gt, lr) in enumerate(train_dataloader):

            batches_done = epoch * len(train_dataloader) + batch_num
            
            # Transfer in-memory data to CUDA devices to speed up training
            gt = gt.to(device=config.MODEL.DEVICE, non_blocking=True)
            lr = lr.to(device=config.MODEL.DEVICE, non_blocking=True)

            # Set the real sample label to 1, and the false sample label to 0
            batch_size, _, height, width = gt.shape
            real_label = torch.full([batch_size, 1], 0.9, dtype=gt.dtype, device=config.MODEL.DEVICE)
            fake_label = torch.full([batch_size, 1], 0.0, dtype=gt.dtype, device=config.MODEL.DEVICE)

            # ----------------
            #  Train Generator
            # ----------------
            g_optimizer.zero_grad()

            sr = generator(lr)

            # Apply losses ? maybe just pixel
            pixel_loss = 1.0 * pixel_criterion(gt, sr)

            if epoch < config.EXP.N_WARMUP_EPOCHS:
                pixel_loss.backward()
                g_optimizer.step()
                if batches_done % config.LOG_TRAIN_PERIOD != 0:
                    continue
                print("[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]" % (epoch, config.EXP.N_EPOCHS, batch_num, len(train_dataloader), pixel_loss.item()))
                continue
        
            # Extract validity predictions from discriminator
            pred_gt = discriminator(gt)
            pred_sr = discriminator(sr).detach()

            adversarial_loss = adversarial_criterion(pred_sr - pred_gt.mean(0, keepdim=True), real_label) * config.MODEL.G_LOSS.CRITERION_WEIGHTS['Adversarial']
            content_loss = 1.0 * content_criterion(gt, sr)

            g_loss = content_loss + pixel_loss + adversarial_loss # TODO: Add weighting

            g_loss.backward()
            g_optimizer.step()

            # --------------------
            #  Train Discriminator
            # --------------------
            d_optimizer.zero_grad()

            pred_gt = discriminator(gt)
            pred_sr = discriminator(sr.detach())
            
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
            if batches_done % config.LOG_TRAIN_PERIOD != 0:
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
                    config.EXP.N_EPOCHS,
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
                
                gt = gt.to(device=config.MODEL.DEVICE, non_blocking=True)
                lr = lr.to(device=config.MODEL.DEVICE, non_blocking=True)

                sr = generator(lr)

                psnr = psnr_model(sr, gt)
                ssim = ssim_model(sr, gt)
                # psnr1 = PeakSignalNoiseRatio(data_range=1.0).to(device=config.MODEL.DEVICE, non_blocking=True)(sr, gt)
                # ssim1 = structural_similarity_index_measure(sr, gt)

                avg_psnr += psnr
                avg_ssim += ssim
                # avg_psnr1 += psnr1
                # avg_ssim1 += ssim1 

                # Print training log information
                if batch_num % config.LOG_VALIDATION_PERIOD == 0:
                    print(f"[Test: {batch_num+1}/{len(test_dataloader)}] [PSNR: {psnr.item()}] [SSIM: {ssim.item()}]")
                    # print(f"[Test: {batch_num+1}/{len(test_dataloader)}] [PSNR1: {psnr1.item()}] [SSIM1: {ssim1.item()}]")

        # Write avg PSNR and SSIM to Tensorflow and logs
        avg_psnr = (avg_psnr / len(test_dataloader)).item()
        avg_ssim = (avg_ssim / len(test_dataloader)).item()
        # avg_psnr1 = (avg_psnr1 / len(test_dataloader)).item()
        # avg_ssim1 = (avg_ssim1 / len(test_dataloader)).item()
        writer.add_scalar(f"Test/PSNR", avg_psnr, epoch + 1)
        writer.add_scalar(f"Test/SSIM", avg_ssim, epoch + 1)
        
        print(f"-----\n[AVG PSNR: {avg_psnr}] [AVG SSIM: {avg_ssim}]\n-----\n")
        # print(f"-----\n[AVG PSNR: {avg_psnr1}] [AVG SSIM: {avg_ssim1}]\n-----\n")
        
        # Update learning rate
        d_scheduler.step()
        g_scheduler.step()

        # ----------------
        #  Save best model
        # ----------------
        is_best = best_psnr < psnr and best_ssim < ssim
        is_last = config.EXP.START_EPOCH + epoch == config.EXP.N_EPOCHS - 1

        results_dir = f"results/{config.EXP.NAME}"
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