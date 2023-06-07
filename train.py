import os

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from model import Generator, Discriminator
from dataset import TrainImageDataset, TestImageDataset

from utils import init_random_seed
from validate import _validate


def train(config: Config = None):
    
    # Set seed
    init_random_seed(config.DATA.SEED)

    # House keeping variables
    best_psnr = best_ssim = 0.0
    loss_values = dict()

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
    
    # Define models
    discriminator = Discriminator(config).to(config.DEVICE)
    generator = Generator(config).to(config.DEVICE)


    # Define losses
    adversarial_criterion = torch.nn.BCEWithLogitsLoss().to(config.DEVICE)

    # Optimizers
    d_optimizer = torch.optim.Adam(
        params = discriminator.parameters(),
        lr     = config.SOLVER.D_BASE_LR,
        betas  = (config.SOLVER.D_BETA1, config.SOLVER.D_BETA2),
        eps    = config.SOLVER.D_EPS,
        weight_decay = config.SOLVER.D_WEIGHT_DECAY
    )
    g_optimizer = torch.optim.Adam(
        params = generator.parameters(),
        lr     = config.SOLVER.G_BASE_LR,
        betas  = (config.SOLVER.G_BETA1, config.SOLVER.G_BETA2),
        eps    = config.SOLVER.G_EPS,
        weight_decay = config.SOLVER.G_WEIGHT_DECAY
    )

    # Schedulers
    d_scheduler = lr_scheduler.MultiStepLR(
        optimizer = d_optimizer,
        milestones = [ 10 ],
        gamma = config.SCHEDULER.GAMMA
    )
    g_scheduler = lr_scheduler.MultiStepLR(
        optimizer = g_optimizer,
        milestones = [ 10 ], # TODO: Move to config
        gamma = config.SCHEDULER.GAMMA
    )

    # Should model weights be loaded from warmup?
    if config.MODEL.CONTINUE_FROM_WARMUP:
        weights = torch.load(config.MODEL.WARMUP_WEIGHTS)
        if "state_dict" in weights:
            weights = weights['state_dict']
        generator.load_state_dict(weights, strict=False)


    # Init Tensorboard writer to store train and test info
    # also save the config used in this run to Tensorboard
    writer = SummaryWriter(f"samples/logs/{config.EXP.NAME}")
    writer.add_text("Config/Params", config.get_all_params())

    for epoch in range(config.EXP.START_EPOCH, config.EXP.N_EPOCHS):
        print(f"Beginning train epoch: {epoch+1}")

        # ----------------
        #  Train
        # ----------------
        generator.train()
        discriminator.train()

        # Set the real sample label to 1, and the false sample label to 0
        real_label = torch.full([config.DATA.BATCH_SIZE, 1], 1.0 - config.EXP.LABEL_SMOOTHING, dtype=torch.float, device=config.DEVICE)
        fake_label = torch.full([config.DATA.BATCH_SIZE, 1], 0.0, dtype=torch.float, device=config.DEVICE)

        for batch_num, (gt, lr) in enumerate(train_dataloader):

            # Transfer in-memory data to CUDA devices to speed up training
            gt = gt.to(device=config.DEVICE, non_blocking=True)
            lr = lr.to(device=config.DEVICE, non_blocking=True)

            # ----------------
            #  Update Generator
            # ----------------
            for p in discriminator.parameters():
                p.requires_grad = False
            
            generator.zero_grad()

            sr = generator(lr)
            g_loss = torch.tensor(0.0, device=config.DEVICE)
            for name, criterion in config.MODEL.G_LOSS.CRITERIONS.items():
                weight = config.MODEL.G_LOSS.CRITERION_WEIGHTS[name]

                if name == 'Adversarial':
                    loss = criterion(discriminator(sr), real_label)
                else:
                    loss = criterion(sr, gt)
                
                g_loss = g_loss + (loss * weight)
                loss_values[name] = loss.item() # Used for logging to Tensorboard

            g_loss.backward()
            g_optimizer.step()

            # --------------------
            #  Update Discriminator
            # --------------------
            for p in discriminator.parameters():
                p.requires_grad = True
            
            discriminator.zero_grad()

            pred_gt = discriminator(gt)
            loss_real = adversarial_criterion(pred_gt, real_label)

            pred_sr = discriminator(sr.detach().clone())
            loss_fake = adversarial_criterion(pred_sr, fake_label)

            d_loss = loss_real + loss_fake

            d_loss.backward()
            g_optimizer.step()

            # -------------
            #  Log Progress
            # -------------
            if batch_num % config.LOG_TRAIN_PERIOD != 0:
                continue
            
            # Log to TensorBoard
            batches_done = batch_num + epoch * len(train_dataloader)
            writer.add_scalar("Train/D_Loss", d_loss.item(), batches_done)
            writer.add_scalar("Train/G_Loss", g_loss.item(), batches_done)
            for name, loss in loss_values.items():
                writer.add_scalar(f"Train/G_{name}", loss, batches_done)
            writer.add_scalar("Train/D(GT)_Probability", torch.sigmoid_(torch.mean(pred_gt.detach())).item(), batches_done)
            writer.add_scalar("Train/D(SR)_Probability", torch.sigmoid_(torch.mean(pred_sr.detach())).item(), batches_done)

            # Print to terminal / log
            print(f"[Epoch {epoch+1}/{config.EXP.N_EPOCHS}] [Batch {batch_num}/{len(train_dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [G losses: {loss_values}]")

        # Update learning rates
        g_scheduler.step()
        d_scheduler.step()

        # ----------------
        #  Validate
        # ----------------
        generator.eval()

        psnr, ssim = _validate(generator, test_dataloader, config)

        # Print training log information
        if epoch % config.LOG_VALIDATION_PERIOD == 0:
            print(f"[Test: {epoch}/{config.EXP.N_EPOCHS}] [PSNR: {psnr}] [SSIM: {ssim}]")

        # Write avg PSNR and SSIM to Tensorflow and logs
        writer.add_scalar(f"Test/PSNR", psnr, epoch + 1)
        writer.add_scalar(f"Test/SSIM", ssim, epoch + 1)
        

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