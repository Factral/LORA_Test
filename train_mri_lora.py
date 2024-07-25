import argparse
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from utils import save_metrics, AverageMeter, save_npy_metric, get_weights, save_reconstructed_images
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import numpy as np
from models.networks import UNetRes
import peft
from torchsummary import summary
from pathlib import Path
from load_mri import load_dataset
from torchvision import transforms


def add_gaussian_noise(img, mean=0.0, std=1.5):
    noise = torch.randn_like(img) * std + mean
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)


def get_dataloader_with_noise(batch_size, img_size, data_dir, dataset_name, train=True, mean=0.0, std=1.0):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Lambda(lambda img: add_gaussian_noise(img, mean, std))
    ])

    dataset = load_dataset(
        dataset_name, data_dir, transform, train=train
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=4, pin_memory=True
    )
    
    return dataloader


def initialize_model(n_channels, device, use_lora, lora_rank, learning_rate):
    
    model = UNetRes(in_nc=n_channels, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, 
                    act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose").to(device)
    
    get_weights()
    model.load_state_dict(torch.load('weights/drunet_color.pth'))
    
    if use_lora:
        print(f'Using LORA with rank {lora_rank}')
        weights_lora = [n for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        config = peft.LoraConfig(r=lora_rank, target_modules=weights_lora)
        model = peft.get_peft_model(model, config)
        model.print_trainable_parameters()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    return model, optimizer


def main(args):
    path_name = f'{args.experiment_name}_lr_{args.lr}_b_{args.batch_size}_e_{args.epochs}_r_{args.rank}_lora_{args.lora}'


    args.save_path = args.save_path + path_name
    if os.path.exists(args.save_path):
        print("Experiment already done")
        exit() 

    torch.manual_seed(args.seed)

    images_path, model_path, metrics_path = save_metrics(f'{args.save_path}')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device) 


    data_dir = Path("./data")
    trainloader = get_dataloader_with_noise(args.batch_size, 128, data_dir, 'fastmri_knee_singlecoil', train=True, mean=0.0, std=1.0)
    valoader = get_dataloader_with_noise(args.batch_size, 128, data_dir, 'fastmri_knee_singlecoil', train=False, mean=0.0, std=1.0)


    model, optimizer = initialize_model(2, device, args.lora, args.rank, args.lr)
    print(summary(model, (2, 128, 128)))

    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler()

    wandb.login(key="fe0119224af6709c85541483adf824cec731879e")

    wandb.init(
        project="lora-mri-test",
        name=path_name,
        config=args,
    )

    current_psnr = 0

    for epoch in range(args.epochs):
        model.train()

        train_loss = AverageMeter()
        train_ssim = AverageMeter()
        train_psnr = AverageMeter()
        val_loss = AverageMeter()
        val_ssim = AverageMeter()
        val_psnr = AverageMeter()

        data_loop_train = tqdm(trainloader, colour='red')
        for _, train_data in data_loop_train:

            clean, noisy = train_data
            clean, noisy = clean.to(device), noisy.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred = model(noisy)
                loss_train = criterion(pred, clean)
            
            scaler.scale(loss_train).backward()
            scaler.step(optimizer)
            scaler.update()


            train_loss.update(loss_train.item())
            train_ssim.update(SSIM(pred, clean).item())
            train_psnr.update(PSNR(pred, clean).item())

            data_loop_train.set_description(f'Epoch: {epoch+1}/{args.epochs}')
            data_loop_train.set_postfix(loss=train_loss.avg, ssim=train_ssim.avg, psnr=train_psnr.avg)
        
        data_loop_val = tqdm(valoader, total=len(valoader), colour='green')
        with torch.inference_mode():
            model.eval()
            for val_data in data_loop_val:

                clean, noisy = val_data
                clean, noisy = clean.to(device), noisy.to(device)

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = model(noisy)
                    loss_val = criterion(pred, clean)

                val_loss.update(loss_val.item())
                val_ssim.update(SSIM(pred, clean).item())
                val_psnr.update(PSNR(pred, clean).item())

                data_loop_val.set_description(f'Epoch: {epoch+1}/{args.epochs}')
                data_loop_val.set_postfix(loss=val_loss.avg, ssim=val_ssim.avg, psnr=val_psnr.avg)

        if val_psnr.avg > current_psnr:
            current_psnr = val_psnr.avg
            torch.save(model.state_dict(), f'{model_path}/model.pth')

        recs_array, psnr_imgs, ssim_imgs = save_reconstructed_images(noisy[:,:,:,:], clean, pred, 3, 2, images_path, f'reconstructed_images_{epoch}', PSNR, SSIM)
        recs_images = wandb.Image(recs_array, caption=f'Epoch: {epoch}\nReal\nRec\nPSNRs: {psnr_imgs}\nSSIMs: {ssim_imgs}')


        wandb.log({
            'train_loss': train_loss.avg,
            'train_ssim': train_ssim.avg,
            'train_psnr': train_psnr.avg,
            'val_loss': val_loss.avg,
            'val_ssim': val_ssim.avg,
            'val_psnr': val_psnr.avg,
            'recs_images': recs_images,
        })

    wandb.finish()
 

if __name__ == '__main__':
 
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--epochs', type=int, default='20')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_path', type=str, default='weights/')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--lora', default=False, action=argparse.BooleanOptionalAction, help='train with lora')
    parser.add_argument('--experiment_name', type=str, default='lora_mri1')
 
    args = parser.parse_args()

    main(args)
