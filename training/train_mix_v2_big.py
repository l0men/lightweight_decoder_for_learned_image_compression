import argparse
import random
import shutil
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import image_models
from compressai.layers import GDN

from compressai.models.utils import conv, deconv

import wandb

def depthwise_separable_deconv(in_channels, out_channels, kernel_size=5, stride=2):
    """Depthwise Separable Transposed Convolution"""
    return nn.Sequential(
         nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size//2, 
                          output_padding=stride-1, groups=in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=1)
       
    )

# Depthwise Decoder Mix
class DepthwiseDecoderMix(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.net = nn.Sequential(
                depthwise_separable_deconv(M, N),
                GDN(N, inverse=True),
                depthwise_separable_deconv(N, N),
                GDN(N, inverse=True),
                depthwise_separable_deconv(N, N),
                GDN(N, inverse=True),
                deconv(N, 3),
            )  

    def forward(self, x):
        return self.net(x)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    model.g_a.eval()  # freeze analysis transform
    device = next(model.parameters()).device

    mse_loss_meter = AverageMeter()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net["x_hat"], d)
        out_criterion.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.g_s.parameters(), clip_max_norm)
        optimizer.step()

        mse_loss_meter.update(out_criterion.item(), d.size(0))

        if i % 100 == 0:
            print(f"Train epoch {epoch}: [{i}/{len(train_dataloader)}] Loss: {out_criterion.item():.4f}")
    
    return {"mse": mse_loss_meter.avg}


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    mse_loss_meter = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net["x_hat"], d)

            mse_loss_meter.update(out_criterion.item(), d.size(0))

    print(
        f"Test epoch {epoch}: Average losses: "
        f"MSE loss: {mse_loss_meter.avg:.6f}"
    )

    return {"mse": mse_loss_meter.avg,}


def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=3,
        help="Quality level for the pretrained model (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    run = wandb.init(
        project="compressai-frozen_test", 
        config=args,
        name=f"{args.model}_depthwise_mix_frozen_v2_COCO_QUAL{args.quality}", 
        group="experiment_batch_1"
    )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size, pad_if_needed=True), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.Resize(args.patch_size[0]), transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = image_models[args.model](quality=args.quality, pretrained=True)

    latent_channels = net.g_s[0].in_channels
    if (args.quality > 5):
        net.g_s = DepthwiseDecoderMix(N=192, M=latent_channels)
    else:
        net.g_s = DepthwiseDecoderMix(N=128, M=latent_channels)
    net = net.to(device)

    for param in net.parameters():
        param.requires_grad = False # Freeze all parameters
    
    for param in net.g_s.parameters():
        param.requires_grad = True  # Unfreeze decoder parameters

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer = optim.Adam(net.g_s.parameters(), lr=args.learning_rate)
    aux_optimizer = None
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = nn.MSELoss()
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        except Exception as e:
            print(f"Warning: Cannot load optimizer state ({e}). Starting from scratch.")

    print(f"Is encoder trainable? {list(net.g_a.parameters())[0].requires_grad}")
    print(f"Is decoder trainable? {list(net.g_s.parameters())[0].requires_grad}")
    
    best_loss = float("inf")

    save_dir = "checkpoints"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Early stopping parameters
    patience = 30
    epochs_without_improvement = 0
    min_lr_threshold = 1e-7

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_metrics = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            None,
            epoch,
            args.clip_max_norm,
        )
        val_metrics = test_epoch(epoch, test_dataloader, net, criterion)
        loss = val_metrics["mse"]

        wandb.log(
            {
                "train/mse": train_metrics["mse"],
                "val/mse": val_metrics["mse"],
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch
        )

        lr_scheduler.step(loss)

        current_lr = optimizer.param_groups[0]['lr']
        if loss < best_loss:
            is_best = True
            best_loss = loss
            epochs_without_improvement = 0
        else:
            is_best = False
            epochs_without_improvement += 1

        if args.save:
                net.update()
                
                current_name = f"{args.model}_depthwise_mix_frozen_v2_COCO_QUAL{args.quality}_checkpoint.pth.tar"
                best_name = f"{args.model}_depthwise_mix_frozen_v2_COCO_QUAL{args.quality}_best_loss.pth.tar"

                save_path = os.path.join(save_dir, current_name)
                best_path = os.path.join(save_dir, best_name)

                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    filename=save_path,
                    best_filename=best_path
                )

        if epochs_without_improvement >= patience:
            if current_lr <= min_lr_threshold:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}!")
                print(f"No improvement for {patience} epochs AND learning rate ({current_lr}) reached minimum threshold.")
                break
            else:
                print(f"⚠️ Patience reached ({epochs_without_improvement}/{patience}), but continuing as learning rate ({current_lr}) hasn't reached minimum.")

    wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
