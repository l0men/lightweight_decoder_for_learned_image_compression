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


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


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
        # Le nom du run sur le dashboard inclura le lambda
        name=f"{args.model}_ref_COCO_QUAL{args.quality}", 
        # Optionnel : Groupe les runs pour comparer facilement
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
    net_empty = image_models[args.model](quality=args.quality, pretrained=False)  # On crée une instance vide pour récupérer la structure du decoder

    #  --- MODIF 2 : Change the decoder ---
    net.g_s = net_empty.g_s  # On remplace le decoder pré-entraîné par un nouveau non entraîné

    net = net.to(device)

    for param in net.parameters():
        param.requires_grad = False # Freeze all parameters
    
    for param in net.g_s.parameters():
        param.requires_grad = True  # Unfreeze decoder parameters

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer = optim.Adam(net.g_s.parameters(), lr=args.learning_rate)  # Remove the aux optimizer
    aux_optimizer = None
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = nn.MSELoss() # Use only MSE for the decoder training
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
            print(f"Warning: Impossible de charger l'état de l'optimiseur ({e}). On repart à zéro.")

    print(f"Encodeur trainable ? {list(net.g_a.parameters())[0].requires_grad}") # Doit être False
    print(f"Décodeur trainable ? {list(net.g_s.parameters())[0].requires_grad}") # Doit être True
    
    best_loss = float("inf")

    save_dir = "checkpoints"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # --- Paramètres de l'Early Stopping ---
    patience = 30  # Nombre d'époques sans amélioration avant de s'inquiéter
    epochs_without_improvement = 0
    min_lr_threshold = 1e-7  # Le seuil de LR en dessous duquel on autorise l'arrêt
    # --------------------------------------

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

        # 2. Récupération du Learning Rate actuel
        current_lr = optimizer.param_groups[0]['lr']
        # 3. Logique du compteur de patience
        if loss < best_loss:
            is_best = True
            best_loss = loss
            epochs_without_improvement = 0  # On reset le compteur car on a fait un meilleur score
        else:
            is_best = False
            epochs_without_improvement += 1

        if args.save:
                # Mettre à jour les tables CDF avant de sauvegarder pour remplir les buffers
                net.update()
                
                # On construit un nom unique pour ne pas écraser les autres runs
                # Exemple : checkpoints/bmshj2018-factorized_0.0130_checkpoint.pth.tar
                current_name = f"{args.model}_ref_COCO_QUAL{args.quality}_checkpoint.pth.tar"
                best_name = f"{args.model}_ref_COCO_QUAL{args.quality}_best_loss.pth.tar"

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
                    filename=save_path,      # Le fichier de l'époque actuelle
                    best_filename=best_path  # Le fichier du meilleur modèle
                )

        # 4. Condition d'Early Stopping intelligente
        if epochs_without_improvement >= patience:
            if current_lr <= min_lr_threshold:
                print(f"\n🛑 Early stopping déclenché à l'époque {epoch} !")
                print(f"Aucune amélioration depuis {patience} époques ET le LR ({current_lr}) a atteint le seuil minimum.")
                break  # On quitte la boucle d'entraînement
            else:
                print(f"⚠️ Patience atteinte ({epochs_without_improvement}/{patience}), mais on continue car le LR ({current_lr}) n'est pas encore au minimum.")

    wandb.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
