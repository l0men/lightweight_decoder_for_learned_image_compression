import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.layers import GDN
from compressai.losses import RateDistortionLoss
from compressai.models.utils import deconv
from compressai.zoo import bmshj2018_factorized, image_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATASET_DIR = PROJECT_ROOT / "datasets" / "kodak"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

def depthwise_separable_deconv(in_channels, out_channels, kernel_size=5, stride=2):
    """Depthwise Separable Transposed Convolution"""
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size//2, 
                          output_padding=stride-1, groups=in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=1)
    )


def depthwise_separable_deconv_full(in_channels, out_channels, kernel_size=5, stride=2):
    """Depthwise full decoder block used during depthwise_full training."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=stride - 1,
            groups=out_channels,
        ),
    )

class DepthwiseDecoderMixV2(nn.Module):
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

class DepthwiseDecoder(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.net = nn.Sequential(
                depthwise_separable_deconv_full(M, N),
                GDN(N, inverse=True),
                depthwise_separable_deconv_full(N, N),
                GDN(N, inverse=True),
                depthwise_separable_deconv_full(N, N),
                GDN(N, inverse=True),
                depthwise_separable_deconv_full(N, 3),
            )

    def forward(self, x):
        return self.net(x)

def load_depthwise_model(checkpoint_path, model_name='bmshj2018-factorized', quality=1, device='cuda'):
    """
    Load a model with the trained DepthwiseDecoder from a checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth.tar file
        model_name: Model name (e.g. 'bmshj2018-factorized')
        quality: Quality level (1-8) - must match the value used during training
        device: 'cuda' ou 'cpu'
    
    Returns:
        net: The loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    net = image_models[model_name](quality=quality, pretrained=True)
    
    latent_channels = net.g_s[0].in_channels
    N = 192 if quality > 5 else 128
    print(f"M: {latent_channels}, N: {N} (quality {quality})")
    
    if quality > 5:
        net.g_s = DepthwiseDecoder(N=192, M=latent_channels)
    else:
        net.g_s = DepthwiseDecoder(N=128, M=latent_channels)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        keys_to_remove = [k for k in state_dict.keys() if '_offset' in k or '_quantized_cdf' in k or '_cdf_length' in k]
        for k in keys_to_remove:
            del state_dict[k]

        nn.Module.load_state_dict(net, state_dict, strict=False)
        net.update()
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.6f}")
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total number of parameters ({model_name}): {total_params}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    net = net.to(device)
    net.eval()
    net.update(force=True)
    return net

def load_depthwise_models_for_qualities_big(qualities, model_name='bmshj2018-factorized', 
                                         checkpoint_dir=CHECKPOINT_DIR, device='cuda'):
    """
    Load several depthwise models for different quality levels.
    
    Args:
        qualities: List of quality levels (e.g. [1, 2, 3])
        model_name: Base model name
        checkpoint_dir: Checkpoint directory
        device: 'cuda' ou 'cpu'
    
    Returns:
        dict: {quality: model} for each quality
    """
    models = {}
    
    for q in qualities:
        checkpoint_name = f"{model_name}_depthwise_full_COCO_QUAL{q}_best_loss.pth.tar"
        checkpoint_path = Path(checkpoint_dir) / checkpoint_name
        
        try:
            models[q] = load_depthwise_model(checkpoint_path, model_name, quality=q, device=device)
        except FileNotFoundError as e:
            print(f"⚠ Quality {q}: {e}")
    
    return models

def load_mix_model_v2(checkpoint_path, model_name='bmshj2018-factorized', quality=1, device='cuda'):
    """
    Load a model with the trained DepthwiseDecoderMix from a checkpoint.
    
    Args:
        checkpoint_path: Path to the .pth.tar file
        model_name: Model name (e.g. 'bmshj2018-factorized')
        quality: Quality level (1-8) - must match the value used during training
        device: 'cuda' ou 'cpu'
    
    Returns:
        net: The loaded model in eval mode
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    net = image_models[model_name](quality=quality, pretrained=True)
    
    latent_channels = net.g_s[0].in_channels
    N = 192 if quality > 5 else 128
    print(f"M: {latent_channels}, N: {N} (quality {quality})")
    
    if quality > 5:
        net.g_s = DepthwiseDecoderMixV2(N=192, M=latent_channels)
    else:
        net.g_s = DepthwiseDecoderMixV2(N=128, M=latent_channels)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        keys_to_remove = [k for k in state_dict.keys() if '_offset' in k or '_quantized_cdf' in k or '_cdf_length' in k]
        for k in keys_to_remove:
            del state_dict[k]

        nn.Module.load_state_dict(net, state_dict, strict=False)
        net.update()
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.6f}")
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total number of parameters ({model_name}): {total_params}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    net = net.to(device)
    net.eval()
    net.update(force=True)
    return net

def load_mix_models_for_qualities_v2_big(qualities, model_name='bmshj2018-factorized', 
                                         checkpoint_dir=CHECKPOINT_DIR, device='cuda'):
    """
    Load several depthwise models for different quality levels.
    
    Args:
        qualities: List of quality levels (e.g. [1, 2, 3])
        model_name: Base model name
        checkpoint_dir: Checkpoint directory
        device: 'cuda' ou 'cpu'
    
    Returns:
        dict: {quality: model} for each quality
    """
    models = {}
    
    for q in qualities:
        checkpoint_name = f"{model_name}_depthwise_mix_frozen_v2_COCO_QUAL{q}_best_loss.pth.tar"
        checkpoint_path = Path(checkpoint_dir) / checkpoint_name
        
        try:
            models[q] = load_mix_model_v2(checkpoint_path, model_name, quality=q, device=device)
        except FileNotFoundError as e:
            print(f"⚠ Quality {q}: {e}")
    
    return models

def load_custom_model(checkpoint_path, model_name='bmshj2018-factorized', quality=1, device='cuda'):
    """Load a custom model from a checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    net = image_models[model_name](quality=quality, pretrained=True)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        keys_to_remove = [k for k in state_dict.keys() if '_offset' in k or '_quantized_cdf' in k or '_cdf_length' in k]
        for k in keys_to_remove:
            del state_dict[k]

        nn.Module.load_state_dict(net, state_dict, strict=False)
        net.update()
        print(f"✓ Model loaded from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.6f}")
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total number of parameters ({model_name}): {total_params}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    net = net.to(device)
    net.eval()
    net.update(force=True)
    return net

def load_custom_models_for_qualities(qualities, model_name='bmshj2018-factorized',
                                         checkpoint_dir=CHECKPOINT_DIR, device='cuda'):
    """Load several models trained on the custom dataset for different quality levels.
    
    Note: These models were all trained with quality=1 (N=128, M=192)
    but with different lambdas for different rate-distortion trade-offs.
    """

    models = {}

    for q in qualities:
        checkpoint_name = f"{model_name}_ref_COCO_QUAL{q}_best_loss.pth.tar"
        checkpoint_path = Path(checkpoint_dir) / checkpoint_name

        try:
            models[q] = load_custom_model(checkpoint_path, model_name, quality=q, device=device)
        except FileNotFoundError as e:
            print(f"⚠ Quality {q}: {e}")
    
    return models

def test_model_on_dataset(model, dataset_path, device, num_images=24, lmbda=1e-2):
    """Test the model on images from the dataset."""
    
    print(f"\n{'='*70}")
    print(f"Testing model on {dataset_path}")
    print(f"{'='*70}\n")
    
    test_transforms = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])
    
    test_dataset = ImageFolder(dataset_path, split="test", transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    criterion = RateDistortionLoss(lmbda=lmbda)
    
    total_loss = 0
    total_bpp = 0
    total_mse = 0
    num_samples = 0
    total_inference_time = 0
    
    with torch.no_grad():
        for i, img in enumerate(test_loader):
            if i >= num_images:
                break
            
            img = img.to(device)
            
            start_time = time.time()
            out = model(img)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            out_criterion = criterion(out, img)
            
            loss = out_criterion["loss"].item()
            bpp = out_criterion["bpp_loss"].item()
            mse = out_criterion["mse_loss"].item()
            
            total_loss += loss
            total_bpp += bpp
            total_mse += mse
            num_samples += 1

    avg_bpp = total_bpp / num_samples
    avg_mse = total_mse / num_samples
    avg_psnr = 10 * math.log10(1 / max(avg_mse, 1e-10))
    avg_inference_time = total_inference_time / num_samples
    
    print(f"\n📊 Performance summary:")
    print(f"   Average PSNR: {avg_psnr:.2f} dB")
    print(f"   Average BPP:  {avg_bpp:.4f}")
    print(f"   Average inference time: {avg_inference_time:.4f} seconds")
    
    return avg_psnr, avg_bpp, avg_inference_time

def load_pretrained_model(quality=3):
    """Load a pretrained bmshj-factorized model with the specified quality."""
    
    print(f"\n📂 Loading pretrained model: bmshj-factorized")
    print(f"   Quality: {quality}")
    
    device = "cpu"
    
    try:
        print(f"\n🔧 Creating the model...")
        net = bmshj2018_factorized(quality, pretrained=True)
        net = net.to(device)
        net.eval()
        
        print(f"✅ Pretrained model loaded and ready!")
        print(f"   Device: {device}")
        
        return net 
    
    except Exception as e:
        print(f"❌ Error while loading the model: {e}")
        return None

def gflops_to_kmacs_per_pixel(gflops, image_size=(720, 1280)):
    h, w = image_size
    total_pixels = h * w
    kmacs_px = (gflops * 1e6) / (2 * total_pixels)
    return kmacs_px

def measure_complexity(model, image_size=(720, 1280)):
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.rand(1, 3, image_size[0], image_size[1]).to(device)
    
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()
    gflops = total_flops / 1e9
    
    print(f"   Size {image_size[0]}x{image_size[1]}: {gflops:.3f} GFLOPs")
    return gflops_to_kmacs_per_pixel(gflops)

def count_decoder_params(model):
    """Count only the decoder parameters (g_s)."""
    return sum(p.numel() for p in model.g_s.parameters())

if __name__ == "__main__":
    checkpoint_dir = CHECKPOINT_DIR
    qualities = [1, 2, 3, 4, 5, 6, 7, 8]
    qualities_big = [2, 4, 6, 8]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    models_ref = load_custom_models_for_qualities(
        qualities=qualities_big,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    models_mix_v2_big = load_mix_models_for_qualities_v2_big(
        qualities=qualities_big,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    models_depthwise_big = load_depthwise_models_for_qualities_big(
        qualities=qualities_big,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    dataset_path = DATASET_DIR

    pretrained_psnr = []
    pretrained_bpp = []
    mix_v2_big_psnr = []
    mix_v2_big_bpp = []
    depthwise_big_psnr = []
    depthwise_big_bpp = []
    ref_psnr = []
    ref_bpp = []

    for q in qualities:
        print(f"\n🔍 Testing pretrained model for quality {q}...")
        pretrained_model = load_pretrained_model(quality=q)
        if pretrained_model is not None:
            avg_psnr, avg_bpp, _ = test_model_on_dataset(pretrained_model, dataset_path, device)
            pretrained_psnr.append(avg_psnr)
            pretrained_bpp.append(avg_bpp)
        else:
            print(f"⚠ Pretrained model for quality {q} is unavailable, skipping test.")
    
    for q in qualities_big:
        if q in models_mix_v2_big:
            print(f"\n🔍 Testing mix depthwise v2 model for quality {q}...")
            avg_psnr, avg_bpp, avg_inference_time = test_model_on_dataset(models_mix_v2_big[q], dataset_path, device)
            mix_v2_big_psnr.append(avg_psnr)
            mix_v2_big_bpp.append(avg_bpp)
        else:
            print(f"⚠ Mix v2 model for quality {q} is unavailable, skipping test.")

    for q in qualities_big:
        if q in models_depthwise_big:
            print(f"\n🔍 Testing depthwise v2 model for quality {q}...")
            avg_psnr, avg_bpp, avg_inference_time = test_model_on_dataset(models_depthwise_big[q], dataset_path, device)
            depthwise_big_psnr.append(avg_psnr)
            depthwise_big_bpp.append(avg_bpp)
        else:
            print(f"⚠ Depthwise v2 model for quality {q} is unavailable, skipping test.")
    
    for q in qualities_big:
        if q in models_ref:
            print(f"\n🔍 Testing reference model for quality {q}...")
            avg_psnr, avg_bpp, avg_inference_time = test_model_on_dataset(models_ref[q], dataset_path, device)
            ref_psnr.append(avg_psnr)
            ref_bpp.append(avg_bpp)
        else:
            print(f"⚠ Reference model for quality {q} is unavailable, skipping test.")

    pretrained_complexity = measure_complexity(pretrained_model)
    mix_v2_big_complexity = measure_complexity(models_mix_v2_big[8])
    depthwise_big_complexity = measure_complexity(models_depthwise_big[8])
    ref_complexity = measure_complexity(models_ref[8])

    mix_v2_big_dec_params = count_decoder_params(models_mix_v2_big[8])
    depthwise_big_dec_params = count_decoder_params(models_depthwise_big[8])
    ref_dec_params = count_decoder_params(models_ref[8])

    print(f"\n{'='*70}")
    print("📊 Decoder-only parameter count (g_s):")
    print(f"   Mix V2 Big (COCO)     : {mix_v2_big_dec_params:,} parameters")
    print(f"   Depthwise Big (COCO)  : {depthwise_big_dec_params:,} parameters")
    print(f"   Reference (COCO)      : {ref_dec_params:,} parameters")
    print(f"{'='*70}\n")

    plt.figure(figsize=(10, 6))
    plt.plot(pretrained_bpp, pretrained_psnr, linestyle='dashed', marker='x', color='b', label=f'bmshj2018-factorized (pretrained) [{pretrained_complexity:.2f} kMACs/pixel]')
    plt.plot(mix_v2_big_bpp, mix_v2_big_psnr, marker='^', color='orange', label=f'Depthwise Mixed Decoder [{mix_v2_big_complexity:.2f} kMACs/pixel]')
    plt.plot(depthwise_big_bpp, depthwise_big_psnr, marker='^', color='m', label=f'Full Depthwise Decoder [{depthwise_big_complexity:.2f} kMACs/pixel]')
    plt.plot(ref_bpp, ref_psnr, marker='^', color='r', label=f'Reference [{ref_complexity:.2f} kMACs/pixel]')
    plt.legend()
    plt.title("PSNR vs BPP curve for different models [Complexity in kMACs/pixel]")
    plt.xlabel("BPP (bits per pixel)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    
    save_path = RESULTS_DIR / 'rd_curve.png'
    os.makedirs(save_path.parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot saved: {save_path}")




 