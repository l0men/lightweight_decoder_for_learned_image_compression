import torch
import os
import sys
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import math
import matplotlib.pyplot as plt
import time
import torch.profiler
from fvcore.nn import FlopCountAnalysis

# Ajouter le chemin vers compressai
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CompressAI_tests'))

from compressai.zoo import image_models
from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.zoo import bmshj2018_factorized
from compressai.models.utils import conv, deconv
from main import DepthwiseDecoder
from train_mix import DepthwiseDecoderMix
from train_upsample import DepthwiseDecoderUpsample
from train_upsample_2_v2 import DepthwiseDecoderUpsample2v2

from compressai.layers import GDN

from compressai.models.utils import conv, deconv

def depthwise_separable_deconv(in_channels, out_channels, kernel_size=5, stride=2):
    """Depthwise Separable Transposed Convolution"""
    return nn.Sequential(
         # Channelwise first (groups = out_channels)
        nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size//2, 
                          output_padding=stride-1, groups=in_channels),
        # Pointwise then
        nn.Conv2d(in_channels, out_channels, kernel_size=1)
       
    )

# --- MODIF 1 : Depthwise Decoder ---
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

def load_depthwise_model(checkpoint_path, model_name='bmshj2018-factorized', quality=1, device='cuda'):
    """
    Charge un modèle avec le décodeur DepthwiseDecoder entraîné depuis un checkpoint.
    
    Args:
        checkpoint_path: Chemin vers le fichier .pth.tar
        model_name: Nom du modèle (ex: 'bmshj2018-factorized')
        quality: Niveau de qualité (1-8) - doit correspondre à celui utilisé pour l'entraînement
        device: 'cuda' ou 'cpu'
    
    Returns:
        net: Le modèle chargé en mode eval
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Créer le modèle de base avec la qualité correspondante
    net = image_models[model_name](quality=quality, pretrained=True)
    
    # Récupérer le nombre de canaux latents AVANT de remplacer le décodeur
    latent_channels = net.g_s[0].in_channels
    N = 192 if quality > 5 else 128
    print(f"M: {latent_channels}, N: {N} (qualité {quality})")
    
    # Remplacer le décodeur par la version depthwise (comme dans main.py)
    if (quality > 5):
         net.g_s = DepthwiseDecoder(N=192, M=latent_channels)
    else:
        net.g_s = DepthwiseDecoder(N=128, M=latent_channels)
    
    # Charger le checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        # print(f"Checkpoint trouvé: {state_dict.keys()}")
        
        # Supprimer les buffers d'entropy bottleneck vides qui causent des size mismatch
        keys_to_remove = [k for k in state_dict.keys() if '_offset' in k or '_quantized_cdf' in k or '_cdf_length' in k]
        for k in keys_to_remove:
            del state_dict[k]
        
        # Utiliser directement nn.Module.load_state_dict pour éviter la logique custom de CompressAI
        nn.Module.load_state_dict(net, state_dict, strict=False)
        # Mettre à jour les tables de l'entropy bottleneck
        net.update()
        print(f"✓ Modèle chargé depuis: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.6f}")
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'Total number of parameters ({model_name}): {total_params}')
    else:
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    
    net = net.to(device)
    net.eval()
    net.update(force=True)
    return net

# def load_depthwise_models_for_qualities(qualities, model_name='bmshj2018-factorized', 
#                                          checkpoint_dir='../checkpoints', device='cuda'):
#     """
#     Charge plusieurs modèles depthwise pour différentes qualités.
    
#     Args:
#         qualities: Liste des niveaux de qualité (ex: [1, 2, 3])
#         model_name: Nom du modèle base
#         checkpoint_dir: Répertoire des checkpoints
#         device: 'cuda' ou 'cpu'
    
#     Returns:
#         dict: {quality: model} pour chaque qualité
#     """
#     models = {}
    
#     for q in qualities:
#         checkpoint_name = f"{model_name}_depthwise_2_QUAL{q}_best_loss.pth.tar"
#         checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
#         try:
#             models[q] = load_depthwise_model(checkpoint_path, model_name, quality=q, device=device)
#         except FileNotFoundError as e:
#             print(f"⚠ Qualité {q}: {e}")
    
#     return models

def load_depthwise_models_for_qualities_big(qualities, model_name='bmshj2018-factorized', 
                                         checkpoint_dir='../checkpoints', device='cuda'):
    """
    Charge plusieurs modèles depthwise pour différentes qualités.
    
    Args:
        qualities: Liste des niveaux de qualité (ex: [1, 2, 3])
        model_name: Nom du modèle base
        checkpoint_dir: Répertoire des checkpoints
        device: 'cuda' ou 'cpu'
    
    Returns:
        dict: {quality: model} pour chaque qualité
    """
    models = {}
    
    for q in qualities:
        checkpoint_name = f"{model_name}_depthwise_full_COCO_QUAL{q}_best_loss.pth.tar"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        try:
            models[q] = load_depthwise_model(checkpoint_path, model_name, quality=q, device=device)
        except FileNotFoundError as e:
            print(f"⚠ Qualité {q}: {e}")
    
    return models

# def load_mix_model(checkpoint_path, model_name='bmshj2018-factorized', quality=1, device='cuda'):
#     """
#     Charge un modèle avec le décodeur DepthwiseDecoder entraîné depuis un checkpoint.
    
#     Args:
#         checkpoint_path: Chemin vers le fichier .pth.tar
#         model_name: Nom du modèle (ex: 'bmshj2018-factorized')
#         quality: Niveau de qualité (1-8) - doit correspondre à celui utilisé pour l'entraînement
#         device: 'cuda' ou 'cpu'
    
#     Returns:
#         net: Le modèle chargé en mode eval
#     """
#     device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
#     # Créer le modèle de base avec la qualité correspondante
#     net = image_models[model_name](quality=quality, pretrained=True)
    
#     # Récupérer le nombre de canaux latents AVANT de remplacer le décodeur
#     latent_channels = net.g_s[0].in_channels
#     N = 192 if quality > 5 else 128
#     print(f"M: {latent_channels}, N: {N} (qualité {quality})")
    
#     # Remplacer le décodeur par la version depthwise (comme dans main.py)
#     if (quality > 5):
#          net.g_s = DepthwiseDecoderMix(N=192, M=latent_channels)
#     else:
#         net.g_s = DepthwiseDecoderMix(N=128, M=latent_channels)
    
#     # Charger le checkpoint
#     if os.path.exists(checkpoint_path):
#         checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
#         state_dict = checkpoint['state_dict']
#         # print(f"Checkpoint trouvé: {state_dict.keys()}")
        
#         # Supprimer les buffers d'entropy bottleneck vides qui causent des size mismatch
#         keys_to_remove = [k for k in state_dict.keys() if '_offset' in k or '_quantized_cdf' in k or '_cdf_length' in k]
#         for k in keys_to_remove:
#             del state_dict[k]
        
#         # Utiliser directement nn.Module.load_state_dict pour éviter la logique custom de CompressAI
#         nn.Module.load_state_dict(net, state_dict, strict=False)
#         # Mettre à jour les tables de l'entropy bottleneck
#         net.update()
#         print(f"✓ Modèle chargé depuis: {checkpoint_path}")
#         print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.6f}")
#         total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
#         print(f'Total number of parameters ({model_name}): {total_params}')
#     else:
#         raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    
#     net = net.to(device)
#     net.eval()
#     net.update(force=True)
#     return net

# def load_mix_models_for_qualities(qualities, model_name='bmshj2018-factorized', 
#                                          checkpoint_dir='../checkpoints', device='cuda'):
#     """
#     Charge plusieurs modèles depthwise pour différentes qualités.
    
#     Args:
#         qualities: Liste des niveaux de qualité (ex: [1, 2, 3])
#         model_name: Nom du modèle base
#         checkpoint_dir: Répertoire des checkpoints
#         device: 'cuda' ou 'cpu'
    
#     Returns:
#         dict: {quality: model} pour chaque qualité
#     """
#     models = {}
    
#     for q in qualities:
#         checkpoint_name = f"{model_name}_depthwise_mix_frozen_QUAL{q}_best_loss.pth.tar"
#         checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
#         try:
#             models[q] = load_mix_model(checkpoint_path, model_name, quality=q, device=device)
#         except FileNotFoundError as e:
#             print(f"⚠ Qualité {q}: {e}")
    
#     return models

def load_mix_model_v2(checkpoint_path, model_name='bmshj2018-factorized', quality=1, device='cuda'):
    """
    Charge un modèle avec le décodeur DepthwiseDecoder entraîné depuis un checkpoint.
    
    Args:
        checkpoint_path: Chemin vers le fichier .pth.tar
        model_name: Nom du modèle (ex: 'bmshj2018-factorized')
        quality: Niveau de qualité (1-8) - doit correspondre à celui utilisé pour l'entraînement
        device: 'cuda' ou 'cpu'
    
    Returns:
        net: Le modèle chargé en mode eval
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Créer le modèle de base avec la qualité correspondante
    net = image_models[model_name](quality=quality, pretrained=True)
    
    # Récupérer le nombre de canaux latents AVANT de remplacer le décodeur
    latent_channels = net.g_s[0].in_channels
    N = 192 if quality > 5 else 128
    print(f"M: {latent_channels}, N: {N} (qualité {quality})")
    
    # Remplacer le décodeur par la version depthwise (comme dans main.py)
    if (quality > 5):
         net.g_s = DepthwiseDecoderMixV2(N=192, M=latent_channels)
    else:
        net.g_s = DepthwiseDecoderMixV2(N=128, M=latent_channels)
    
    # Charger le checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        # print(f"Checkpoint trouvé: {state_dict.keys()}")
        
        # Supprimer les buffers d'entropy bottleneck vides qui causent des size mismatch
        keys_to_remove = [k for k in state_dict.keys() if '_offset' in k or '_quantized_cdf' in k or '_cdf_length' in k]
        for k in keys_to_remove:
            del state_dict[k]
        
        # Utiliser directement nn.Module.load_state_dict pour éviter la logique custom de CompressAI
        nn.Module.load_state_dict(net, state_dict, strict=False)
        # Mettre à jour les tables de l'entropy bottleneck
        net.update()
        print(f"✓ Modèle chargé depuis: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.6f}")
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'Total number of parameters ({model_name}): {total_params}')
    else:
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    
    net = net.to(device)
    net.eval()
    net.update(force=True)
    return net

# def load_mix_models_for_qualities_v2(qualities, model_name='bmshj2018-factorized', 
#                                          checkpoint_dir='../checkpoints', device='cuda'):
#     """
#     Charge plusieurs modèles depthwise pour différentes qualités.
    
#     Args:
#         qualities: Liste des niveaux de qualité (ex: [1, 2, 3])
#         model_name: Nom du modèle base
#         checkpoint_dir: Répertoire des checkpoints
#         device: 'cuda' ou 'cpu'
    
#     Returns:
#         dict: {quality: model} pour chaque qualité
#     """
#     models = {}
    
#     for q in qualities:
#         checkpoint_name = f"{model_name}_depthwise_mix_frozen_v2_QUAL{q}_best_loss.pth.tar"
#         checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
#         try:
#             models[q] = load_mix_model_v2(checkpoint_path, model_name, quality=q, device=device)
#         except FileNotFoundError as e:
#             print(f"⚠ Qualité {q}: {e}")
    
#     return models

def load_mix_models_for_qualities_v2_big(qualities, model_name='bmshj2018-factorized', 
                                         checkpoint_dir='../checkpoints', device='cuda'):
    """
    Charge plusieurs modèles depthwise pour différentes qualités.
    
    Args:
        qualities: Liste des niveaux de qualité (ex: [1, 2, 3])
        model_name: Nom du modèle base
        checkpoint_dir: Répertoire des checkpoints
        device: 'cuda' ou 'cpu'
    
    Returns:
        dict: {quality: model} pour chaque qualité
    """
    models = {}
    
    for q in qualities:
        checkpoint_name = f"{model_name}_depthwise_mix_frozen_v2_COCO_QUAL{q}_best_loss.pth.tar"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        try:
            models[q] = load_mix_model_v2(checkpoint_path, model_name, quality=q, device=device)
        except FileNotFoundError as e:
            print(f"⚠ Qualité {q}: {e}")
    
    return models

def load_custom_model(checkpoint_path, model_name='bmshj2018-factorized', quality=1, device='cuda'):
    """Charge un modèle custom depuis un checkpoint."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Créer le modèle de base avec la qualité correspondante
    net = image_models[model_name](quality=quality, pretrained=True)
    
    
    # Charger le checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        # print(f"Checkpoint trouvé: {state_dict.keys()}")
        
        # Supprimer les buffers d'entropy bottleneck vides qui causent des size mismatch
        keys_to_remove = [k for k in state_dict.keys() if '_offset' in k or '_quantized_cdf' in k or '_cdf_length' in k]
        for k in keys_to_remove:
            del state_dict[k]
        
        # Utiliser directement nn.Module.load_state_dict pour éviter la logique custom de CompressAI
        nn.Module.load_state_dict(net, state_dict, strict=False)
        # Mettre à jour les tables de l'entropy bottleneck
        net.update()
        print(f"✓ Modèle chargé depuis: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.6f}")
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'Total number of parameters ({model_name}): {total_params}')
    else:
        raise FileNotFoundError(f"Checkpoint non trouvé: {checkpoint_path}")
    
    net = net.to(device)
    net.eval()
    net.update(force=True)
    return net

def load_custom_models_for_qualities(qualities, model_name='bmshj2018-factorized',
                                         checkpoint_dir='../checkpoints', device='cuda'):
    """Charge plusieurs modèles entrainés sur un dataset custom pour différentes qualités.
    
    Note: Ces modèles ont tous été entraînés avec quality=1 (N=128, M=192) 
    mais avec différents lambdas pour différents trade-offs rate-distortion.
    """

    models = {}
    # Les lambdas correspondent aux noms de fichiers existants

    for q in qualities:
        # Formater avec 4 décimales pour correspondre aux noms de fichiers
        checkpoint_name = f"{model_name}_ref_COCO_QUAL{q}_best_loss.pth.tar"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        try:
            # Toujours quality=1 car tous les modèles custom ont été entraînés avec cette architecture
            models[q] = load_custom_model(checkpoint_path, model_name, quality=q, device=device)
        except FileNotFoundError as e:
            print(f"⚠ Qualité {q}: {e}")
    
    return models

def test_model_on_dataset(model, dataset_path, device, num_images=24, lmbda=1e-2):
    """Teste le modèle sur des images du dataset"""
    
    print(f"\n{'='*70}")
    print(f"Test du modèle sur {dataset_path}")
    print(f"{'='*70}\n")
    
    # Préparer le dataset
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
            
    # Calculer les moyennes
    avg_bpp = total_bpp / num_samples
    avg_mse = total_mse / num_samples
    avg_psnr = 10 * math.log10(1 / max(avg_mse, 1e-10))
    avg_inference_time = total_inference_time / num_samples
    
    # Afficher les résultats finaux
    print(f"\n📊 Résumé des performances:")
    print(f"   PSNR moyen: {avg_psnr:.2f} dB")
    print(f"   BPP moyen:  {avg_bpp:.4f}")
    print(f"   Temps d'inférence moyen: {avg_inference_time:.4f} secondes")
    
    return avg_psnr, avg_bpp, avg_inference_time

def load_pretrained_model(quality=3):
    """Charge un modèle pré-entraîné bmshj-factorized avec la qualité spécifiée"""
    
    print(f"\n📂 Chargement du modèle pré-entraîné: bmshj-factorized")
    print(f"   Qualité: {quality}")
    
    device = "cpu"
    
    try:
        # Charger le modèle pré-entraîné
        print(f"\n🔧 Création du modèle...")
        net = bmshj2018_factorized(quality, pretrained=True)
        net = net.to(device)
        net.eval()
        
        print(f"✅ Modèle pré-entraîné chargé et prêt!")
        print(f"   Device: {device}")
        
        return net 
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None

def gflops_to_kmacs_per_pixel(gflops, image_size=(720, 1280)):
    h, w = image_size
    total_pixels = h * w
    # On multiplie par 10^6 et on divise par (2 * pixels)
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
    
    print(f"   Taille {image_size[0]}x{image_size[1]} : {gflops:.3f} GFLOPs")
    return gflops_to_kmacs_per_pixel(gflops)


# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les modèles pour 3 qualités différentes
    checkpoint_dir = os.path.dirname(__file__) + '/../checkpoints'
    qualities = [1, 2, 3, 4, 5, 6, 7, 8]  # Ajustez selon vos besoins
    qualities_big = [2, 4, 6, 8]  # Seules les qualités 2, 4, 6 et 8 ont été entraînées sur COCO
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    models_ref = load_custom_models_for_qualities(
        qualities=qualities_big,
        checkpoint_dir=checkpoint_dir,
        device=device
    )

    # models_mix = load_mix_models_for_qualities(
    #     qualities=qualities,
    #     checkpoint_dir=checkpoint_dir,
    #     device=device
    # )

    # models_mix_v2 = load_mix_models_for_qualities_v2(
    #     qualities=qualities,
    #     checkpoint_dir=checkpoint_dir,
    #     device=device
    # )

    # models_depthwise = load_depthwise_models_for_qualities(
    #     qualities=qualities,
    #     checkpoint_dir=checkpoint_dir,
    #     device=device
    # )

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

    
    # Tester chaque modèle sur le dataset de test
    dataset_path = "/home/infres/lmennrath-25/PRIM_Project/datasets/kodak"

    pretrained_psnr = []
    pretrained_bpp = []
    # custom_psnr = []
    # custom_bpp = []
    # mix_psnr = []
    # mix_bpp = []
    # mix_v2_psnr = []
    # mix_v2_bpp = []
    # full_psnr = []
    # full_bpp = []
    mix_v2_big_psnr = []
    mix_v2_big_bpp = []
    depthwise_big_psnr = []
    depthwise_big_bpp = []
    ref_psnr = []
    ref_bpp = []

    # for q in qualities:
    #     if q in models_mix:
    #         print(f"\n🔍 Test du modèle mix depthwise pour qualité {q}...")
    #         avg_psnr, avg_bpp, avg_inference_time = test_model_on_dataset(models_mix[q], dataset_path, device)
    #         mix_psnr.append(avg_psnr)
    #         mix_bpp.append(avg_bpp)
    #     else:
    #         print(f"⚠ Modèle mix pour qualité {q} non disponible, saut du test.")

    for q in qualities:
        print(f"\n🔍 Test du modèle pré-entraîné pour qualité {q}...")
        pretrained_model = load_pretrained_model(quality=q)
        if pretrained_model is not None:
            avg_psnr, avg_bpp, _ = test_model_on_dataset(pretrained_model, dataset_path, device)
            pretrained_psnr.append(avg_psnr)
            pretrained_bpp.append(avg_bpp)
        else:
            print(f"⚠ Modèle pré-entraîné pour qualité {q} non disponible, saut du test.")

    # for q in qualities:
    #     if q in models_mix_v2:
    #         print(f"\n🔍 Test du modèle mix depthwise v2 pour qualité {q}...")
    #         avg_psnr, avg_bpp, avg_inference_time = test_model_on_dataset(models_mix_v2[q], dataset_path, device)
    #         mix_v2_psnr.append(avg_psnr)
    #         mix_v2_bpp.append(avg_bpp)
    #     else:
    #         print(f"⚠ Modèle mix v2 pour qualité {q} non disponible, saut du test.")
    
    # for q in qualities:
    #     if q in models_depthwise:
    #         print(f"\n🔍 Test du modèle full depthwise pour qualité {q}...")
    #         avg_psnr, avg_bpp, avg_inference_time = test_model_on_dataset(models_depthwise[q], dataset_path, device)
    #         full_psnr.append(avg_psnr)
    #         full_bpp.append(avg_bpp)
    #     else:
    #         print(f"⚠ Modèle mix v2 pour qualité {q} non disponible, saut du test.")
    
    for q in qualities_big:
        if q in models_mix_v2_big:
            print(f"\n🔍 Test du modèle mix depthwise v2 pour qualité {q}...")
            avg_psnr, avg_bpp, avg_inference_time = test_model_on_dataset(models_mix_v2_big[q], dataset_path, device)
            mix_v2_big_psnr.append(avg_psnr)
            mix_v2_big_bpp.append(avg_bpp)
        else:
            print(f"⚠ Modèle mix v2 pour qualité {q} non disponible, saut du test.")

    for q in qualities_big:
        if q in models_depthwise_big:
            print(f"\n🔍 Test du modèle depthwise v2 pour qualité {q}...")
            avg_psnr, avg_bpp, avg_inference_time = test_model_on_dataset(models_depthwise_big[q], dataset_path, device)
            depthwise_big_psnr.append(avg_psnr)
            depthwise_big_bpp.append(avg_bpp)
        else:
            print(f"⚠ Modèle depthwise v2 pour qualité {q} non disponible, saut du test.")
    
    for q in qualities_big:
        if q in models_ref:
            print(f"\n🔍 Test du modèle depthwise v2 pour qualité {q}...")
            avg_psnr, avg_bpp, avg_inference_time = test_model_on_dataset(models_ref[q], dataset_path, device)
            ref_psnr.append(avg_psnr)
            ref_bpp.append(avg_bpp)
        else:
            print(f"⚠ Modèle depthwise v2 pour qualité {q} non disponible, saut du test.")

    # for q in qualities:
    #     if q in models_custom:
    #         print(f"\n🔍 Test du modèle custom pour qualité {q}...")
    #         avg_psnr, avg_bpp, avg_inference_time = test_model_on_dataset(models_custom[q], dataset_path, device)
    #         custom_psnr.append(avg_psnr)
    #         custom_bpp.append(avg_bpp)
    #     else:
    #         print(f"⚠ Modèle pour qualité {q} non disponible, saut du test.")

    pretrained_complexity = measure_complexity(pretrained_model)
    
    # mix_v2_complexity = measure_complexity(models_mix_v2[8])
    # depthwise_complexity = measure_complexity(models_depthwise[8])
    mix_v2_big_complexity = measure_complexity(models_mix_v2_big[8])
    depthwise_big_complexity = measure_complexity(models_depthwise_big[8])
    ref_complexity = measure_complexity(models_ref[8])

    def count_decoder_params(model):
        """Compte uniquement les paramètres du décodeur (g_s)"""
        return sum(p.numel() for p in model.g_s.parameters())

    # pretrained_dec_params = count_decoder_params(models_custom[8])
    # mix_v2_dec_params = count_decoder_params(models_mix_v2[8])
    # depthwise_dec_params = count_decoder_params(models_depthwise[8])
    mix_v2_big_dec_params = count_decoder_params(models_mix_v2_big[8])
    depthwise_big_dec_params = count_decoder_params(models_depthwise_big[8])
    ref_dec_params = count_decoder_params(models_ref[8])

    print(f"\n{'='*70}")
    print("📊 Nombre de paramètres du DÉCODEUR uniquement (g_s) :")
    # print(f"   Pretrained (bmshj2018) : {pretrained_dec_params:,} paramètres")
    # print(f"   Mix V2 Depthwise       : {mix_v2_dec_params:,} paramètres")
    # print(f"   Full Depthwise         : {depthwise_dec_params:,} paramètres")
    print(f"   Mix V2 Big (COCO)     : {mix_v2_big_dec_params:,} paramètres")
    print(f"   Depthwise Big (COCO)  : {depthwise_big_dec_params:,} paramètres")
    print(f"   Reference (COCO)      : {ref_dec_params:,} paramètres")
    print(f"{'='*70}\n")

    # custom_psnr = [25.5, 27.3, 28.75, 31, 32.55, 33.8, 35.1, 35.5]
    # custom_bpp = [0.185, 0.35, 0.495, 0.75, 1.13, 1.495, 1.92, 2.25 ]

    # depthwise_psnr = [22.45, 18.2, 24.95, 25, 24.9, 26.75, 27, 27.3]
    # depthwise_bpp = [0.13, 1.3, 0.25, 0.35, 0.43, 0.7, 0.95, 1.42]

    # pretrained_psnr = [25.327981202859075, 26.536315323702624, 27.855493335644404, 29.516379678534754, 31.247053222380323, 34.00489463780341, 36.150511306307564, 38.51602016523591]
    # pretrained_bpp = [0.13721988070756197, 0.21094316678742567, 0.3217161490271489, 0.49330520319441956, 0.725018839041392, 1.0900484969218571, 1.5235166400671005, 2.058193857471148]   

    # Tracer les résultats
    plt.figure(figsize=(10, 6))
    # plt.plot(frozen_bpp, frozen_psnr, marker='o', color='m', label='Depthwise Decoders (Frozen Encoder)')
    plt.plot(pretrained_bpp, pretrained_psnr, linestyle='dashed', marker='x', color='b', label=f'bmshj2018-factorized (pretrained) [{pretrained_complexity:.2f} kMACs/pixel]')
    # plt.plot(custom_bpp, custom_psnr, marker='o', linestyle='dashed', color='r', label=f'Reference (CLIC)  [{pretrained_complexity:.2f} kMACs/pixel]')
    # plt.plot(mix_v2_bpp, mix_v2_psnr, marker='o', linestyle='dashed', color='orange', label=f'Depthwise Mixed Decoder (CLIC) [{mix_v2_complexity:.2f} kMACs/pixel]')
    plt.plot(mix_v2_big_bpp, mix_v2_big_psnr, marker='^', color='orange', label=f'Depthwise Mixed Decoder [{mix_v2_big_complexity:.2f} kMACs/pixel]')
    # plt.plot(mix_v2_bpp, mix_v2_psnr, marker='o', color='g', label='Depthwise Mixed Decoders V2 (Depthwise + Pointwise)')
    # plt.plot(full_bpp, full_psnr, marker='o', linestyle='dashed', color='m', label=f'Full Depthwise Decoder (CLIC)[{depthwise_complexity:.2f} kMACs/pixel]')
    plt.plot(depthwise_big_bpp, depthwise_big_psnr, marker='^', color='m', label=f'Full Depthwise Decoder [{depthwise_big_complexity:.2f} kMACs/pixel]')
    plt.plot(ref_bpp, ref_psnr, marker='^', color='r', label=f'Reference [{ref_complexity:.2f} kMACs/pixel]')
    plt.legend()
    plt.title("Courbe PSNR vs BPP pour différents modèles [Complexité en kMACs/pixel]")
    plt.xlabel("BPP (bits per pixel)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    
    # Enregistrer le plot au format PNG
    save_path = os.path.join(os.path.dirname(__file__), 'results', 'rd_curve.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot enregistré: {save_path}")
    # print(pretrained_psnr)
    # print(pretrained_bpp)





 