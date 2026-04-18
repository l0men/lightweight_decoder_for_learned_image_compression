#!/bin/bash -l
#SBATCH --job-name=compressai_array_frozen
#SBATCH --array=1-3
#SBATCH --output=logs/ref_coco_%A_%a.out
#SBATCH --error=logs/ref_coco_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=P100,3090

#SBATCH --mail-type=END,FAIL   
#SBATCH --mail-user=/** EMAIL_HIDDEN_FOR_PRIVACY ***/

#SBATCH --export=WANDB_API_KEY= /** WAND_API_KEY_HIDDEN_FOR_PRIVACY ***/
#SBATCH --export=WANDB_MODE=online
#SBATCH --export WANDB_JOB_TYPE="training"

set -x

pwd
echo $CUDA_VISIBLE_DEVICES

# 1. Chargement de l'environnement
module purge
module load miniconda3/25.5.1  
source $(conda info --base)/etc/profile.d/conda.sh
conda activate compressai_3090

# 2. Définition des paramètres variables
QUALITY_LIST=(2 4 6 8)
CURRENT_QUALITY=${QUALITY_LIST[$SLURM_ARRAY_TASK_ID]}
echo "I'm task $SLURM_ARRAY_TASK_ID and my quality = $CURRENT_QUALITY"

# 3. Définition des chemins
PROJECT_DIR="/home/infres/lmennrath-25/PRIM_Project"
DATASET_DIR="/home/infres/lmennrath-25/PRIM_Project/datasets/COCO"
SCRIPT_PATH="/home/infres/lmennrath-25/PRIM_Project/repos/frozen_training/training/train_reference_big.py"

# --- 🌟 LA MAGIE DE LA REPRISE AUTOMATIQUE 🌟 ---
# On définit le nom du fichier checkpoint attendu pour cette qualité
CHECKPOINT_FILE="checkpoints/bmshj2018-factorized_ref_COCO_QUAL${CURRENT_QUALITY}_checkpoint.pth.tar"

# On crée une variable vide par défaut
CHECKPOINT_ARG=""

# On vérifie si le fichier existe physiquement sur le disque
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "✅ Checkpoint trouvé ! Reprise de l'entraînement depuis $CHECKPOINT_FILE"
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT_FILE"
else
    echo "🚀 Aucun checkpoint trouvé. Démarrage de l'entraînement de zéro."
fi
# ------------------------------------------------

# 4. Lancement du script (avec la variable dynamique à la fin)
echo "Lancement de l'entrainement sur le dataset : $DATASET_DIR"

python -u $SCRIPT_PATH \
  -d $DATASET_DIR \
  -m bmshj2018-factorized \
  --quality $CURRENT_QUALITY \
  --epochs 600 \
  -lr 1e-4 \
  -n 4 \
  --batch-size 16 \
  --cuda \
  --save \
  $CHECKPOINT_ARG

conda deactivate

echo "End of task $SLURM_ARRAY_TASK_ID"