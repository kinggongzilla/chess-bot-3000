#!/bin/bash -l
#SBATCH --account=EUHPC_D18_005
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=48:00:00
#SBATCH --chdir=/leonardo_scratch/fast/EUHPC_D18_005/david/chess-bot-3000
#SBATCH --output=/leonardo_scratch/fast/EUHPC_D18_005/david/outputs/chess.out
#SBATCH --error=/leonardo_scratch/fast/EUHPC_D18_005/david/outputs/chess.err

source nanotron_env/bin/activate

export HF_HOME="/leonardo_work/EUHPC_D18_005/david/hf-datasets-cache"

export WANDB_MODE=offline

cd nanotron

wandb enabled
wandb offline

CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 run_train.py --config-file /leonardo_scratch/fast/EUHPC_D18_005/david/chess-bot-3000/nanotron_train_configs/100m_smollm3_chess_leonardo_jan24.yaml