# chess-bot-3000

A language model trained from scratch on chess games to learn chess-specific patterns and strategies. This project uses the Nanotron framework to train a 100M parameter SmolLM3-based model on Lichess game data.

## Overview

This project trains a transformer language model on chess games represented in UCI (Universal Chess Interface) notation. The model learns to predict chess moves given game context, including player Elo ratings and game outcomes.

## Key Features

- **Model Architecture**: 100M parameter Qwen2-style transformer (12 layers, 768 hidden size, 8 attention heads)
- **Custom Tokenizer**: Based on nsarrazin/chessformer with special tokens for:
  - Game boundaries: `<BOG>` (beginning of game), `<EOG>` (end of game)
  - Player Elo ratings: `<WHITE:1500>`, `<BLACK:2000>`, etc. (0-3500 in 100-point increments)
  - Game outcomes: `<WHITE_WIN>`, `<BLACK_WIN>`, `<DRAW>`
- **Training Data**: Lichess game database in UCI notation format
- **Framework**: Nanotron with distributed training support (PyTorch, FSDP)
- **Hardware**: Optimized for HPC environments with multi-GPU support

## Project Structure

```
chess-bot-3000/
├── data/                           # Data processing scripts
│   ├── download_lichess.py         # Download games from Lichess database
│   ├── preprocess_lichess.py       # Convert PGN to UCI format
│   └── preprocess_add_elo_and_result.py  # Add Elo and result tokens
├── nanotron/                       # Nanotron framework (submodule/fork)
├── nanotron_train_configs/         # Training configuration files
│   └── 100m_smollm3_chess_leonardo_jan24.yaml
├── slurm_scripts/                  # HPC job submission scripts
│   └── leonardo.sh                 # SLURM script for Leonardo supercomputer
├── tokenizer/                      # Custom tokenizer
│   ├── tokenizer_with_special_tokens.py
│   ├── uci_tokenizer_with_special_tokens/  # Tokenizer files
│   └── pgn_tokenizer_with_special_tokens/  # Alternative PGN tokenizer
└── nanotron_venv/                  # Python virtual environment
```

## Setup

### Prerequisites

- Python 3.11+
- CUDA-capable GPU(s)
- 64GB+ RAM recommended for data preprocessing
- Access to Lichess database (for training data)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chess-bot-3000
```

2. Create and activate virtual environment:
```bash
python -m venv nanotron_venv
source nanotron_venv/bin/activate
```

3. Install dependencies:
```bash
cd nanotron
pip install -e .
cd ..
pip install transformers datasets pandas
```

## Data Processing Pipeline

### 1. Download Lichess Data

Download games from the Lichess database:
```bash
python data/download_lichess.py
```

### 2. Preprocess to UCI Format

Convert PGN format to UCI notation:
```bash
python data/preprocess_lichess.py
```

This converts games from Standard Algebraic Notation (SAN):
```
1. e4 e5 2. Nf3 Nc6 ...
```

To UCI format with special tokens:
```
<BOG> e2e4 e7e5 g1f3 b8c6 ... <EOG>
```

### 3. Add Elo and Result Tokens

Augment the data with player ratings and game outcomes:
```bash
python data/preprocess_add_elo_and_result.py
```

Final format:
```
<BOG> <WHITE:1600> <BLACK:1550> <WHITE_WIN> e2e4 e7e5 g1f3 ... <EOG>
```

## Training

### Local Training

```bash
cd nanotron
torchrun --nproc_per_node=4 run_train.py \
  --config-file ../nanotron_train_configs/100m_smollm3_chess_leonardo_jan24.yaml
```

### HPC Training (SLURM)

For training on HPC systems like Leonardo:

```bash
sbatch slurm_scripts/leonardo.sh
```

The SLURM script is configured for:
- 4 GPUs on a single node
- 48-hour time limit
- Data parallel training (DP=4)
- Automatic checkpoint saving every 100 steps

### Training Configuration

Key hyperparameters (see `nanotron_train_configs/100m_smollm3_chess_leonardo_jan24.yaml`):

- **Batch size**: 64 per GPU, 4 gradient accumulation steps (effective batch: 1024)
- **Sequence length**: 256 tokens
- **Learning rate**: 6e-4 with cosine decay to 6e-5
- **Warmup**: 520 steps (~2% of training)
- **Total steps**: 26,400
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95, weight decay=0.1)
- **Precision**: bfloat16

## Model Details

### Architecture

- **Base model**: SmolLM3-100M architecture (Qwen2-style)
- **Parameters**: ~100M
- **Layers**: 12 transformer blocks
- **Hidden size**: 768
- **Attention heads**: 8 (2 KV heads for GQA)
- **Intermediate size**: 3072
- **Vocabulary size**: 4687 tokens
- **Max sequence length**: 256 tokens

### Special Features

- Flash Attention 2 for efficient training
- Grouped Query Attention (GQA)
- Fused RMS normalization and rotary embeddings
- Document masking for multi-game batches
- Z-loss for training stability

## Tokenizer

The custom tokenizer is based on `nsarrazin/chessformer` with added special tokens:

- **Chess moves**: Standard UCI format (e.g., `e2e4`, `e7e5`, `e1g1` for castling)
- **Game structure**: `<BOG>`, `<EOG>`, `[PAD]`
- **Elo tokens**: 72 tokens for player ratings (`<WHITE:0>` to `<WHITE:3500>`, `<BLACK:0>` to `<BLACK:3500>`)
- **Outcome tokens**: `<WHITE_WIN>`, `<BLACK_WIN>`, `<DRAW>`

## Monitoring

Training progress is logged to Weights & Biases (wandb):

- Project: `chess-bot-3000`
- Run name: `100m-jan24`
- Metrics: loss, perplexity, learning rate, throughput

## Checkpoints

Checkpoints are saved every 100 steps to:
```
nanotron/checkpoints/smollm3-100m-chess/
```

Each checkpoint includes:
- Model weights
- Optimizer state
- Training metadata
- Config files

## Future Work

- Evaluate model on chess puzzles and game completion tasks
- Experiment with larger model sizes (300M, 1B parameters)
- Fine-tune for specific chess styles or openings
- Add support for chess variants (Fischer Random, etc.)
- Implement reinforcement learning from self-play

## References

- Nanotron: https://github.com/huggingface/nanotron
- Lichess Database: https://database.lichess.org/
- Chessformer: https://huggingface.co/nsarrazin/chessformer
- UCI Protocol: https://www.chessprogramming.org/UCI

## License

[Add your license here]

## Acknowledgments

- Lichess for providing open chess game data
- HuggingFace for the Nanotron training framework
- CINECA for computational resources on Leonardo supercomputer
