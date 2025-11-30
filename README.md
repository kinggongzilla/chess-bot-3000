# chess-bot-3000

A language model trained from scratch on chess games to learn chess-specific patterns and strategies. This project uses the Nanotron framework to train a SmolLM3-based model on Lichess game data.

Try the model with huggingface: https://huggingface.co/daavidhauser/chess-bot-3000-100m

*Disclaimer*: The documentation in this readme is LLM generated and may contain mistakes.

## Overview

This project trains a transformer language model on chess games represented in UCI (Universal Chess Interface) notation. The model learns to predict chess moves given game context, including player Elo ratings and game outcomes.

## Key Features

- **Model Architecture**: 100M & 250m parameter Qwen2-style transformer. Can be changed in the yaml files in `/nanotron_train_configs/` directory
- **UCI notation Tokenizer**: A tokenizer for chess moves in UCI notation including the following special tokens:
  - Game boundaries: `<BOG>` (beginning of game), `<EOG>` (end of game)
  - Player Elo ratings: `<WHITE:1500>`, `<BLACK:2000>`, etc. (0-3500 in 100-point increments)
  - Game outcomes: `<WHITE_WIN>`, `<BLACK_WIN>`, `<DRAW>`
- **Training Data**: Lichess game database in UCI notation format
- **Framework**: Nanotron with distributed training support

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

### Training Configuration

Key hyperparameters (see `nanotron_train_configs/100m_smollm3_chess_leonardo_jan24.yaml`):

- **Batch size**: 64 per GPU, 4 gradient accumulation steps (effective batch: 1024)
- **Sequence length**: 256 tokens
- **Learning rate**: 6e-4 with cosine decay to 6e-5
- **Warmup**: 520 steps (~2% of training)
- **Total steps**: 26,400
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95, weight decay=0.1)
- **Precision**: bfloat16
