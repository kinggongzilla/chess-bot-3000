# Stockfish Multi-Level Elo Evaluation

Evaluation system for testing the chess bot against Stockfish at incrementally increasing Elo levels.

## Overview

This system evaluates your chess bot by playing against Stockfish at multiple Elo levels (1500-3000 in 100-point steps by default). It plays 10 games at each level, alternating colors, and reports win/loss/draw percentages to help determine your bot's approximate strength.

## Prerequisites

Install Stockfish:
```bash
# Ubuntu/Debian
sudo apt-get install stockfish

# Or download from https://stockfishchess.org/download/
```

## Usage

### Basic Usage (Default Settings)
```bash
python evaluation/run_stockfish_evaluation.py
```

This will:
- Test bot at Elo 3000 (configurable)
- Play against Stockfish from 1500 to 3000 Elo in 100-point steps (16 levels)
- Play 10 games per level
- Alternate colors (5 games as White, 5 as Black per level)

### Custom Configuration

```bash
# Quick test (smaller range, fewer games)
python evaluation/run_stockfish_evaluation.py \
  --start-elo 1500 \
  --end-elo 1800 \
  --games-per-level 4

# Different bot Elo
python evaluation/run_stockfish_evaluation.py --bot-elo 2500

# Custom Stockfish path
python evaluation/run_stockfish_evaluation.py \
  --stockfish-path /usr/local/bin/stockfish

# More games for better statistics
python evaluation/run_stockfish_evaluation.py --games-per-level 20

# Custom output file
python evaluation/run_stockfish_evaluation.py \
  --output my_stockfish_results.json
```

## Command-Line Options

- `--bot-elo INT` - Bot Elo rating (default: 3000)
- `--start-elo INT` - Starting Stockfish Elo (default: 1500)
- `--end-elo INT` - Ending Stockfish Elo (default: 3000)
- `--elo-step INT` - Elo increment per level (default: 100)
- `--games-per-level INT` - Games to play at each level (default: 10)
- `--model-path PATH` - Path to HuggingFace model (default: ../hf_model_chess_100m)
- `--stockfish-path PATH` - Path to Stockfish binary (auto-detected if not specified)
- `--output FILE` - Output JSON file (default: stockfish_evaluation_results.json)

## Output

### Console Output
```
Stockfish Multi-Level Evaluation
================================================================================
Bot Elo: 3000
Stockfish Elo Range: 1500 - 3000 (step: 100)
Games per level: 10

Loading chess bot model...
Model loaded.

Level 1/16: Stockfish Elo 1500
--------------------------------------------------------------------------------
  Game 1/10 (Bot: White)... ✓ Bot wins (checkmate)
  Game 2/10 (Bot: White)... ✓ Bot wins (checkmate)
  ...
  Level Summary: 10W - 0L - 0D (100.0% win rate)

...

Stockfish Evaluation Results
================================================================================
Elo      Games    Wins     Losses   Draws    Win %
--------------------------------------------------------------------------------
1500     10       10       0        0        100.0%
1600     10       10       0        0        100.0%
1700     10       9        1        0        90.0%
...
3000     10       2        7        1        20.0%
================================================================================

Results saved to: evaluation/stockfish_evaluation_results.json
```

### JSON Output

Results are saved with detailed statistics:
```json
{
  "configuration": {
    "bot_elo": 3000,
    "start_elo": 1500,
    "end_elo": 3000,
    "elo_step": 100,
    "games_per_level": 10,
    "stockfish_path": "/usr/games/stockfish"
  },
  "results_by_elo": {
    "1500": {
      "wins": 10,
      "losses": 0,
      "draws": 0,
      "total_games": 10,
      "win_rate": 100.0,
      "loss_rate": 0.0,
      "draw_rate": 0.0
    },
    ...
  },
  "summary": {
    "elo_range": [1500, 3000],
    "total_levels": 16
  }
}
```

## Evaluation Details

### Color Alternation
- At each Elo level, first 5 games: bot plays White
- Last 5 games: bot plays Black
- This ensures fair evaluation across both sides

### Time Control
- Stockfish: 2 seconds per move
- Bot: No time limit (generates move immediately)

### Game Termination
- Checkmate, stalemate, insufficient material
- 50-move rule, threefold repetition
- Illegal move = immediate loss
- Max 500 moves per game = draw

## Performance Estimates

Approximate evaluation times:
- Quick test (1500-1800, 4 games): ~10-15 minutes
- Medium test (1500-2400, 10 games): ~2-3 hours
- Full evaluation (1500-3000, 10 games): ~4-5 hours

## Files

- `stockfish_player.py` - Stockfish engine wrapper
- `stockfish_evaluator.py` - Multi-level evaluation orchestrator
- `run_stockfish_evaluation.py` - CLI entry point
- `utils.py` - Extended with `StockfishResultsTracker`

## Troubleshooting

### Stockfish Not Found
```
Error: Stockfish not found. Please install it:
  Ubuntu/Debian: sudo apt-get install stockfish
  Or download from: https://stockfishchess.org/download/
```

Solution: Install Stockfish or use `--stockfish-path` to specify location

### Engine Terminated Error
If Stockfish crashes during evaluation, the system will skip that game and continue.

## Interpreting Results

- **100% win rate**: Bot is significantly stronger than that Elo level
- **~50% win rate**: Bot's approximate Elo strength
- **<20% win rate**: Bot is significantly weaker than that Elo level

Use the results to estimate your bot's playing strength by finding where it maintains around 50% win rate.
