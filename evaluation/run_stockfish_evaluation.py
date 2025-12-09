#!/usr/bin/env python3
"""CLI entry point for Stockfish multi-level evaluation."""

import argparse
import os
import sys
import shutil

from stockfish_evaluator import StockfishEvaluator


def find_stockfish() -> str:
    """Try to find Stockfish in common locations.

    Returns:
        Path to Stockfish binary

    Raises:
        FileNotFoundError: If Stockfish not found
    """
    # Common Stockfish paths
    common_paths = [
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
        "/usr/local/bin/stockfish",
        shutil.which("stockfish")  # Check PATH
    ]

    for path in common_paths:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError(
        "Stockfish not found. Please install it:\n"
        "  Ubuntu/Debian: sudo apt-get install stockfish\n"
        "  Or download from: https://stockfishchess.org/download/\n"
        "  Then use --stockfish-path to specify location"
    )


def main():
    """Main entry point for Stockfish evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate chess bot against Stockfish at multiple Elo levels"
    )

    parser.add_argument(
        "--bot-elo",
        type=int,
        default=3000,
        help="Bot Elo rating (default: 3000)"
    )

    parser.add_argument(
        "--start-elo",
        type=int,
        default=1500,
        help="Starting Stockfish Elo (default: 1500)"
    )

    parser.add_argument(
        "--end-elo",
        type=int,
        default=3000,
        help="Ending Stockfish Elo (default: 3000)"
    )

    parser.add_argument(
        "--elo-step",
        type=int,
        default=100,
        help="Elo increment per level (default: 100)"
    )

    parser.add_argument(
        "--games-per-level",
        type=int,
        default=10,
        help="Games to play at each Elo level (default: 10)"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/david/chess-bot-3000/hf_model_chess_100m",
        help="Path to HuggingFace model"
    )

    parser.add_argument(
        "--stockfish-path",
        type=str,
        default=None,
        help="Path to Stockfish binary (auto-detect if not specified)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="stockfish_evaluation_results.json",
        help="Output JSON file (default: stockfish_evaluation_results.json)"
    )

    args = parser.parse_args()

    # Find Stockfish
    try:
        stockfish_path = args.stockfish_path or find_stockfish()
        print(f"Using Stockfish: {stockfish_path}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Create evaluator
    evaluator = StockfishEvaluator(
        bot_elo=args.bot_elo,
        start_elo=args.start_elo,
        end_elo=args.end_elo,
        elo_step=args.elo_step,
        games_per_level=args.games_per_level,
        model_path=args.model_path,
        stockfish_path=stockfish_path
    )

    print("Using model path:", args.model_path)

    # Run evaluation
    results = evaluator.run_evaluation()

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), args.output)

    configuration = {
        "bot_elo": args.bot_elo,
        "start_elo": args.start_elo,
        "end_elo": args.end_elo,
        "elo_step": args.elo_step,
        "games_per_level": args.games_per_level,
        "model_path": args.model_path,
        "stockfish_path": stockfish_path
    }

    evaluator.results_tracker.save_json(output_path, configuration)
    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
