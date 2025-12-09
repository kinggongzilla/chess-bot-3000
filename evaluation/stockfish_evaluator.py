"""Multi-level Elo evaluator for testing bot against Stockfish."""

from chess_bot_player import ChessBotPlayer
from stockfish_player import StockfishPlayer
from game_engine import GameEngine
from utils import ResultsTracker, StockfishResultsTracker


class StockfishEvaluator:
    """Evaluates chess bot against Stockfish at multiple Elo levels."""

    def __init__(self, bot_elo: int, start_elo: int, end_elo: int,
                 elo_step: int, games_per_level: int, model_path: str,
                 stockfish_path: str):
        """Initialize Stockfish evaluator.

        Args:
            bot_elo: Chess bot Elo rating
            start_elo: Starting Stockfish Elo
            end_elo: Ending Stockfish Elo
            elo_step: Elo increment per level
            games_per_level: Number of games at each level
            model_path: Path to HuggingFace model
            stockfish_path: Path to Stockfish binary
        """
        self.bot_elo = bot_elo
        self.start_elo = start_elo
        self.end_elo = end_elo
        self.elo_step = elo_step
        self.games_per_level = games_per_level
        self.model_path = model_path
        self.stockfish_path = stockfish_path
        self.results_tracker = StockfishResultsTracker()

    def run_evaluation(self) -> dict:
        """Run evaluation across all Elo levels.

        Returns:
            Dictionary with summary statistics
        """
        # Print header
        print("Stockfish Multi-Level Evaluation")
        print("=" * 80)
        print(f"Bot Elo: {self.bot_elo}")
        print(f"Stockfish Elo Range: {self.start_elo} - {self.end_elo} (step: {self.elo_step})")
        print(f"Games per level: {self.games_per_level}")
        print()

        # Load bot model once
        print("Loading chess bot model...")
        bot_player = ChessBotPlayer(self.model_path, self.bot_elo)
        print("Model loaded.\n")

        # Generate Elo levels
        elo_levels = range(self.start_elo, self.end_elo + 1, self.elo_step)
        total_levels = len(list(elo_levels))

        # Evaluate at each level
        for level_idx, stockfish_elo in enumerate(elo_levels, 1):
            print(f"Level {level_idx}/{total_levels}: Stockfish Elo {stockfish_elo}")
            print("-" * 80)

            # Create Stockfish player for this Elo level
            stockfish_player = StockfishPlayer(self.stockfish_path, stockfish_elo)

            # Track results for this level
            level_tracker = ResultsTracker()

            # Play games at this level
            for game_num in range(self.games_per_level):
                # Alternate colors: first half bot is white, second half black
                bot_color = "white" if game_num < self.games_per_level // 2 else "black"

                # Setup players
                if bot_color == "white":
                    white_player = bot_player
                    black_player = stockfish_player
                else:
                    white_player = stockfish_player
                    black_player = bot_player

                # Play game
                game_engine = GameEngine(white_player, black_player, max_moves=500)

                print(f"  Game {game_num + 1}/{self.games_per_level} (Bot: {bot_color})...",
                      end=" ", flush=True)

                try:
                    result = game_engine.play_game(game_num + 1, bot_color)
                    level_tracker.record_game(result)

                    # Print outcome
                    if result.outcome == "bot_win":
                        print(f"✓ Bot wins ({result.reason})")
                    elif result.outcome == "draw":
                        print(f"= Draw ({result.reason})")
                    else:
                        print(f"✗ Stockfish wins ({result.reason})")

                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

            # Record level results
            summary = level_tracker.get_summary()
            self.results_tracker.record_level(
                stockfish_elo,
                summary['bot_wins'],
                summary['gpt5_wins'],  # Actually stockfish wins
                summary['draws']
            )

            # Print level summary
            print(f"  Level Summary: {summary['bot_wins']}W - "
                  f"{summary['gpt5_wins']}L - {summary['draws']}D "
                  f"({summary['bot_win_rate']:.1f}% win rate)")
            print()

            # Clean up Stockfish player
            stockfish_player.close()

        # Print final report
        self.results_tracker.print_report()

        return self.results_tracker.get_summary()
