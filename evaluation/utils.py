"""Utility classes and functions for chess bot evaluation."""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class GameResult:
    """Result of a single game."""
    game_number: int
    bot_color: str  # "white" or "black"
    outcome: str    # "bot_win", "opponent_win", "draw"
    reason: str     # "checkmate", "stalemate", "illegal_move", etc.
    move_count: int
    illegal_move_player: Optional[str] = None  # "bot" or "opponent" if applicable


class ResultsTracker:
    """Tracks and reports statistics across all games."""

    def __init__(self):
        self.bot_wins = 0
        self.gpt5_wins = 0
        self.draws = 0
        self.bot_illegal_moves = 0
        self.gpt5_illegal_moves = 0
        self.games_played = 0
        self.game_details = []

    def record_game(self, result: GameResult):
        """Record a game result."""
        self.games_played += 1
        self.game_details.append(result)

        # Update win/draw counts
        if result.outcome == "bot_win":
            self.bot_wins += 1
        elif result.outcome == "gpt5_win":
            self.gpt5_wins += 1
        elif result.outcome == "draw":
            self.draws += 1

        # Update illegal move counts
        if result.illegal_move_player == "bot":
            self.bot_illegal_moves += 1
        elif result.illegal_move_player == "gpt5":
            self.gpt5_illegal_moves += 1

    def get_summary(self) -> dict:
        """Get summary statistics as dictionary."""
        return {
            "total_games": self.games_played,
            "bot_wins": self.bot_wins,
            "gpt5_wins": self.gpt5_wins,
            "draws": self.draws,
            "bot_win_rate": (self.bot_wins / self.games_played * 100) if self.games_played > 0 else 0.0,
            "gpt5_win_rate": (self.gpt5_wins / self.games_played * 100) if self.games_played > 0 else 0.0,
            "draw_rate": (self.draws / self.games_played * 100) if self.games_played > 0 else 0.0,
            "bot_illegal_moves": self.bot_illegal_moves,
            "gpt5_illegal_moves": self.gpt5_illegal_moves
        }

    def print_report(self):
        """Print formatted results report to console."""
        print("Evaluation Complete!")
        print("=" * 50)
        print(f"Total games: {self.games_played}")
        print(f"Bot wins: {self.bot_wins} ({self.bot_wins / self.games_played * 100:.1f}%)")
        print(f"GPT-5 wins: {self.gpt5_wins} ({self.gpt5_wins / self.games_played * 100:.1f}%)")
        print(f"Draws: {self.draws} ({self.draws / self.games_played * 100:.1f}%)")
        print(f"Bot illegal moves: {self.bot_illegal_moves}")
        print(f"GPT-5 illegal moves: {self.gpt5_illegal_moves}")

    def save_json(self, filepath: str, configuration: dict):
        """Save results to JSON file."""
        data = {
            "configuration": configuration,
            "summary": self.get_summary(),
            "games": [asdict(game) for game in self.game_details]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class StockfishResultsTracker:
    """Tracks results across multiple Elo levels."""

    def __init__(self):
        self.results_by_elo = {}  # {1500: {"wins": 5, "losses": 3, "draws": 2}}
        self.elo_levels = []

    def record_level(self, elo: int, wins: int, losses: int, draws: int):
        """Record results for an Elo level."""
        total = wins + losses + draws
        self.elo_levels.append(elo)
        self.results_by_elo[elo] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "total_games": total,
            "win_rate": (wins / total * 100) if total > 0 else 0.0,
            "loss_rate": (losses / total * 100) if total > 0 else 0.0,
            "draw_rate": (draws / total * 100) if total > 0 else 0.0
        }

    def print_report(self):
        """Print formatted report by Elo level."""
        print("\nStockfish Evaluation Results")
        print("=" * 80)
        print(f"{'Elo':<8} {'Games':<8} {'Wins':<8} {'Losses':<8} {'Draws':<8} {'Win %':<10}")
        print("-" * 80)

        for elo in sorted(self.elo_levels):
            r = self.results_by_elo[elo]
            print(f"{elo:<8} {r['total_games']:<8} {r['wins']:<8} {r['losses']:<8} "
                  f"{r['draws']:<8} {r['win_rate']:.1f}%")

        print("=" * 80)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            "elo_levels": sorted(self.elo_levels),
            "results_by_elo": self.results_by_elo
        }

    def save_json(self, filepath: str, configuration: dict):
        """Save results to JSON."""
        data = {
            "configuration": configuration,
            "results_by_elo": self.results_by_elo,
            "summary": {
                "elo_range": [min(self.elo_levels), max(self.elo_levels)] if self.elo_levels else [0, 0],
                "total_levels": len(self.elo_levels)
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
