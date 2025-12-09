"""Stockfish player using python-chess engine module."""

import chess
import chess.engine


class StockfishPlayer:
    """Player that uses Stockfish chess engine."""

    def __init__(self, stockfish_path: str, elo: int, time_limit: float = 2.0):
        """Initialize Stockfish player.

        Args:
            stockfish_path: Path to Stockfish binary
            elo: Elo rating (800-3500)
            time_limit: Time per move in seconds
        """
        self.stockfish_path = stockfish_path
        self.elo = elo
        self.time_limit = time_limit

        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Stockfish not found at: {stockfish_path}\n"
                "Please install Stockfish:\n"
                "  Ubuntu/Debian: sudo apt-get install stockfish\n"
                "  Or download from: https://stockfishchess.org/download/"
            )

        # Configure Elo limiting
        self.engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": elo
        })

    def get_move(self, board: chess.Board, color: str) -> str:
        """Get move from Stockfish.

        Args:
            board: Current board state
            color: Color to play (unused, Stockfish uses board.turn)

        Returns:
            UCI move string

        Raises:
            RuntimeError: If engine fails to generate move
        """
        try:
            result = self.engine.play(
                board,
                chess.engine.Limit(time=self.time_limit)
            )
            return result.move.uci()
        except chess.engine.EngineTerminatedError as e:
            raise RuntimeError(f"Stockfish engine terminated: {e}")
        except Exception as e:
            raise RuntimeError(f"Stockfish error: {e}")

    def close(self):
        """Close engine connection."""
        if hasattr(self, 'engine'):
            try:
                self.engine.quit()
            except:
                pass  # Ignore errors during cleanup

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
