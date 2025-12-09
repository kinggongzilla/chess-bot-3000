"""Game engine for orchestrating chess games between two players."""

import chess
from utils import GameResult


class GameEngine:
    """Orchestrates a single chess game between two players."""

    def __init__(self, white_player, black_player, max_moves: int = 500):
        """Initialize game engine.

        Args:
            white_player: Player object for white pieces
            black_player: Player object for black pieces
            max_moves: Maximum moves before declaring draw
        """
        self.white_player = white_player
        self.black_player = black_player
        self.max_moves = max_moves

    def play_game(self, game_number: int, bot_color: str) -> GameResult:
        """Play a complete game and return result.

        Args:
            game_number: Game number for tracking
            bot_color: Color the bot is playing ("white" or "black")

        Returns:
            GameResult with outcome details
        """
        board = chess.Board()
        move_count = 0

        while not board.is_game_over() and move_count < self.max_moves:
            # Determine current player
            current_color = "white" if board.turn else "black"

            if current_color == "white":
                current_player = self.white_player
                player_name = "white"
            else:
                current_player = self.black_player
                player_name = "black"

            # Get move from player
            try:
                if hasattr(current_player, 'get_move'):
                    # Bot player - needs board and color
                    if player_name == bot_color:
                        move_str = current_player.get_move(board, current_color, opponent_elo=1500)
                    else:
                        move_str = current_player.get_move(board, current_color)
                else:
                    raise AttributeError(f"Player has no get_move method")
            except Exception as e:
                # Player failed to generate move
                opponent = "gpt5" if player_name == bot_color else "bot"
                return GameResult(
                    game_number=game_number,
                    bot_color=bot_color,
                    outcome=f"{opponent}_win",
                    reason="move_generation_error",
                    move_count=move_count,
                    illegal_move_player="bot" if player_name == bot_color else "gpt5"
                )

            # Validate and apply move
            valid = self._validate_and_apply_move(move_str, board)

            if not valid:
                # Illegal move - opponent wins
                opponent = "gpt5" if player_name == bot_color else "bot"
                return GameResult(
                    game_number=game_number,
                    bot_color=bot_color,
                    outcome=f"{opponent}_win",
                    reason="illegal_move",
                    move_count=move_count,
                    illegal_move_player="bot" if player_name == bot_color else "gpt5"
                )

            move_count += 1

        # Game ended naturally or hit max moves
        return self._determine_outcome(board, game_number, bot_color, move_count)

    def _validate_and_apply_move(self, move_str: str, board: chess.Board) -> bool:
        """Validate and apply move.

        Args:
            move_str: UCI move string
            board: Current board state

        Returns:
            True if move is valid and applied, False otherwise
        """
        try:
            move = chess.Move.from_uci(move_str.strip())

            if move in board.legal_moves:
                board.push(move)
                return True
            else:
                return False

        except (ValueError, chess.InvalidMoveError):
            return False

    def _determine_outcome(self, board: chess.Board, game_number: int,
                           bot_color: str, move_count: int) -> GameResult:
        """Determine game outcome from board state.

        Args:
            board: Final board state
            game_number: Game number
            bot_color: Color the bot is playing
            move_count: Number of moves played

        Returns:
            GameResult with outcome details
        """
        if board.is_checkmate():
            # Winner is opposite of current turn
            winner_color = "black" if board.turn else "white"
            outcome = "bot_win" if winner_color == bot_color else "gpt5_win"
            return GameResult(
                game_number=game_number,
                bot_color=bot_color,
                outcome=outcome,
                reason="checkmate",
                move_count=move_count
            )

        elif board.is_stalemate():
            return GameResult(
                game_number=game_number,
                bot_color=bot_color,
                outcome="draw",
                reason="stalemate",
                move_count=move_count
            )

        elif board.is_insufficient_material():
            return GameResult(
                game_number=game_number,
                bot_color=bot_color,
                outcome="draw",
                reason="insufficient_material",
                move_count=move_count
            )

        elif board.can_claim_fifty_moves():
            return GameResult(
                game_number=game_number,
                bot_color=bot_color,
                outcome="draw",
                reason="fifty_move_rule",
                move_count=move_count
            )

        elif board.can_claim_threefold_repetition():
            return GameResult(
                game_number=game_number,
                bot_color=bot_color,
                outcome="draw",
                reason="threefold_repetition",
                move_count=move_count
            )

        else:
            # Max moves reached
            return GameResult(
                game_number=game_number,
                bot_color=bot_color,
                outcome="draw",
                reason="max_moves",
                move_count=move_count
            )
