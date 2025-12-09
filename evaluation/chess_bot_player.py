"""Chess bot player using the trained model."""

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ChessBotPlayer:
    """Player that uses the chess bot model to generate moves."""

    def __init__(self, model_path: str, bot_elo: int):
        """Initialize chess bot player.

        Args:
            model_path: Path to HuggingFace model directory
            bot_elo: Elo rating for the bot
        """
        self.bot_elo = bot_elo
        
        # Check if CUDA is available and set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and move to the appropriate device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def get_move(self, board: chess.Board, color: str, opponent_elo: int = 1500) -> str:
        """Generate next move for given position.

        Args:
            board: Current chess board state
            color: Color to play ("white" or "black")
            opponent_elo: Opponent's Elo rating

        Returns:
            UCI move string (e.g., "e2e4")
        """
        prompt = self._build_prompt(board, color, opponent_elo)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move input tensors to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=1,  # Generate exactly one token (one UCI move)
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Move outputs back to CPU for decoding
        outputs = outputs.cpu()
        generated = self.tokenizer.decode(outputs[0])
        move = self._extract_move(generated, board)
        return move

    def _build_prompt(self, board: chess.Board, color: str, opponent_elo: int) -> str:
        """Build prompt with special tokens.

        Args:
            board: Current chess board state
            color: Color to play ("white" or "black")
            opponent_elo: Opponent's Elo rating

        Returns:
            Formatted prompt string
        """
        # Get move history in UCI format
        moves = " ".join(move.uci() for move in board.move_stack)

        # Round Elo to nearest 100 (clamped 0-3500)
        bot_elo_rounded = max(0, min(3500, round(self.bot_elo / 100) * 100))
        opp_elo_rounded = max(0, min(3500, round(opponent_elo / 100) * 100))

        # Build based on color
        if color == "white":
            white_elo = f"<WHITE:{bot_elo_rounded}>"
            black_elo = f"<BLACK:{opp_elo_rounded}>"
            outcome = "<WHITE_WIN>"
        else:
            white_elo = f"<WHITE:{opp_elo_rounded}>"
            black_elo = f"<BLACK:{bot_elo_rounded}>"
            outcome = "<BLACK_WIN>"

        # Construct prompt
        if moves:
            return f"<BOG> {white_elo} {black_elo} {outcome} {moves}"
        else:
            return f"<BOG> {white_elo} {black_elo} {outcome}"

    def _extract_move(self, generated_text: str, board: chess.Board) -> str:
        """Extract UCI move from generated text.

        Since the tokenizer uses UCI moves as individual tokens,
        the generated move is simply the last token in the output.

        Args:
            generated_text: Full generated text from model
            board: Current board state (unused but kept for interface)

        Returns:
            UCI move string

        Raises:
            ValueError: If no valid UCI move found
        """
        # Split by spaces and get last token (the newly generated move)
        tokens = generated_text.split()

        # The last token should be the UCI move
        # Filter out any special tokens that might appear
        for token in reversed(tokens):
            # Check if it looks like a UCI move (not a special token)
            if not token.startswith('<') and not token.startswith('['):
                return token.strip().lower()

        raise ValueError(f"No UCI move found in output: {generated_text}")