#!/usr/bin/env python3
"""Simple chess UI to play against the chess bot model."""

import chess
import torch
import os
import sys
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
        print(f"Loading model from: {model_path}")
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


class ChessUI:
    """Simple text-based chess UI to play against the bot."""

    def __init__(self, model_path: str = "hf_model_chess_100m", bot_elo: int = 1500):
        """Initialize chess UI.

        Args:
            model_path: Path to HuggingFace model directory
            bot_elo: Elo rating for the bot
        """
        self.model_path = model_path
        self.bot_elo = bot_elo
        self.board = chess.Board()
        self.bot_player = None
        self.game_over = False

    def run(self):
        """Run the chess game."""
        print("🏁 Chess Bot 3000 - Play against the AI!")
        print(f"Bot ELO: {self.bot_elo}")
        print("Type 'quit' to exit, 'help' for commands")
        print()
        
        # Initialize bot player
        try:
            self.bot_player = ChessBotPlayer(self.model_path, self.bot_elo)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        
        # Determine who plays as white
        bot_color = self._choose_bot_color()
        print(f"Bot is playing as: {bot_color}")
        print()
        
        # Main game loop
        while not self.game_over:
            self._display_board()
            
            if self.board.is_game_over():
                self._handle_game_over()
                break
            
            current_color = "white" if self.board.turn else "black"
            print(f"\n🎯 {current_color.capitalize()}'s turn")
            
            if (current_color == "white" and bot_color == "white") or \
               (current_color == "black" and bot_color == "black"):
                # Bot's turn
                self._bot_move()
            else:
                # Human's turn
                self._human_move()
        
        print("🎮 Game over!")

    def _choose_bot_color(self) -> str:
        """Let user choose if bot plays as white or black."""
        while True:
            choice = input("Should bot play as white or black? (w/b): ").lower().strip()
            if choice in ['w', 'white']:
                return "white"
            elif choice in ['b', 'black']:
                return "black"
            elif choice == 'quit':
                sys.exit(0)
            else:
                print("Please enter 'w' for white or 'b' for black")

    def _display_board(self):
        """Display the current chess board with chess piece figures."""
        print(self._get_board_with_figures())
        print()

    def _get_board_with_figures(self) -> str:
        """Generate board display with Unicode chess figures."""
        # Unicode chess pieces
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        board_str = ""
        board_str += "  a b c d e f g h\n"
        board_str += " +-----------------+\n"
        
        for row in range(8):
            board_str += f"{8-row}|"
            for col in range(8):
                piece = self.board.piece_at(chess.square(col, 7-row))
                if piece:
                    symbol = piece_symbols.get(piece.symbol(), '?')
                    board_str += f"{symbol} "
                else:
                    board_str += ". "
            board_str += f"|{8-row}\n"
        
        board_str += " +-----------------+\n"
        board_str += "  a b c d e f g h\n"
        
        # Add turn indicator
        current_player = "White" if self.board.turn == chess.WHITE else "Black"
        board_str += f"\n🎯 {current_player}'s turn to move\n"
        
        # Add move count
        move_count = len(self.board.move_stack)
        board_str += f"📊 Moves played: {move_count}\n"
        
        return board_str

    def _human_move(self):
        """Handle human player's move."""
        while True:
            move_input = input("Your move (e.g., e2e4, or 'help' for commands): ").strip()
            
            if move_input.lower() == 'quit':
                print("👋 Goodbye!")
                sys.exit(0)
            elif move_input.lower() == 'help':
                self._show_help()
                continue
            elif move_input.lower() == 'board':
                self._display_board()
                continue
            elif move_input.lower() == 'undo':
                if len(self.board.move_stack) > 0:
                    self.board.pop()
                    print("↩️  Move undone")
                    self._display_board()
                else:
                    print("⚠️  No moves to undo")
                continue
            
            try:
                # Try to parse as UCI move
                move = chess.Move.from_uci(move_input)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    print(f"✅ You played: {move_input}")
                    break
                else:
                    print("❌ Illegal move. Try again.")
                    print("Legal moves:", [move.uci() for move in self.board.legal_moves])
            except ValueError:
                print("❌ Invalid move format. Use UCI notation (e.g., e2e4, e7e5)")

    def _bot_move(self):
        """Handle bot's move."""
        print("🤖 Bot is thinking...")
        
        try:
            current_color = "white" if self.board.turn else "black"
            move = self.bot_player.get_move(self.board, current_color)
            
            # Validate move
            try:
                move_obj = chess.Move.from_uci(move)
                if move_obj in self.board.legal_moves:
                    self.board.push(move_obj)
                    print(f"🤖 Bot played: {move}")
                else:
                    print(f"❌ Bot generated illegal move: {move}")
                    print("Bot resigns!")
                    self.game_over = True
            except ValueError:
                print(f"❌ Bot generated invalid move: {move}")
                print("Bot resigns!")
                self.game_over = True
                
        except Exception as e:
            print(f"❌ Error generating bot move: {e}")
            print("Bot resigns!")
            self.game_over = True

    def _handle_game_over(self):
        """Handle game over state."""
        print("\n🏁 GAME OVER!")
        
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            print(f"🏆 Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            print("🤝 Stalemate! Game drawn.")
        elif self.board.is_insufficient_material():
            print("🤝 Insufficient material! Game drawn.")
        elif self.board.is_seventyfive_moves():
            print("🤝 75-move rule! Game drawn.")
        elif self.board.is_fivefold_repetition():
            print("🤝 Fivefold repetition! Game drawn.")
        else:
            print("🤝 Game drawn.")
        
        self.game_over = True

    def _show_help(self):
        """Show help information."""
        print("\n📖 HELP:")
        print("Commands:")
        print("  quit       - Exit the game")
        print("  help       - Show this help")
        print("  board      - Show the current board")
        print("  undo       - Undo last move")
        print()
        print("🏁 Chess Piece Figures:")
        print("  White: ♔ King, ♕ Queen, ♖ Rook, ♗ Bishop, ♘ Knight, ♙ Pawn")
        print("  Black: ♚ King, ♛ Queen, ♜ Rook, ♝ Bishop, ♞ Knight, ♟ Pawn")
        print()
        print("Move format: UCI notation (e.g., e2e4, e7e5, g1f3)")
        print("  - e2e4: Move pawn ♙ from e2 to e4")
        print("  - g1f3: Move knight ♘ from g1 to f3")
        print("  - e1g1: Castle kingside (white)")
        print("  - e7e8q: Promote pawn to queen ♛")
        print("  - a2a4: Move pawn ♙ from a2 to a4")
        print()
        print("💡 Tips:")
        print("  - Type 'board' anytime to see the current position")
        print("  - Use 'undo' to correct mistakes")
        print("  - The bot plays at 1500 ELO by default")
        print()


def main():
    """Main function."""
    # Parse command line arguments
    model_path = "hf_model_chess_100m"
    bot_elo = 1500
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"❌ Model directory not found: {model_path}")
        print("Please ensure the model is downloaded and available.")
        return
    
    # Create and run chess UI
    ui = ChessUI(model_path=model_path, bot_elo=bot_elo)
    ui.run()


if __name__ == "__main__":
    main()