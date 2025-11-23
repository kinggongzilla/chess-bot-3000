import os
import pandas as pd
import chess
import chess.pgn
from io import StringIO
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def pgn_to_uci(movetext):
    """
    Convert PGN movetext to UCI notation with BOG/EOG tokens.
    
    Args:
        movetext: PGN format moves (e.g., "1. e4 e5 2. Nf3 Nc6")
    
    Returns:
        UCI format moves with tokens (e.g., "<BOG> e2e4 e7e5 g1f3 b8c6 <EOG>")
    """
    try:
        # Create a game from the movetext
        pgn = StringIO(movetext)
        game = chess.pgn.read_game(pgn)
        
        if game is None:
            return "<BOG> <EOG>"
        
        # Extract UCI moves
        board = game.board()
        uci_moves = []
        
        for move in game.mainline_moves():
            uci_moves.append(move.uci())
            board.push(move)
        
        # Add BOG and EOG tokens
        if uci_moves:
            return "<BOG> " + " ".join(uci_moves) + " <EOG>"
        else:
            return "<BOG> <EOG>"
    
    except Exception as e:
        # Return empty tokens if parsing fails
        return "<BOG> <EOG>"


def process_parquet_file(args):
    """
    Process a single parquet file: add movetext_preprocessed column.
    
    Args:
        args: Tuple of (input_path, output_path)
    
    Returns:
        Tuple of (success: bool, file_path: str, error_message: str or None)
    """
    input_path, output_path = args
    
    try:
        # Read the parquet file
        df = pd.read_parquet(input_path)
        
        # Convert movetext to UCI notation
        df['movetext_preprocessed'] = df['movetext'].apply(pgn_to_uci)
        
        # Save to output path
        df.to_parquet(output_path, index=False)
        
        return (True, str(input_path), None)
    
    except Exception as e:
        return (False, str(input_path), str(e))


def process_dataset(data_dir="data", output_dir="data_processed", max_workers=None):
    """
    Process all parquet files in the dataset structure using multiprocessing.
    
    Args:
        data_dir: Root directory containing year folders
        output_dir: Output directory for processed files
        max_workers: Number of worker processes (default: CPU count)
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Find all parquet files
    parquet_files = list(data_path.rglob("*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files to process")
    
    # Prepare file pairs (input, output)
    file_pairs = []
    for input_file in parquet_files:
        # Create corresponding output path
        relative_path = input_file.relative_to(data_path)
        output_file = output_path / relative_path
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already processed
        if output_file.exists():
            print(f"Skipping {relative_path} (already processed)")
            continue
        
        file_pairs.append((input_file, output_file))
    
    if not file_pairs:
        print("No files to process (all already completed)")
        return
    
    num_workers = max_workers or multiprocessing.cpu_count()
    print(f"Processing {len(file_pairs)} files using {num_workers} workers...")
    
    # Process files in parallel
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(file_pairs), desc="Processing files") as pbar:
            for success, file_path, error in executor.map(process_parquet_file, file_pairs):
                if success:
                    completed += 1
                else:
                    failed += 1
                    print(f"\nError processing {file_path}: {error}")
                
                pbar.update(1)
                pbar.set_postfix({"completed": completed, "failed": failed})
    
    print(f"\nProcessing complete!")
    print(f"  Successfully processed: {completed} files")
    print(f"  Failed: {failed} files")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    # You can customize these paths
    DATA_DIR = "lichess_data/data"
    OUTPUT_DIR = "lichess_data/data_processed"
    
    # Use all 24 cores - each will process one file at a time
    MAX_WORKERS = 24
    
    print("Starting PGN to UCI conversion...")
    print(f"Input directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Workers: {MAX_WORKERS}")
    print()
    
    process_dataset(DATA_DIR, OUTPUT_DIR, max_workers=MAX_WORKERS)
