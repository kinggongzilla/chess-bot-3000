import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


# Mapping from PGN result notation to special tokens
RESULT_TO_TOKEN = {
    "1-0": "<WHITE_WIN>",
    "0-1": "<BLACK_WIN>",
    "1/2-1/2": "<DRAW>",
}

# Default Elo rating when not available or invalid
DEFAULT_ELO = 1500

# Valid Elo range
MIN_ELO = 0
MAX_ELO = 3500


def elo_to_token(elo, color):
    """
    Convert an Elo rating to a token, rounding to the nearest 100.
    
    Args:
        elo: Elo rating (int, float, or string)
        color: "WHITE" or "BLACK"
    
    Returns:
        Token string like "<WHITE:1500>" or "<BLACK:2000>"
    """
    try:
        elo_value = int(float(elo))
        # Round to nearest 100
        rounded_elo = round(elo_value / 100) * 100
        # Clamp to valid range
        rounded_elo = max(MIN_ELO, min(MAX_ELO, rounded_elo))
    except (ValueError, TypeError):
        # Use default if conversion fails
        rounded_elo = DEFAULT_ELO
    
    return f"<{color}:{rounded_elo}>"


def add_tokens_to_uci(movetext_uci, result, white_elo, black_elo):
    """
    Add Elo and result tokens to an existing UCI movetext string.
    
    Transforms: "<BOG> e2e4 e7e5 ... <EOG>"
    Into:       "<BOG> <WHITE:1500> <BLACK:1600> <WHITE_WIN> e2e4 e7e5 ... <EOG>"
    
    Args:
        movetext_uci: Existing UCI string with <BOG> and <EOG>
        result: Game result (e.g., "1-0", "0-1", "1/2-1/2")
        white_elo: White player's Elo rating
        black_elo: Black player's Elo rating
    
    Returns:
        Updated UCI string with Elo and result tokens
    """
    # Get result token
    result_token = RESULT_TO_TOKEN.get(result, "")
    
    # Get Elo tokens
    white_elo_token = elo_to_token(white_elo, "WHITE")
    black_elo_token = elo_to_token(black_elo, "BLACK")
    
    # Build the tokens to insert after <BOG>
    if result_token:
        insert_tokens = f"{white_elo_token} {black_elo_token} {result_token}"
    else:
        insert_tokens = f"{white_elo_token} {black_elo_token}"
    
    # Replace "<BOG> " with "<BOG> <tokens> "
    if movetext_uci.startswith("<BOG> "):
        return "<BOG> " + insert_tokens + " " + movetext_uci[6:]
    elif movetext_uci.startswith("<BOG>"):
        # Handle case where there's no space after <BOG>
        return "<BOG> " + insert_tokens + movetext_uci[5:]
    else:
        # Fallback: prepend tokens
        return "<BOG> " + insert_tokens + " " + movetext_uci


def process_parquet_file(args):
    """
    Process a single parquet file: add Elo and result tokens to existing UCI data.
    Overwrites the processed file in place using atomic rename.
    
    Args:
        args: Tuple of (original_path, processed_path)
    
    Returns:
        Tuple of (success: bool, file_path: str, error_message: str or None)
    """
    original_path, processed_path = args
    
    try:
        # Read the original data (for Result and Elo columns)
        df_original = pd.read_parquet(original_path)
        
        # Read the processed data (for movetext_uci)
        df_processed = pd.read_parquet(processed_path)
        
        # Verify row counts match
        if len(df_original) != len(df_processed):
            return (False, str(processed_path), 
                    f"Row count mismatch: original={len(df_original)}, processed={len(df_processed)}")
        
        # Add tokens to each row
        df_processed['movetext_uci'] = [
            add_tokens_to_uci(
                uci,
                df_original.iloc[i].get('Result', ''),
                df_original.iloc[i].get('WhiteElo', DEFAULT_ELO),
                df_original.iloc[i].get('BlackElo', DEFAULT_ELO)
            )
            for i, uci in enumerate(df_processed['movetext_uci'])
        ]
        
        # Write to temp file first, then atomic rename to overwrite
        temp_path = processed_path.with_suffix('.parquet.tmp')
        df_processed.to_parquet(temp_path, index=False)
        temp_path.rename(processed_path)
        
        return (True, str(processed_path), None)
    
    except Exception as e:
        # Clean up temp file if it exists
        temp_path = processed_path.with_suffix('.parquet.tmp')
        if temp_path.exists():
            temp_path.unlink()
        return (False, str(processed_path), str(e))


def is_already_processed(processed_path):
    """
    Check if a file has already been processed by looking for Elo tokens.
    
    Args:
        processed_path: Path to the processed parquet file
    
    Returns:
        True if file already contains Elo tokens, False otherwise
    """
    try:
        df = pd.read_parquet(processed_path)
        if len(df) == 0:
            return False
        # Check first row for Elo token pattern
        first_row = df.iloc[0]['movetext_uci']
        return '<WHITE:' in first_row and '<BLACK:' in first_row
    except Exception:
        return False


def process_dataset(original_dir="data", processed_dir="data_processed", max_workers=None):
    """
    Add Elo and result tokens to all processed parquet files (in-place overwrite).
    
    Args:
        original_dir: Directory containing original data (with Result, WhiteElo, BlackElo)
        processed_dir: Directory containing processed data (with movetext_uci)
        max_workers: Number of worker processes (default: CPU count)
    """
    original_path = Path(original_dir)
    processed_path = Path(processed_dir)
    
    # Find all processed parquet files
    processed_files = list(processed_path.rglob("*.parquet"))
    
    if not processed_files:
        print(f"No parquet files found in {processed_dir}")
        return
    
    print(f"Found {len(processed_files)} parquet files to check")
    
    # Prepare file pairs (original, processed)
    file_pairs = []
    skipped = 0
    missing_original = 0
    
    for proc_file in processed_files:
        # Get relative path
        relative_path = proc_file.relative_to(processed_path)
        
        # Find corresponding original file
        orig_file = original_path / relative_path
        
        # Check if already processed (contains Elo tokens)
        if is_already_processed(proc_file):
            skipped += 1
            continue
        
        # Check if original exists
        if not orig_file.exists():
            print(f"Warning: Original file not found for {relative_path}")
            missing_original += 1
            continue
        
        file_pairs.append((orig_file, proc_file))
    
    if skipped > 0:
        print(f"Skipping {skipped} files (already contain Elo tokens)")
    if missing_original > 0:
        print(f"Skipping {missing_original} files (original not found)")
    
    if not file_pairs:
        print("No files to process")
        return
    
    num_workers = max_workers or multiprocessing.cpu_count()
    print(f"Processing {len(file_pairs)} files using {num_workers} workers...")
    
    # Process files in parallel
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(file_pairs), desc="Adding tokens") as pbar:
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


if __name__ == "__main__":
    # Customize these paths as needed
    ORIGINAL_DIR = "lichess_data/data"           # Original data with Result, WhiteElo, BlackElo
    PROCESSED_DIR = "lichess_data/data_processed" # Processed data with movetext_uci (will be overwritten)
    
    # Use all available cores
    MAX_WORKERS = 24
    
    print("Adding Elo and result tokens to processed data (in-place)...")
    print(f"Original data directory: {ORIGINAL_DIR}")
    print(f"Processed data directory: {PROCESSED_DIR}")
    print(f"Workers: {MAX_WORKERS}")
    print()
    
    process_dataset(ORIGINAL_DIR, PROCESSED_DIR, max_workers=MAX_WORKERS)