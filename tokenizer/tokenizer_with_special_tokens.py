from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("nsarrazin/chessformer")

# Define special tokens
bog_token = "<BOG>"
eog_token = "<EOG>"
white_win_token = "<WHITE_WIN>"
black_win_token = "<BLACK_WIN>"
draw_token = "<DRAW>"

# Generate Elo tokens for range 0-3500 in steps of 100
elo_tokens = []
for elo in range(0, 3600, 100):  # 0, 100, 200, ..., 3500
    elo_tokens.append(f"<WHITE:{elo}>")
    elo_tokens.append(f"<BLACK:{elo}>")

# Combine all special tokens
all_special_tokens = [
    bog_token, 
    eog_token, 
    white_win_token, 
    black_win_token, 
    draw_token
] + elo_tokens

# Add special tokens as additional tokens
num_added = tokenizer.add_special_tokens({'additional_special_tokens': all_special_tokens})
print(f"Added {num_added} special tokens")

# Explicitly set bos_token and eos_token
tokenizer.bos_token = bog_token
tokenizer.eos_token = eog_token

# Save the modified tokenizer locally
tokenizer.save_pretrained("./uci_tokenizer_with_special_tokens")

# Verify
print(f"BOS token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
print(f"BOG token ID: {tokenizer.convert_tokens_to_ids(bog_token)}")
print(f"EOG token ID: {tokenizer.convert_tokens_to_ids(eog_token)}")
print(f"WHITE_WIN token ID: {tokenizer.convert_tokens_to_ids(white_win_token)}")
print(f"BLACK_WIN token ID: {tokenizer.convert_tokens_to_ids(black_win_token)}")
print(f"DRAW token ID: {tokenizer.convert_tokens_to_ids(draw_token)}")
print(f"Example Elo token <WHITE:1500> ID: {tokenizer.convert_tokens_to_ids('<WHITE:1500>')}")
print(f"Example Elo token <BLACK:2000> ID: {tokenizer.convert_tokens_to_ids('<BLACK:2000>')}")
print(f"Vocab size: {len(tokenizer)}")
print(f"Total Elo tokens added: {len(elo_tokens)} (36 levels x 2 colors)")