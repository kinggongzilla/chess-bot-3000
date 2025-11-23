from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("nsarrazin/chessformer")

bog_token = "<BOG>"
eog_token = "<EOG>"

# Add special tokens as additional tokens
num_added = tokenizer.add_special_tokens({'additional_special_tokens': [bog_token, eog_token]})
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
print(f"Vocab size: {len(tokenizer)}")
