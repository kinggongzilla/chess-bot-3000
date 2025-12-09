from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("hf_model_chess_100m")
tokenizer = AutoTokenizer.from_pretrained("hf_model_chess_100m")

# Generate a move at 1500 ELO level
prompt = "<BOG> <WHITE:1500> <BLACK:1600> <BLACK_WIN> e2e4"
inputs = tokenizer(prompt, return_tensors="pt")
# Remove token_type_ids if present
input_ids = inputs['input_ids']
attention_mask = inputs.get('attention_mask', None)
outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50)
game = tokenizer.decode(outputs[0])
print(game)
