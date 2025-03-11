import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
print("Enter input")
text = input()
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(**inputs, max_length=100)
print("OUTPUT")
print(tokenizer.decode(output[0]))

