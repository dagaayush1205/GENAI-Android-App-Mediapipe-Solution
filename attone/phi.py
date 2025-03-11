import torch
from transformers import pipelinelr
AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name="microsoft/phi-1.5"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Enter input")
text = input()
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(**inputs, max_length=100)
print("OUTPUT")
print(tokenizer.decode(output[0]))
print(output)

