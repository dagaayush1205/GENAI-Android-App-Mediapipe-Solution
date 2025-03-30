import torch
import time
from transformers import pipeline, AutoTokenizer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM
model_name="microsoft/phi-1.5"
# breakpoint()
start_time = 0
elapsed_time = 0
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Enter input")
text = input()
start_time = time.time()
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(**inputs, max_length=100)
elapsed_time = time.time() - start_time
print("OUTPUT: time elapsed=", elapsed_time)
print(tokenizer.decode(output[0]))
print(output)

