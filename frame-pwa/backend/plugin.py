from peft import LoraConfig, get_peft_model
from transformers import DetrForObjectDetection

base_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

target_modules = [
    "model.encoder.layers.0.self_attn.k_proj",
    "model.encoder.layers.0.self_attn.v_proj",
    "model.encoder.layers.0.self_attn.q_proj",
    "model.decoder.layers.0.self_attn.k_proj",
    "model.decoder.layers.0.self_attn.v_proj",
    "model.encoder.layers.1.self_attn.k_proj",
    "model.encoder.layers.1.self_attn.v_proj",
    "model.encoder.layers.1.self_attn.q_proj",
    "model.decoder.layers.1.self_attn.k_proj",
    "model.decoder.layers.1.self_attn.v_proj",
    "model.encoder.layers.2.self_attn.k_proj",
    "model.encoder.layers.2.self_attn.v_proj",
    "model.encoder.layers.2.self_attn.q_proj",
    "model.decoder.layers.2.self_attn.k_proj",
    "model.decoder.layers.2.self_attn.v_proj",
    "model.encoder.layers.3.self_attn.k_proj",
    "model.encoder.layers.3.self_attn.v_proj",
    "model.encoder.layers.3.self_attn.q_proj",
    "model.decoder.layers.3.self_attn.k_proj",
    "model.decoder.layers.3.self_attn.v_proj",
    "model.encoder.layers.4.self_attn.k_proj",
    "model.encoder.layers.4.self_attn.v_proj",
    "model.encoder.layers.4.self_attn.q_proj",
    "model.decoder.layers.4.self_attn.k_proj",
    "model.decoder.layers.4.self_attn.v_proj",
    "model.encoder.layers.5.self_attn.k_proj",
    "model.encoder.layers.5.self_attn.v_proj",
    "model.encoder.layers.5.self_attn.q_proj",
    "model.decoder.layers.5.self_attn.k_proj",
    "model.decoder.layers.5.self_attn.v_proj"
]


for name, module in base_model.named_modules():
    print(name)
lora_left_config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules, lora_dropout=0.05)
lora_right_config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules, lora_dropout=0.05)
lora_left = get_peft_model(base_model, lora_left_config)
lora_right = get_peft_model(base_model, lora_right_config)

lora_left.save_pretrained("lora_left_hand")
lora_right.save_pretrained("lora_right_hand")
