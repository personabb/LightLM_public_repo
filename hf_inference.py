#!/usr/bin/env python3
import torch
import argparse
import time
from transformers import AutoModel, AutoTokenizer
import time

model_name = "asap-bb/mylightlm_sample"
#model_name = "./hf_model"  # ローカルのHuggingFace形式モデルを使用する場合
prompt = "I'm Mike. I live in"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, dtype=dtype).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

print(f"Device: {device}")
print(f"Data type: {dtype}")
print(f"Model loaded on {device} with {dtype}")
print()

# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

with torch.no_grad():
    start_time = time.time()
    # use_cacheをTrueに設定してKVキャッシュを使用
    with torch.autocast(device_type=device, dtype=dtype):
        output_ids = model.generate(
            input_ids,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            use_cache=True,
        )

    end_time = time.time()
    print(f"⏱️ Generation took {end_time - start_time:.2f} seconds")

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"\n" + "="*50)
print(f"📄 Generated Text:")
print(f"="*50)
print(generated_text)
print(f"="*50)