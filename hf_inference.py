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
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# トークナイズ
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

with torch.no_grad():
    start_time = time.time()
    # use_cacheをTrueに設定してKVキャッシュを使用
    output_ids = model.generate(
        input_ids,
        max_tokens=100,
        temperature=0.001,
        top_p=0.9,
        repetition_penalty=1.0,
        use_cache=False,
    )

    end_time = time.time()
    print(f"⏱️ Generation took {end_time - start_time:.2f} seconds")

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"\n" + "="*50)
print(f"📄 Generated Text:")
print(f"="*50)
print(generated_text)
print(f"="*50)