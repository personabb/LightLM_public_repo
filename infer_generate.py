import torch
from transformers import AutoTokenizer
from model import Transformer, ModelConfig

# 1. 同じ設定でモデルを初期化
tokenizer_id = "HuggingFaceTB/SmolLM-360M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token

config = ModelConfig(
    vocab_size = tokenizer.vocab_size,
    num_dims = 512,
    num_heads = 16,
    num_kv_heads = 4,
    num_layers = 32,
    ffn_hidden_dims = 512 * 4,
    rmsnorm_eps = 1e-6,
    rope_theta = 1e5,
    context_len = 1024,
    use_cache = False,
    use_flash = True,
    use_moe = True,
    moe_num_experts = 4,
    moe_active_experts = 1,
    moe_eps = 1e-6,
    moe_aux_loss_coef = 0.01,
    moe_shared_experts = 1,
    use_lossfreebalance = False,
)

model = Transformer(config)

# 2. 保存済みチェックポイントをロード
checkpoint_path = "./model_testing/model.checkpoint.epoch0_step23500_global23500.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# モデル重みを読み込み
state_dict = checkpoint['model']
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("_orig_mod."):  # DDPやcompileのprefix対応
        new_state_dict[k[len("_orig_mod."):]] = v
    else:
        new_state_dict[k] = v
model.load_state_dict(new_state_dict, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
model.to(device, dtype=dtype)
model.eval()

print(f"Device: {device}")
print(f"Data type: {dtype}")
print(f"Model loaded on {device} with {dtype}")
print()

# 3. テキスト生成の設定
text = "I am Mike. I live in"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

# 生成パラメータ
max_tokens = 100
temperature = 0.7
top_k = 50
top_p = 0.9
repetition_penalty = 1.1
use_cache = True

print(f"Prompt: '{text}'")
print(f"Generation parameters:")
print(f"  max_tokens: {max_tokens}")
print(f"  temperature: {temperature}")
print(f"  top_k: {top_k}")
print(f"  top_p: {top_p}")
print(f"  repetition_penalty: {repetition_penalty}")
print(f"  use_cache: {use_cache}")
print()

# 4. model.generate()を使用してテキスト生成
with torch.no_grad():
    with torch.autocast(device_type=device.type, dtype=dtype):
        generated_ids = model.generate(
            x=input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache
        )

# 5. 結果の表示
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Generated text:")
print(generated_text)