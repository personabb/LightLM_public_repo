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
checkpoint_path = "./model_testing/model.checkpoint.epoch0_step11000_global11000.pt"  # 例
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
model.to(device)
model.eval()

# 3. 推論する
text = "I am Mike. I live in"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

# 生成する最大トークン数を指定
max_new_tokens = 100
# EOS (End of Sentence) トークンが出たら生成を停止するためのID
eos_token_id = tokenizer.eos_token_id

print(f"Prompt: '{text}'")
print("Generated: ")

with torch.no_grad():
    for _ in range(max_new_tokens):
        # 1. 現在のシーケンスで次のトークンを予測
        #    use_cache=Trueにすると高速化できますが、まずは基本的な動作を確認
        logits, _, _ = model(input_ids)

        # 2. logitsの最後のトークン位置から、最も確率の高いものを選択
        #    logits[:, -1, :] はシーケンスの最後の単語に対する予測結果
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)

        # 3. 予測されたトークンIDを入力シーケンスに追加
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)

        # 4. もし予測されたトークンがEOSトークンなら、生成を終了
        if next_token_id.item() == eos_token_id:
            print("\n[EOS token detected. Stopping generation.]")
            break

        # 5. 生成されたトークンを都度デコードして表示（任意）
        generated_token = tokenizer.decode(next_token_id)
        print(generated_token, end="", flush=True)

print("\n\n--- Final Output ---")
# 最終的に生成されたシーケンス全体をデコード
final_output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(final_output)