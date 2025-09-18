import os
import sys
# LightLMローカルモジュール
from model import Transformer, ModelConfig
from trainer import Trainer, TrainerConfig, DataLoader


# 外部ライブラリ
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
import json

print("全てのインポート完了")

torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()

tokenizer_id = "HuggingFaceTB/SmolLM-360M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.pad_token = tokenizer.eos_token

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 設定パラメータ（Colab環境最適化）
print("🛠️ 学習・モデル設定を構成中...")

# Colab環境向けTrainerConfig
train_config = TrainerConfig(
    vocab_size=tokenizer.vocab_size,
    num_epochs=4,  # Colab環境での短時間テスト

    # Colab環境設定（単一GPU）
    use_ddp=False,  # Colab は単一GPU環境
    use_moe=True,
    use_lossfreebalance=False,
    clean_cuda_cache=True,
    use_compile=True,  # PyTorch 2.0 最適化
    use_dtype="bfloat16" if device == 'cuda' else "float32",

    seed=42,
    max_seq_len=512,  # Colab GPU メモリ制限のため短縮
    batch_size=2,     # Colab GPU メモリに適合
    accumulation_steps=64,  # 実効バッチサイズ32を維持

    weight_decay=0.1,
    warmup_ratio=0.1,
    learning_rate=5e-4,
    betas=(0.90, 0.95),
    update_rate=1e-5,

    val_ratio=0.005,
    steps_for_eval=10000,  # より頻繁な評価
    eval_interval=500,   # より短い間隔

    checkpoints_frequency=250,
    path_to_checkpoints="./model_testing",

    # データセットパスをマイドライブ基準に変更
    #tokenized_dataset_path = "HuggingFaceTB/cosmopedia",
    tokenized_dataset_path = "HuggingFaceFW/fineweb-edu",
    sub_target_files = "data/CC-MAIN-2025-26/*.parquet",
    #sub_target_files = "data/CC-MAIN-2013-20/train-00000-of-00014.parquet",
    eval_log_file="./log/eval.txt",

    continue_train = True,
    checkpoint_path = 'model_testing/model.checkpoint.epoch0_step16000_global16000.pt',
)

# Colab環境向けModelConfig
config = ModelConfig(
    vocab_size=tokenizer.vocab_size,

    num_dims=512,      # 効率的なサイズ
    num_heads=16,
    num_kv_heads=4,    # GQA による効率化
    num_layers=24,     # Colab GPU に適したサイズ
    ffn_hidden_dims=512 * 4,

    rmsnorm_eps=1e-6,
    rope_theta=1e5,

    context_len=1024,  # Colab 環境での制限

    use_cache=False,
    use_flash=True,    # 利用可能な場合
    use_moe=True,     # シンプル構成

    # MoE設定（未使用）
    moe_num_experts=4,
    moe_active_experts=1,
    moe_eps=1e-6,
    moe_aux_loss_coef=0.01,
    moe_shared_experts=1,
    use_lossfreebalance=False,
)


# 必要なディレクトリを作成
os.makedirs("./model_testing", exist_ok=True)
os.makedirs("./log", exist_ok=True)

# モデル初期化
model = Transformer(config)

# パラメータ数を正確に計算
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"実際のパラメータ数: {total_params:,} ({total_params/1e6:.1f}M)")
print(f"学習可能パラメータ: {trainable_params:,}")

# データローダー初期化
data_loader = DataLoader(train_config, tokenizer=tokenizer, hf_split="train", cache = "../LightLM_private/cache", use_cache=True)

# トレーナー初期化
trainer = Trainer(train_config, model, tokenizer)

print("全コンポーネント初期化完了！")
trainer.train(data_loader)
