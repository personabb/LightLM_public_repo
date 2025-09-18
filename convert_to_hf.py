#!/usr/bin/env python3
import os
import torch
import json
import shutil
from transformers import AutoTokenizer
from model import Transformer, ModelConfig

def convert_checkpoint_to_hf(checkpoint_path, output_dir="./hf_model"):
    """
    PyTorchチェックポイントをHuggingFace形式に変換

    Args:
        checkpoint_path: .ptチェックポイントファイルのパス
        output_dir: 変換後のモデルを保存するディレクトリ
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # チェックポイント読み込み
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']

    print(f"Checkpoint keys: {len(state_dict.keys())}")
    print("Sample keys:", list(state_dict.keys())[:5])

    # _orig_mod.プレフィックスを除去
    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            clean_key = key[len('_orig_mod.'):]
            clean_state_dict[clean_key] = value
        else:
            clean_state_dict[key] = value

    print(f"Cleaned state dict keys: {len(clean_state_dict.keys())}")
    print("Sample cleaned keys:", list(clean_state_dict.keys())[:5])

    # トークナイザー情報（train.pyから）
    tokenizer_id = "HuggingFaceTB/SmolLM-360M"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    # ModelConfig作成（train.pyの設定に基づく）
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        num_dims=512,
        num_heads=16,
        num_kv_heads=4,
        num_layers=24,
        ffn_hidden_dims=512 * 4,
        rmsnorm_eps=1e-6,
        rope_theta=1e5,
        context_len=1024,
        use_cache=False,
        use_flash=True,
        use_moe=True,
        moe_num_experts=4,
        moe_active_experts=1,
        moe_eps=1e-6,
        moe_aux_loss_coef=0.01,
        moe_shared_experts=1,
        use_lossfreebalance=False,
    )

    print(f"Model config: {config}")

    # モデルインスタンス作成
    model = Transformer(config)

    # 重みをロード
    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # HuggingFace形式で保存（PyTorchModelHubMixinを使用）
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir)

    # トークナイザーも保存
    tokenizer.save_pretrained(output_dir)

    # model.pyをコピー（HuggingFaceに必要）
    model_py_path = os.path.join(output_dir, 'model.py')
    if os.path.exists('model.py'):
        shutil.copy2('model.py', model_py_path)
        print(f"Copied model.py to: {model_py_path}")
    else:
        print("Warning: model.py not found in current directory")

    # 追加のconfig情報を保存
    config_dict = {
        'model_type': 'llama',
        'vocab_size': config.vocab_size,
        'num_dims': config.num_dims,
        'num_heads': config.num_heads,
        'num_kv_heads': config.num_kv_heads,
        'num_layers': config.num_layers,
        'ffn_hidden_dims': config.ffn_hidden_dims,
        'context_len': config.context_len,
        'use_cache': config.use_cache,
        'use_flash': config.use_flash,
        'use_moe': config.use_moe,
        'moe_num_experts': config.moe_num_experts,
        'moe_active_experts': config.moe_active_experts,
        'moe_eps': config.moe_eps,
        'moe_aux_loss_coef': config.moe_aux_loss_coef,
        'moe_shared_experts': config.moe_shared_experts,
        'use_lossfreebalance': config.use_lossfreebalance,
        'rmsnorm_eps': config.rmsnorm_eps,
        'rope_theta': config.rope_theta,
        'tokenizer_id': tokenizer_id,
        'architecture': 'LightLM',
        'torch_dtype': 'float32',
        # HuggingFace auto_map設定（標準的な読み込み方法）
        'auto_map': {
            'AutoModel': 'model.Transformer'
        },
        'trust_remote_code': True,
    }

    # config.jsonに追加情報を書き込み
    config_path = os.path.join(output_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing_config = json.load(f)
        existing_config.update(config_dict)
    else:
        existing_config = config_dict

    with open(config_path, 'w') as f:
        json.dump(existing_config, f, indent=2)

    print(f"✅ Conversion completed successfully!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Files created:")
    for file in os.listdir(output_dir):
        print(f"     - {file}")

    return output_dir

def main():
    # デフォルトのチェックポイントパス（train.pyから）
    default_checkpoint = "model_testing/model.checkpoint.epoch0_step11000_global11000.pt"

    if not os.path.exists(default_checkpoint):
        print(f"❌ Checkpoint not found: {default_checkpoint}")
        print("Available checkpoints:")
        checkpoint_dir = "model_testing"
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pt'):
                    print(f"   - {os.path.join(checkpoint_dir, file)}")
        return

    print(f"🚀 Converting checkpoint to HuggingFace format...")
    output_dir = convert_checkpoint_to_hf(default_checkpoint)
    print(f"🎉 Done! Model is ready for upload to HuggingFace Hub.")
    print(f"    Next step: Run 'python upload_to_hub.py' to upload to HuggingFace")

if __name__ == "__main__":
    main()