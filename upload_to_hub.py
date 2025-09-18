#!/usr/bin/env python3
import os
import json
from transformers import AutoTokenizer
from huggingface_hub import HfApi

def create_model_card(model_dir, model_name):
    """
    モデルカード（README.md）を作成
    """
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    readme_content = f"""
"""

    readme_path = os.path.join(model_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"✅ Model card created: {readme_path}")
    return readme_path

def upload_to_huggingface(model_dir="./hf_model", repo_name=None, private=False, token=None):
    """
    HuggingFace Hubにモデルをアップロード

    Args:
        model_dir: HuggingFace形式のモデルディレクトリ
        repo_name: HuggingFace Hubでのリポジトリ名
        private: プライベートリポジトリにするかどうか
        token: HuggingFace アクセストークン
    """
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    # モデル名の決定
    if repo_name is None:
        repo_name = "sample"

    print(f"🚀 Uploading model to HuggingFace Hub as: {repo_name}")

    # モデルカード作成
    create_model_card(model_dir, repo_name)

    # HfApiを使って個別ファイルを直接アップロード
    api = HfApi()

    # リポジトリを作成（存在しない場合）
    try:
        print(f"🔧 Creating repository if it doesn't exist...")
        api.create_repo(
            repo_id=repo_name,
            private=private,
            token=token
        )
        print(f"✅ Repository created or already exists: {repo_name}")
    except Exception as e:
        if "409" in str(e) or "already exists" in str(e).lower():
            print(f"✅ Repository already exists: {repo_name}")
        else:
            print(f"❌ Failed to create repository: {e}")
            print("   Make sure you have:")
            print("   1. Valid HuggingFace token")
            print("   2. Proper permissions")
            print("   3. Unique repository name")
            return None

    print(f"📤 Uploading files directly to hub...")

    # 1. model.py をアップロード
    model_py_path = os.path.join(model_dir, 'model.py')
    if os.path.exists(model_py_path):
        print("📤 Uploading model.py...")
        api.upload_file(
            path_or_fileobj=model_py_path,
            path_in_repo="model.py",
            repo_id=repo_name,
            token=token,
            commit_message="Upload model.py for custom architecture"
        )
        print("✅ model.py uploaded successfully")
    else:
        print("⚠️  Warning: model.py not found in model directory")

    # 2. model.safetensors を直接アップロード（メタデータ保持）
    safetensors_path = os.path.join(model_dir, 'model.safetensors')
    if os.path.exists(safetensors_path):
        print("📤 Uploading model.safetensors...")
        api.upload_file(
            path_or_fileobj=safetensors_path,
            path_in_repo="model.safetensors",
            repo_id=repo_name,
            token=token,
            commit_message="Upload safetensors with preserved metadata"
        )
        print("✅ model.safetensors uploaded successfully")
    else:
        print("❌ Error: model.safetensors not found in model directory")
        return None

    # 3. config.json をアップロード
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        print("📤 Uploading config.json...")
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=repo_name,
            token=token,
            commit_message="Upload model configuration"
        )
        print("✅ config.json uploaded successfully")

    # 4. トークナイザーもpush
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("📤 Uploading tokenizer...")
    tokenizer.push_to_hub(repo_name, private=private, token=token)

    print(f"✅ Upload completed successfully!")
    print(f"   Model URL: https://huggingface.co/{repo_name}")
    print(f"   You can now use the model with:")
    print(f"     python hf_inference.py --model {repo_name}")

    return f"https://huggingface.co/{repo_name}"

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Upload LightLM model to HuggingFace Hub')
    parser.add_argument('--model_dir', default='./hf_model',
                      help='Directory containing HuggingFace format model')
    parser.add_argument('--repo_name', default='lightlm-150m-moe',
                      help='Repository name on HuggingFace Hub')
    parser.add_argument('--private', action='store_true',
                      help='Create private repository')
    parser.add_argument('--token',
                      help='HuggingFace access token (or set HF_TOKEN env var)')

    args = parser.parse_args()

    # 環境変数からトークンを取得
    if args.token is None:
        args.token = os.getenv('HF_TOKEN')

    if args.token is None:
        print("⚠️  Warning: No HuggingFace token provided.")
        print("   Set HF_TOKEN environment variable or use --token argument")
        print("   You can get a token from: https://huggingface.co/settings/tokens")

    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        print("   Run 'python convert_to_hf.py' first to convert your checkpoint")
        return

    try:
        url = upload_to_huggingface(
            model_dir=args.model_dir,
            repo_name=args.repo_name,
            private=args.private,
            token=args.token
        )
        print(f"🎉 Success! Your model is now available at: {url}")

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("   Make sure you have:")
        print("   1. Valid HuggingFace token")
        print("   2. Proper permissions")
        print("   3. Unique repository name")

if __name__ == "__main__":
    main()