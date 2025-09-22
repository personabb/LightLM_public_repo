import time
import math
import os
import glob
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from datatrove.utils.dataset import DatatroveFolderDataset

# datasets を使って Hugging Face Hub から読み込む
from datasets import load_dataset
from tqdm import tqdm

@dataclass
class TrainerConfig:
    vocab_size: int                 
    num_epochs: int                 

    use_ddp: bool                   
    use_moe: bool                   # enable mixture-of-experts
    use_lossfreebalance: bool       # use Auxiliary-loss-free load balancing strategy for mixture-of-experts from DeepSeek https://arxiv.org/pdf/2408.15664
    clean_cuda_cache: bool = True   # Helps prevent OOM errors during eval on large models
    use_compile: bool = True        # use torch.compile()
    use_dtype: str = "bfloat16"

    seed: int = 1998                
    max_seq_len: int = 1024         # maximum context length for batch
    batch_size: int = 1             # numbe of batches
    accumulation_steps: int = 1
    
    # Optimizer parameters
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    learning_rate: float = 1e-3
    betas: Tuple[float, float] = (0.90, 0.95)
    update_rate: float = 1e-5  # update_rate of biases for loss-free balancing

    val_ratio: int = 0.005
    steps_for_eval: int = 20                            # number of steps for evaluation
    eval_interval: int = 50

    checkpoints_frequency: int = 500
    path_to_checkpoints: str = "./model_testing"        # path to directory to save checkpoints
    max_checkpoints_to_keep: int = 0  # 0の場合は全て保持、-1の場合は最新1つのチェックポイントを保持

    tokenized_dataset_path: str = ""                    # path to directory with tokeized dataset
    sub_target_files: str | list = ""                     # 単一ファイル、ワイルドカード、またはファイルリスト  (if "", load all files)
    eval_log_file: str = "logs/eval.txt"                # path to file to write eval results

    continue_train: bool = False                     # continue training from checkpoint
    checkpoint_path: str = ""                      # path to checkpoint to continue training from



class DataLoader():
    def __init__(self, config, tokenizer=None, rank=0, world_size=1, hf_split="train", streaming=False, small_data_size=None, cache = None, use_cache = False):
        print("Initializing DataLoader...")
        """
        config: TrainerConfig
        tokenizer: transformers tokenizer (必須 if loading from HF)
        rank, world_size: DDP 用（従来通り）
        hf_split: HuggingFace datasets の split（通常 "train"）
        streaming: HuggingFace datasets の streaming モードで読み込むか（巨大データセット用）
        small_data_size: 動作検証用にデータセットを小さくするか（HF のみ）（int型で取り出す件数を指定）
        """
        self.config = config
        self.current_epoch = 0
        self.seed = config.seed
        self.token_size = 2 if config.vocab_size < 65535 else 4
        self.rank = rank
        self.tokenizer = tokenizer
        self.hf_split = hf_split
        self.streaming = streaming
        self.small_data_size = small_data_size
        self.cache = cache
        self.use_cache = use_cache

        # データセット読み込み（内部で local .ds / HF のどちらかを選択）
        self.load_dataset(self.seed)
        self.len_dataset = len(self.dataset)

        if rank == 0:
            print(f"{'Total tokens loaded: '} {self.len_dataset * config.max_seq_len:,}")

        self.train_len_dataset = math.ceil((1-config.val_ratio) * self.len_dataset)
        self.val_len_dataset = self.len_dataset - self.train_len_dataset

        shard_size = self.train_len_dataset // world_size 
        self.train_start_idx = rank * shard_size
        self.train_end_idx = self.train_start_idx + shard_size
        self.train_current_idx = self.train_start_idx

        self.val_start_idx = self.train_len_dataset
        self.val_current_idx = self.val_start_idx

        print(f"DataLoader initialized. Dataset size: {self.len_dataset}, Train size: {self.train_len_dataset}, Val size: {self.val_len_dataset}")
        print(f"Train indices: {self.train_start_idx} to {self.train_end_idx}, Val indices: {self.val_start_idx} to {self.len_dataset}")

    def get_batch(self, current_idx: int, start_idx: int, end_idx: int):
        new_idx = current_idx + self.config.batch_size

        x_l, y_l = zip(*[(self.dataset[idx]['input_ids'][:-1], self.dataset[idx]['input_ids'][1:])
                    for idx in range(current_idx, min(new_idx, self.len_dataset))])
        x, y = torch.stack(list(x_l)), torch.stack(list(y_l))

        if new_idx >= end_idx:
            print("Epoch finished.")
            print(f"current_idx: {current_idx} ,end_idx: {end_idx}, new_idx: {new_idx}, len_dataset: {self.len_dataset}, start_idx: {start_idx}")
            new_idx = start_idx
            self.new_epoch()

        return x, y, new_idx

    def next_batch(self, split):
        if split == "train":
            x, y, self.train_current_idx = self.get_batch(self.train_current_idx, self.train_start_idx, self.train_end_idx)
        else: # validation
            x, y, self.val_current_idx = self.get_batch(self.val_current_idx, self.val_start_idx, self.len_dataset)
        return x, y

    def reset(self, rank: int = 0, world_size: int = 1):
        print("Resetting DataLoader...")
        self.current_epoch = 0
        self.seed = self.config.seed
        self.load_dataset(self.seed)
        self.len_dataset = len(self.dataset)

        self.val_len_dataset = self.len_dataset - self.train_len_dataset

        shard_size = self.len_dataset // world_size 
        self.train_start_idx = rank * shard_size
        self.train_end_idx = self.train_start_idx + shard_size
        self.train_current_idx = self.train_start_idx

        self.val_start_idx = self.train_len_dataset
        self.val_current_idx = self.val_start_idx

    def new_epoch(self):
        print("Starting new epoch...")
        self.current_epoch += 1
        print(f"Current epoch: {self.current_epoch}")
        self.load_dataset(self.seed + self.current_epoch)

    def load_dataset(self, seed: int):
        """
        1) tokenized_dataset_path がローカルのフォルダで .ds ファイルがある -> DatatroveFolderDataset を使用（従来挙動）
        2) そうでなければ tokenized_dataset_path を Hugging Face の dataset id とみなし、load_dataset -> tokenizer.map -> set_format('torch') する
        """
        path = self.config.tokenized_dataset_path

        # Case A: local directory with .ds and datatrove available
        if os.path.isdir(path):
            self.dataset = DatatroveFolderDataset(
                folder_path=self.config.tokenized_dataset_path,
                filename_pattern=os.path.join(self.config.tokenized_dataset_path, "**", "*.ds"),
                seq_len=self.config.max_seq_len,
                token_size=self.token_size,
                recursive=True,
                shuffle=True,
                seed=seed + self.rank
            )
            return

        # Case B: treat path as Hugging Face dataset id
        # tokenizer が必須
        if self.tokenizer is None:
            raise RuntimeError("tokenizer must be provided to DataLoader when loading from Hugging Face dataset")

        # ストリーミングモードではキャッシュを使用しない
        if self.streaming:
            print(f"Loading Hugging Face dataset '{path}' in streaming mode (cache disabled) ...")
            
            # データセットをロード
            if self.config.sub_target_files == "":
                ds = load_dataset(path, split=self.hf_split, streaming=True)
            else:
                ds = load_dataset(
                    path,
                    data_files={'train': self.config.sub_target_files},
                    split=self.hf_split,
                    streaming=True
                )
            
            # テキストカラム名を推定
            text_col = self._find_text_column(ds)
            
            # トークナイズ
            ds = ds.map(self._get_tokenize_fn(text_col), batched=True, remove_columns=ds.column_names)
            self.dataset = ds
            print("Dataset loaded in streaming mode.")
            return

        # 非ストリーミングモード：キャッシュ処理
        if self.use_cache and self.cache is not None:
            cache_path = self.cache
            if os.path.exists(cache_path):
                # キャッシュが存在する場合は読み込み（load_datasetをスキップ）
                try:
                    from datasets import load_from_disk
                    print(f"Loading tokenized dataset from cache: {cache_path}")
                    ds = load_from_disk(cache_path)
                    ds.set_format(type='torch', columns=['input_ids'])
                    ds = ds.shuffle(seed + self.rank)
                    self.dataset = ds
                    print("Dataset loaded from cache.")
                    return
                except Exception as e:
                    print(f"Failed to load cache: {e}. Reprocessing dataset...")
                    # キャッシュの読み込みに失敗した場合は再処理

        # データセットをロードしてトークナイズ
        print(f"Loading Hugging Face dataset '{path}' split='{self.hf_split}' and tokenizing (seq_len={self.config.max_seq_len}) ...")
        
        if self.config.sub_target_files == "":
            ds = load_dataset(path, split=self.hf_split, streaming=False)
        else:
            ds = load_dataset(
                path,
                data_files={'train': self.config.sub_target_files},
                split=self.hf_split
            )

        if self.small_data_size is not None:
            ds = ds.take(self.small_data_size)

        # テキストカラム名を推定
        text_col = self._find_text_column(ds)
        print(f"Using text column: '{text_col}'")

        # トークナイズ
        ds = ds.map(
            self._get_tokenize_fn(text_col), 
            batched=True, 
            remove_columns=ds.column_names, 
            desc="Tokenizing dataset"
        )
        
        # キャッシュに保存（use_cacheがTrueでcacheパスが指定されている場合）
        if self.use_cache and self.cache is not None:
            os.makedirs(self.cache, exist_ok=True)
            print(f"Saving tokenized dataset to cache: {self.cache}")
            ds.save_to_disk(self.cache)
        
        # torch形式に設定してシャッフル
        ds.set_format(type='torch', columns=['input_ids'])
        ds = ds.shuffle(seed + self.rank)
        self.dataset = ds
        print("Dataset loaded and processed.")

    def _find_text_column(self, ds):
        """テキストカラム名を推定するヘルパーメソッド"""
        if 'text' in ds.column_names:
            return 'text'
        
        # streaming modeの場合はfirst exampleを取得
        if hasattr(ds, 'take'):
            sample = next(iter(ds.take(1)))
        else:
            sample = ds[0]
        
        for col in ds.column_names:
            if isinstance(sample[col], str):
                return col
        
        raise RuntimeError(f"Could not find a text column in dataset. Columns: {ds.column_names}")

    def _get_tokenize_fn(self, text_col):
        """トークナイズ関数を返すヘルパーメソッド"""
        def tokenize_fn(batch):
            return self.tokenizer(
                batch[text_col],
                truncation=True,
                max_length=self.config.max_seq_len,
                padding='max_length'
            )
        return tokenize_fn

    def num_train_steps(self):
        print("Calculating number of training steps...")
        return math.ceil((self.train_end_idx-self.train_start_idx) / self.config.batch_size)



class Trainer():
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.num_epochs = config.num_epochs

        self.use_moe = config.use_moe
        self.use_lossfreebalance = config.use_lossfreebalance if self.use_moe else False
        self.clean_cuda_cache = config.clean_cuda_cache
        self.dtype = getattr(torch, self.config.use_dtype)

        self.steps_for_eval = config.steps_for_eval
        self.weight_decay = config.weight_decay
        self.update_rate = config.update_rate if self.use_moe else 0

        self.device = torch.device(f"cuda:0") if torch.cuda.is_available() else 'cpu' #cpuでは学習はできません
        if self.device == 'cpu':
            raise RuntimeError("Training on CPU is not supported. Please use a GPU.")

        if self.device.type == 'cuda':
            torch.cuda.manual_seed(config.seed)
            n_gpus = torch.cuda.device_count()

        # 再開用の変数
        if config.continue_train and config.checkpoint_path != "":
            self.load_checkpoint(config.checkpoint_path)
            print(f"Continuing training from checkpoint: {config.checkpoint_path}")

        if hasattr(self, "_checkpoint_model_state"):
            print(f"Restoring model weights from checkpoint...")
            state_dict = self._checkpoint_model_state
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[len("_orig_mod."):]] = v
                else:
                    new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)


        use_compile = self.config.use_compile and self.device.type == "cuda" and torch.__version__.startswith("2")
        if use_compile:
            self.model = torch.compile(self.model)
            
        # DDP
        if n_gpus > 1 and config.use_ddp:   
            self.ddp = True
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = torch.device(f"cuda:{self.ddp_local_rank}")
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0

            self.model.to(self.device)
            
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
            self.raw_m = model
        else:
            self.ddp = False
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.master_process = True

            if self.device != "cpu":
                self.model.to(self.device)

        if self.master_process:
            print("Device:", self.device)
            print(f"Model's trainable params: {sum([p.data.numel() for p in self.model.parameters() if p.requires_grad]) / 1e6:.2f}M")
            print(f"Tokens per step: {self.config.batch_size * self.config.max_seq_len * self.ddp_world_size * self.config.accumulation_steps}")
            print(f"use {'torch.compile()'}: {use_compile}")
            print(f"Use MoE: {'Yes ' if self.use_moe else 'No'}")
            if self.use_moe:
                print(f"Number of experts: {self.model.blocks[0].ffn.num_experts}")
                print(f"Number of used experts during inference: {self.model.blocks[0].ffn.moe_active_experts}")
                print(f"Method of aux_loss: {'loss-free-balance' if config.use_lossfreebalance else 'default'}")
                print(f"Number of parameters will be used during inference: {((sum([p.data.numel() for p in self.model.parameters() if p.requires_grad]) - sum(p.numel() for p in self.model.blocks[0].ffn.parameters()) * len(self.model.blocks) * (1-(self.model.blocks[0].ffn.moe_active_experts + self.model.blocks[0].ffn.moe_shared_experts) / (self.model.blocks[0].ffn.num_experts + self.model.blocks[0].ffn.moe_shared_experts)))) / 1e6:.2f}M")
    
        torch.manual_seed(config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(config.seed)
        if self.device.type == 'cpu':
            torch.set_num_threads(os.cpu_count())
            print("Using CPU with num_threads:", torch.get_num_threads())



    def step(self, data_loader, accumulation_steps: int,
              num_tokens: int, split: str = "train"):
        """
        Performs single forward/backward pass with gradient accumulation.
            Returns: (total_loss, cross_entropy_loss, number_of_processed_tokens)
        """
        x, y = data_loader.next_batch(split=split)
        x, y = x.to(self.device), y.to(self.device)
        num_tokens += torch.numel(x)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            _, loss, ce_loss = self.model(x, y)

        loss /= accumulation_steps

        loss.backward()
        return loss, ce_loss, num_tokens
    

    def train(self, data_loader):
        num_steps_per_epoch = math.ceil(data_loader.num_train_steps() / self.config.accumulation_steps)
        print("Preparing for training...")
        print(f"data_loader.num_train_steps(): {data_loader.num_train_steps()}, accumulation_steps: {self.config.accumulation_steps}")
        print("DDP:", self.ddp, f"(world_size: {self.ddp_world_size})" if self.ddp else "")
        print("Use MoE:", self.use_moe)
        print("Use loss-free balancing:", self.use_lossfreebalance if self.use_moe else "N/A")
        print("Batch size per GPU:", self.config.batch_size)
        print("Max sequence length:", self.config.max_seq_len)
        print("Gradient accumulation steps:", self.config.accumulation_steps)
        print("Effective batch size:", self.config.batch_size * self.config.accumulation_steps * self.ddp_world_size)
        print(f"Total tokens per step:", self.config.batch_size * self.config.max_seq_len * self.ddp_world_size * self.config.accumulation_steps)
        print(f"Total tokens per epoch:", self.config.batch_size * self.config.max_seq_len * num_steps_per_epoch * self.config.accumulation_steps)
        print(f"Number of steps per epoch: {num_steps_per_epoch}")

        
        # Configuration of optimizer and schedulers
        optimizer = torch.optim.AdamW(
            self.model.parameters(),  
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.weight_decay,
            fused=(self.device.type=="cuda")
        )

        total_steps = num_steps_per_epoch * self.num_epochs
        warmup_steps = math.floor(self.config.warmup_ratio * total_steps)
        warmup_factor = lambda step: 0.05 + 0.95 * (step / max(warmup_steps, 1))
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_factor
        )

        cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - warmup_steps, 
            eta_min=0.1 * self.config.learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cos_scheduler],
            milestones=[warmup_steps])
        
        # 再開時の処理
        start_epoch = 0
        start_step = 0
        global_step = 0  # グローバルステップカウンタ
        
        #学習設定の復元
        if hasattr(self, "_checkpoint_optimizer_state"):
            print(f"Restoring training state from checkpoint...")
            optimizer.load_state_dict(self._checkpoint_optimizer_state)
            if self._checkpoint_scheduler_state is not None:
                scheduler.load_state_dict(self._checkpoint_scheduler_state)
            data_loader.train_current_idx = self._checkpoint_train_idx
            data_loader.val_current_idx = self._checkpoint_val_idx
            
            # チェックポイントから再開位置を計算
            start_epoch = self._checkpoint_epoch
            start_step = self._checkpoint_step+1
            global_step = self._checkpoint_global_step+1 if hasattr(self, '_checkpoint_global_step') else (start_epoch * num_steps_per_epoch + start_step+1)
            
            if self._checkpoint_rng_state is not None:
                torch.random.set_rng_state(self._checkpoint_rng_state)
            if self._checkpoint_cuda_rng_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self._checkpoint_cuda_rng_state)
            print(f"Training state restored from checkpoint: epoch={start_epoch}, step={start_step}, global_step={global_step}")

        self.model.train()

        print("Starting training...")
        print(f"Total epochs: {self.num_epochs}, steps per epoch: {num_steps_per_epoch}, total steps: {total_steps}")
        if start_epoch > 0 or start_step > 0:
            print(f"Resuming from epoch {start_epoch}, step {start_step} (global step {global_step})")

        # エポックとステップの再開処理
        for epoch in range(start_epoch, self.num_epochs):
            # 最初のエポックの場合は start_step から、それ以外は 0 から開始
            epoch_start_step = start_step if epoch == start_epoch else 0
            
            for step in range(epoch_start_step, num_steps_per_epoch):
                
                t0 = time.perf_counter()
                accumulated_loss = 0.0
                num_tokens = 0

                ddp_nosync_ctx = self.model.no_sync() if self.ddp else nullcontext()
                with ddp_nosync_ctx:
                    for _ in range(self.config.accumulation_steps - 1):
                        loss, ce_loss, num_tokens = self.step(data_loader, self.config.accumulation_steps, num_tokens, split="train")
                        accumulated_loss += loss

                loss, ce_loss, num_tokens = self.step(data_loader, self.config.accumulation_steps, num_tokens, split="train")
                accumulated_loss += loss.detach()

                # MoE の loss-free balance 処理
                if self.use_moe and self.use_lossfreebalance: 
                    for block in range(len(self.model.blocks)):
                        expert_counts = torch.bincount(ce_loss[1].flatten(), minlength=self.model.blocks[block].ffn.moe_active_experts)  
                        avg_count = expert_counts.float().mean()
                        for i, count in enumerate(expert_counts):
                            error = avg_count - count.float()
                            self.model.blocks[block].ffn.expert_biases.data[i] += self.update_rate * torch.sign(error)

                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                t1 = time.perf_counter()
                tokens_per_sec = num_tokens / (t1 - t0) * self.ddp_world_size

                # Logging 
                if self.master_process:
                    print(f"Epoch: {epoch} | Step: {step}/{num_steps_per_epoch} | Global Step: {global_step} | loss: {accumulated_loss:.4f} | norm: {norm:.4f} | lr: {scheduler.get_last_lr()[0]:.10e} | tok/s: {tokens_per_sec:.1f} | dataset idx: {data_loader.train_current_idx}/{data_loader.train_end_idx}")
                
                # Evaluation 
                if self.master_process and ((global_step > 0 and global_step % self.config.eval_interval == 0) or step == num_steps_per_epoch - 1):
                    
                    self.model.eval() 
                    val_loss = self.eval(data_loader)
                    print(f"Evaluation at global step {global_step}: val_loss = {val_loss:.4f}")

                    with open(self.config.eval_log_file, "a") as f:
                        f.write(f"Global Step: {global_step}, Epoch: {epoch}, Step: {step}, val_loss: {val_loss:.4f}, norm: {norm:.4f}, lr: {scheduler.get_last_lr()[0]:.10e}, time: {t1 - t0:.2f}s, tok/s: {tokens_per_sec:.1f} | dataset idx: {data_loader.val_current_idx}/{data_loader.len_dataset}\n")

                    self.model.train()
                    if self.clean_cuda_cache:
                        print("Cleaning CUDA cache")
                        torch.cuda.empty_cache()

                # Save Checkpoints
                if self.master_process and ((global_step % self.config.checkpoints_frequency == 0 and global_step > 0) or step == num_steps_per_epoch - 1):
                    self.save_checkpoints(
                        optimizer, scheduler, data_loader, 
                        self.config.path_to_checkpoints, 
                        epoch=epoch, step=step, global_step=global_step
                    )

                global_step += 1

    def save_checkpoints(self, optimizer, scheduler, data_loader, path: str, epoch: int, step: int, global_step: int):
        
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"model.checkpoint.epoch{epoch}_step{step}_global{global_step}.pt")
        
        # チェックポイント管理
        if self.config.max_checkpoints_to_keep != 0:  # 0の場合は制限なし
            checkpoint_files = glob.glob(os.path.join(path, "model.checkpoint.*.pt"))

            # global_stepの数値でソート（文字列ソートではなく数値ソート）
            def extract_global_step(filename):
                match = re.search(r'global(\d+)\.pt$', filename)
                return int(match.group(1)) if match else 0

            existing_checkpoints = sorted(checkpoint_files, key=extract_global_step)
            
            if self.config.max_checkpoints_to_keep == -1:
                # 最新1つのみ保持（全て削除）
                for old_checkpoint in existing_checkpoints:
                    os.remove(old_checkpoint)
                    print(f"Deleted old checkpoint: {old_checkpoint}")
            elif len(existing_checkpoints) >= self.config.max_checkpoints_to_keep:
                # 指定数以上ある場合、古いものから削除
                num_to_delete = len(existing_checkpoints) - self.config.max_checkpoints_to_keep + 1
                for old_checkpoint in existing_checkpoints[:num_to_delete]:
                    os.remove(old_checkpoint)
                    print(f"Deleted old checkpoint: {os.path.basename(old_checkpoint)}")
        
        # チェックポイント保存
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'global_step': global_step,
            'train_current_idx': data_loader.train_current_idx,
            'val_current_idx': data_loader.val_current_idx,
            'rng_state': torch.random.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path} (keeping max {self.config.max_checkpoints_to_keep} checkpoints)")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self._checkpoint_model_state = checkpoint['model']
        self._checkpoint_optimizer_state = checkpoint['optimizer']
        self._checkpoint_scheduler_state = checkpoint.get('scheduler', None)
        self._checkpoint_epoch = checkpoint.get('epoch', 0)
        self._checkpoint_step = checkpoint.get('step', 0)
        self._checkpoint_global_step = checkpoint.get('global_step', 0) 
        self._checkpoint_train_idx = checkpoint.get('train_current_idx', 0)
        self._checkpoint_val_idx = checkpoint.get('val_current_idx', 0)
        self._checkpoint_rng_state = checkpoint.get('rng_state', None)
        self._checkpoint_cuda_rng_state = checkpoint.get('cuda_rng_state', None)

        print(f"Checkpoint loaded: epoch={self._checkpoint_epoch}, step={self._checkpoint_step}, global_step={self._checkpoint_global_step}")

    def eval(self, data_loader):
        print("Running evaluation...")
        print(f"data_loader.val_current_idx: {data_loader.val_current_idx}, data_loader.val_start_idx: {data_loader.val_start_idx}, data_loader.len_dataset: {data_loader.len_dataset}")
        """
        Evaluates model on validation split using running average of first [steps_for_eval] batches
        """
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in tqdm(range(self.steps_for_eval)):
                x, y = data_loader.next_batch(split="val")
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                    _, loss, ce_loss = self.model(x, y)
                loss /= self.steps_for_eval
                val_loss_accum += loss.detach()
            return val_loss_accum

