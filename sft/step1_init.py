import os
import json
import math
import time
import random
import warnings
from typing import Optional, Tuple, List, Union, Iterator
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from contextlib import nullcontext # For mixed precision

# From Hugging Face libraries
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.activations import ACT2FN
from tokenizers import Tokenizer as HFTokenizer # Renaming to avoid conflict if any
from tokenizers import models as hf_models
from tokenizers import trainers as hf_trainers
from tokenizers import pre_tokenizers as hf_pre_tokenizers
from tokenizers import decoders as hf_decoders

warnings.filterwarnings('ignore')
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

SPECIAL_TOKENS_LIST = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<pad>"]

def logger(content):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {content}")

# 解释 get_lr 函数的作用和实现逻辑（中文注释）：
# get_lr 是一个用于调整学习率的调度函数，常用于深度学习训练过程中的学习率动态调整。
# 它实现了“余弦退火（cosine decay）”与“线性预热（linear warmup）”相结合的学习率策略。
# 主要参数说明：
#   - current_step: 当前训练步数
#   - total_steps: 总训练步数
#   - initial_lr: 初始学习率
#   - min_lr_ratio: 最小学习率与初始学习率的比值（默认0.1，即最低为初始的10%）
#   - warmup_ratio: 预热步数占总步数的比例（默认1%）
# 实现逻辑：
#   1. 在预热阶段（warmup_steps内），学习率从0线性增加到initial_lr。
#   2. 预热结束后，进入余弦退火阶段，学习率从initial_lr逐步下降到min_lr。
#   3. 如果current_step超过total_steps，则学习率保持在min_lr。
# 这种策略有助于模型在训练初期稳定收敛，并在后期细致微调，常用于Transformer等大模型训练。

def get_lr(current_step, total_steps, initial_lr, min_lr_ratio=0.1, warmup_ratio=0.01):
    """Cosine decay learning rate scheduler with linear warmup."""
    warmup_steps = int(warmup_ratio * total_steps)
    min_lr = initial_lr * min_lr_ratio
    if warmup_steps > 0 and current_step < warmup_steps:
        return initial_lr * (current_step / warmup_steps)
    elif current_step > total_steps:
        return min_lr
    else:
        decay_steps = total_steps - warmup_steps
        progress = (current_step - warmup_steps) / max(1, decay_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + coeff * (initial_lr - min_lr)

def print_model_summary(model, model_name="Model"):
    logger(f"--- {model_name} Summary ---")
    if hasattr(model, 'config'):
      logger(f"Configuration: {model.config}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger(f"Total parameters: {total_params / 1e6:.3f} M ({total_params})")
    logger(f"Trainable parameters: {trainable_params / 1e6:.3f} M ({trainable_params})")
    logger("-------------------------")


NOTEBOOK_DATA_DIR = "./dataset_notebook_scratch"
os.makedirs(NOTEBOOK_DATA_DIR, exist_ok=True)
pretrain_file_path = os.path.join(NOTEBOOK_DATA_DIR, "pretrain_data.jsonl")
reasoning_file_path = os.path.join(NOTEBOOK_DATA_DIR, "reasoning_data.jsonl")
sft_file_path = os.path.join(NOTEBOOK_DATA_DIR, "sft_data.jsonl")


# --- Pretraining Data ---
sample_pretrain_data = [
    {"text": "The sun shines brightly in the clear blue sky."},
    {"text": "Cats love to chase mice and play with yarn balls."},
    {"text": "Reading books expands your knowledge and vocabulary."},
    {"text": "Artificial intelligence is a rapidly evolving field of study."},
    {"text": "To bake a cake, you need flour, sugar, eggs, and butter."},
    {"text": "Large language models are trained on vast amounts of text data."},
    {"text": "The quick brown fox jumps over the lazy dog."}
]
pretrain_file_path = os.path.join(NOTEBOOK_DATA_DIR, "pretrain_data.jsonl")
with open(pretrain_file_path, 'w', encoding='utf-8') as f:
    for item in sample_pretrain_data:
        f.write(json.dumps(item) + '\n')
logger(f"Sample pretraining data created at: {pretrain_file_path}")

# --- SFT Data ---
sample_sft_data = [
    {"conversations": [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am doing well, thank you! How can I help you today?"}
    ]},
    {"conversations": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."}
    ]},
    {"conversations": [
        {"role": "user", "content": "Explain gravity in simple terms."},
        {"role": "assistant", "content": "Gravity is the force that pulls objects towards each other. It's why things fall down to the ground!"}
    ]}
]
sft_file_path = os.path.join(NOTEBOOK_DATA_DIR, "sft_data.jsonl")
with open(sft_file_path, 'w', encoding='utf-8') as f:
    for item in sample_sft_data:
        f.write(json.dumps(item) + '\n')
logger(f"Sample SFT data created at: {sft_file_path}")

# --- Reasoning Data ---
sample_reasoning_data = [
    {"conversations": [
        {"role": "user", "content": "If I have 3 apples and eat 1, how many are left?"},
        {"role": "assistant", "content": "<think>The user starts with 3 apples. The user eats 1 apple. This means 1 apple is subtracted from the initial amount. So, 3 - 1 = 2.</think><answer>You have 2 apples left.</answer>"}
    ]},
    {"conversations": [
        {"role": "user", "content": "What are the primary colors?"},
        {"role": "assistant", "content": "<think>The user is asking about primary colors. These are colors that cannot be made by mixing other colors. The standard set of primary colors in additive color models (like light) are Red, Green, and Blue (RGB). For subtractive models (like paint), they are often considered Red, Yellow, Blue (RYB) or Cyan, Magenta, Yellow (CMY).</think><answer>The primary colors are typically considered to be red, yellow, and blue. These are colors that can be mixed to create a range of other colors but cannot be created by mixing other colors themselves.</answer>"}
    ]}
]
reasoning_file_path = os.path.join(NOTEBOOK_DATA_DIR, "reasoning_data.jsonl")
with open(reasoning_file_path, 'w', encoding='utf-8') as f:
    for item in sample_reasoning_data:
        f.write(json.dumps(item) + '\n')
logger(f"Sample reasoning data created at: {reasoning_file_path}")

# --- Device & Seeds ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SEED = 42
# 判断当前环境是否有可用的CUDA（NVIDIA显卡），并且显卡是否支持bfloat16数据类型
# 如果都支持，则使用"bfloat16"；否则使用"float16"
DTYPE_STR = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

# 根据DTYPE_STR和设备类型，选择PyTorch中对应的数据类型对象
# 如果是CUDA设备，使用DTYPE_STR指定的数据类型（bfloat16或float16）
# 如果不是CUDA设备（如CPU），则强制使用float32，保证兼容性和精度
PTDTYPE = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'float16': torch.float16
}[DTYPE_STR if DEVICE.type == 'cuda' else 'float32']

torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)

# --- Tokenizer Configuration ---
DEMO_VOCAB_SIZE = 32000  # 词表大小
DEMO_HIDDEN_SIZE = 1024  # 隐藏层维度
DEMO_NUM_LAYERS = 24     # Transformer 层数
DEMO_NUM_ATTENTION_HEADS = 16  # 注意力头数
DEMO_NUM_KV_HEADS = 16         # KV头数
DEMO_MAX_SEQ_LEN = 1024        # 最大序列长度
DEMO_INTERMEDIATE_SIZE = int(DEMO_HIDDEN_SIZE * 8 / 3)  # 前馈层中间维度
DEMO_INTERMEDIATE_SIZE = 32 * ((DEMO_INTERMEDIATE_SIZE + 32 - 1) // 32)  # 向上取整到32的倍数

# --- Training Hyperparameters (Larger Model) ---
DEMO_PRETRAIN_EPOCHS = 10
DEMO_SFT_EPOCHS = 10
DEMO_REASONING_EPOCHS = 10
DEMO_BATCH_SIZE = 16
DEMO_PRETRAIN_LR = 3e-4
DEMO_SFT_LR = 1e-4
DEMO_REASONING_LR = 5e-5

# --- Directories (unchanged) ---
NOTEBOOK_OUT_DIR = "./out_notebook_scratch"
NOTEBOOK_DATA_DIR = "./dataset_notebook_scratch"
NOTEBOOK_TOKENIZER_PATH = os.path.join(NOTEBOOK_OUT_DIR, "demo_tokenizer.json")
os.makedirs(NOTEBOOK_OUT_DIR, exist_ok=True)
os.makedirs(NOTEBOOK_DATA_DIR, exist_ok=True)
pretrain_file_path = os.path.join(NOTEBOOK_DATA_DIR, "pretrain_data.jsonl")
reasoning_file_path = os.path.join(NOTEBOOK_DATA_DIR, "reasoning_data.jsonl")
sft_file_path = os.path.join(NOTEBOOK_DATA_DIR, "sft_data.jsonl")

def logger(msg):
    print(f"[LOG]: {msg}")

logger(f"Using device: {DEVICE}")
logger(f"Using PyTorch dtype: {PTDTYPE} (derived from DTYPE_STR: {DTYPE_STR if DEVICE.type == 'cuda' else 'float32'})")
logger(f"Output directory: {NOTEBOOK_OUT_DIR}")
logger(f"Data directory: {NOTEBOOK_DATA_DIR}")
logger(f"Trained tokenizer will be saved to/loaded from: {NOTEBOOK_TOKENIZER_PATH}")