
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

import os
from typing import List
from step1_init import DEMO_VOCAB_SIZE, NOTEBOOK_DATA_DIR, NOTEBOOK_TOKENIZER_PATH, logger,pretrain_file_path,DEMO_MAX_SEQ_LEN
from step2_tokenizer import trained_hf_tokenizer

from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.activations import ACT2FN
from tokenizers import Tokenizer as HFTokenizer # Renaming to avoid conflict if any
from tokenizers import models as hf_models
from tokenizers import trainers as hf_trainers
from tokenizers import pre_tokenizers as hf_pre_tokenizers
from tokenizers import decoders as hf_decoders

from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as HFByteLevelPreTokenizer


class DemoCorpusDataset(Dataset):
    """Dataset for pretraining. Loads text, tokenizes, and prepares X, Y pairs."""
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        logger(f"Loading pretraining data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip())['text'])
        logger(f"Loaded {len(self.samples)} samples for pretraining.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        # For pretraining, we typically add BOS and EOS if not implicitly handled by tokenizer during training.
        # Here, we ensure BOS is prepended. EOS will be part of the sequence if it fits. 
        full_text_with_bos = self.tokenizer.bos_token + text

        encoding = self.tokenizer(
            full_text_with_bos,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze(0) # (max_length)

        # Create loss mask: 1 for non-pad tokens, 0 for pad tokens.
        # Loss is calculated on Y (shifted input_ids), so the mask should align with Y.
        effective_loss_mask = (input_ids != self.tokenizer.pad_token_id).long()

        X = input_ids[:-1]  # (max_length - 1)
        Y = input_ids[1:]   # (max_length - 1)
        mask_for_loss_calculation = effective_loss_mask[1:] # Align with Y

        return X, Y, mask_for_loss_calculation

###CallableTokenizerWrapper 是一个包装器（wrapper），
# 它的目的是让 HuggingFace 的 tokenizers 库训练出来的分词器（trained_hf_tokenizer）用起来更像 transformers 里的 AutoTokenizer，
# 方便在后续训练、推理等流程中无缝切换。

class CallableTokenizerWrapper:
    def __init__(self, base_tokenizer):
        self.tokenizer = base_tokenizer
        # Set token IDs
        self.pad_token_id = self.tokenizer.token_to_id("<pad>") if self.tokenizer.token_to_id("<pad>") is not None else self.tokenizer.token_to_id("<|endoftext|>")
        self.eos_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.bos_token_id = self.tokenizer.token_to_id("<|im_start|>")

        # Set string attributes
        self.bos_token = "<|im_start|>"
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<pad>" if self.tokenizer.token_to_id("<pad>") is not None else "<|endoftext|>"
        self.unk_token = "<unk>"

        self.vocab_size = self.tokenizer.get_vocab_size()

    def __call__(self, text, max_length=None, padding=None, truncation=None, return_tensors=None):
        # Implement basic functionality of AutoTokenizer.__call__
        if isinstance(text, list):
            encodings = [self.tokenizer.encode(t) for t in text]
        else:
            encodings = [self.tokenizer.encode(text)]

        # Convert encodings to lists of IDs if they aren't already
        token_ids = []
        for enc in encodings:
            if hasattr(enc, 'ids'):
                token_ids.append(enc.ids)
            else:
                # If encode returns a list directly, use it as is
                token_ids.append(enc)

        # Handle max_length and padding
        if max_length is not None:
            if truncation:
                token_ids = [ids[:max_length] for ids in token_ids]
            if padding == "max_length":
                token_ids = [ids + [self.pad_token_id] * (max_length - len(ids)) if len(ids) < max_length else ids[:max_length] for ids in token_ids]

        # Convert to tensors if requested
        if return_tensors == 'pt':
            import torch
            input_ids = torch.tensor(token_ids)  #这个包装器，将token_ids转换为torch.Tensor类型，并返回一个TokenizerOutput对象

            # Create a proper TokenizerOutput class with a to() method
            class TokenizerOutput:
                def __init__(self, input_ids):
                    self.input_ids = input_ids

                def to(self, device):
                    self.input_ids = self.input_ids.to(device)
                    return self

            return TokenizerOutput(input_ids)

        return token_ids

    def apply_chat_template(self, conversations, tokenize=True, add_generation_prompt=False, return_tensors=None, max_length=None, truncation=None, padding=None):
        """Applies a chat template to format conversation messages."""
        # Define chat template similar to what was used in the original tokenizer
        formatted_text = ""
        for message in conversations:
            formatted_text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"

        # Add generation prompt if requested
        if add_generation_prompt:
            formatted_text += "<|im_start|>assistant\n"

        # Return the string if tokenize=False
        if not tokenize:
            return formatted_text

        # Otherwise tokenize and return tensor
        input_ids = self(
            formatted_text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        ).input_ids

        return input_ids
    def encode(self, text, add_special_tokens=True):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids, skip_special_tokens=False):
        if hasattr(token_ids, 'tolist'):  # If it's a tensor
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

# Wrap the tokenizer with our callable wrapper
tokenizer = CallableTokenizerWrapper(trained_hf_tokenizer)
DEMO_VOCAB_SIZE_FINAL = tokenizer.vocab_size

logger(f"Fallback Tokenizer - Vocab Size: {tokenizer.vocab_size}")
logger(f"Fallback Tokenizer - PAD token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
logger(f"Fallback Tokenizer - EOS token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")
logger(f"Fallback Tokenizer - BOS token: '{tokenizer.bos_token}', ID: {tokenizer.bos_token_id}")

logger("Testing DemoCorpusDataset...")



try:
    test_pretrain_ds = DemoCorpusDataset(pretrain_file_path, tokenizer, DEMO_MAX_SEQ_LEN)
    X_pt_sample, Y_pt_sample, mask_pt_sample = test_pretrain_ds[0]
    logger(f"Sample X (Pretrain): {X_pt_sample.shape}, {X_pt_sample[:10]}...")
    logger(f"Sample Y (Pretrain): {Y_pt_sample.shape}, {Y_pt_sample[:10]}...")
    logger(f"Sample Mask (Pretrain): {mask_pt_sample.shape}, {mask_pt_sample[:10]}...")
    logger(f"Decoded X with BOS: {tokenizer.decode(torch.cat([torch.tensor([tokenizer.bos_token_id]), X_pt_sample[:torch.sum(mask_pt_sample)]]))}")
    logger(f"Decoded Y: {tokenizer.decode(Y_pt_sample[:torch.sum(mask_pt_sample)])}")
except Exception as e:
    logger(f"Error loading trained tokenizer with AutoTokenizer: {e}")
    logger("Falling back to using the HFTokenizer object directly (chat_template might not work as expected).")

    