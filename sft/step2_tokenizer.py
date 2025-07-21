import os
from typing import List
from step1_init import DEMO_VOCAB_SIZE, NOTEBOOK_DATA_DIR, NOTEBOOK_TOKENIZER_PATH, SPECIAL_TOKENS_LIST, logger

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


tokenizer_corpus = [
    "Hello world, this is a demonstration of building a thinking LLM.",
    "Language models learn from text data.",
    "Tokenization is a crucial first step.",
    "We will train a BPE tokenizer.",
    "Think before you answer.",
    "The answer is forty-two.",
    "<think>Let's consider the options.</think><answer>Option A seems best.</answer>",
    "<|im_start|>user\nWhat's up?<|im_end|>\n<|im_start|>assistant\nNot much!<|im_end|>"
]

# Save to a temporary file for the tokenizer trainer
tokenizer_corpus_file = os.path.join(NOTEBOOK_DATA_DIR, "tokenizer_corpus.txt")
with open(tokenizer_corpus_file, 'w', encoding='utf-8') as f:
    for line in tokenizer_corpus:
        f.write(line + "\n")

logger(f"Tokenizer training corpus saved to: {tokenizer_corpus_file}")

def train_demo_tokenizer(corpus_files: List[str], vocab_size: int, save_path: str, special_tokens: List[str]):
    logger(f"Starting tokenizer training with vocab_size={vocab_size}...")


    # 初始化BPE模型
    tokenizer_bpe = HFTokenizer(hf_models.BPE(unk_token="<unk>")) #用 BPE 算法来训练分词器

    # Pre-tokenizer: splits text into words, then processes at byte-level for OOV robustness.
    # ByteLevel(add_prefix_space=False) is common for models like GPT-2/LLaMA.
    tokenizer_bpe.pre_tokenizer = hf_pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)

    # Decoder: Reconstructs text from tokens, handling byte-level tokens correctly.
    tokenizer_bpe.decoder = hf_decoders.ByteLevel()

    # Trainer: BpeTrainer with specified vocab size and special tokens.
    # The initial_alphabet from ByteLevel ensures all single bytes are potential tokens.
    trainer = hf_trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=hf_pre_tokenizers.ByteLevel.alphabet()
    )

    # Train the tokenizer
    if isinstance(corpus_files, str): # If a single file path string
        corpus_files = [corpus_files]

    tokenizer_bpe.train(corpus_files, trainer=trainer)
    logger(f"Tokenizer training complete. Vocab size: {tokenizer_bpe.get_vocab_size()}")

    # Save the tokenizer
    # Saving as a single JSON file makes it compatible with AutoTokenizer.from_pretrained()
    tokenizer_bpe.save(save_path)
    logger(f"Tokenizer saved to {save_path}")
    return tokenizer_bpe

# Train and save our demo tokenizer
trained_hf_tokenizer = train_demo_tokenizer(
    corpus_files=[tokenizer_corpus_file],
    vocab_size=DEMO_VOCAB_SIZE,
    save_path=NOTEBOOK_TOKENIZER_PATH,
    special_tokens=SPECIAL_TOKENS_LIST
)

# Verify special tokens are present and correctly mapped
logger("Trained Tokenizer Vocab (first 10 and special tokens):")
vocab = trained_hf_tokenizer.get_vocab()
for i, (token, token_id) in enumerate(vocab.items()):
    if i < 10 or token in SPECIAL_TOKENS_LIST:
        logger(f"  '{token}': {token_id}")

# Test encoding and decoding with the trained tokenizer object
test_sentence = "Hello <|im_start|> world <think>思考中</think><answer>答案</answer> <|im_end|>"
encoded = trained_hf_tokenizer.encode(test_sentence)
logger(f"Original: {test_sentence}")
logger(f"Encoded IDs: {encoded.ids}")
logger(f"Encoded Tokens: {encoded.tokens}")
decoded = trained_hf_tokenizer.decode(encoded.ids)
logger(f"Decoded: {decoded}")