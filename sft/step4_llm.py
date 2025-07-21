
import math
from typing import List, Optional, Tuple, Union
from step3_auto_tokenizer import DEMO_VOCAB_SIZE_FINAL,tokenizer
from step1_init import DEMO_HIDDEN_SIZE,DEMO_INTERMEDIATE_SIZE,DEMO_NUM_LAYERS,DEMO_NUM_ATTENTION_HEADS,DEMO_NUM_KV_HEADS,DEMO_MAX_SEQ_LEN, DEVICE, logger, print_model_summary
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

import torch.nn.functional as F

from transformers.activations import ACT2FN

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

import torch.nn as nn
import torch

from transformers.generation.utils import GenerationMixin

# 该部分定义了DemoLLMConfig类的初始化方法，主要用于配置和初始化大语言模型的各项超参数。
# 这些参数包括词表大小、隐藏层维度、中间层维度、层数、多头注意力头数、激活函数类型、最大序列长度、归一化epsilon、RoPE位置编码参数、dropout概率、缓存和Flash Attention开关等。
# 同时，还会根据传入的分词器（tokenizer）自动设置BOS、EOS和PAD的token id，确保模型与分词器的特殊符号一致。
# 如果分词器未定义pad_token_id，则默认将其设置为eos_token_id。
# 最后，调用父类的初始化方法，将这些参数传递给基础的配置类。
class DemoLLMConfig(PretrainedConfig):
    model_type = "demo_llm" # Generic name

    def __init__(
        self,
        vocab_size: int = DEMO_VOCAB_SIZE_FINAL,  # 词表大小，等于分词器的vocab size，决定模型能处理多少不同的token
        hidden_size: int = DEMO_HIDDEN_SIZE,      # 隐藏层维度，Transformer每层的特征维数
        intermediate_size: int = DEMO_INTERMEDIATE_SIZE,  # 前馈网络的中间层维度，通常是hidden_size的4倍
        num_hidden_layers: int = DEMO_NUM_LAYERS,         # Transformer的层数（即堆叠多少个Encoder/Decoder Block）
        num_attention_heads: int = DEMO_NUM_ATTENTION_HEADS,  # 多头自注意力机制的头数
        num_key_value_heads: Optional[int] = DEMO_NUM_KV_HEADS, # KV头数，部分模型支持解耦Q和KV的头数
        hidden_act: str = "silu",                   # 激活函数类型，常用有"relu"、"gelu"、"silu"等
        max_position_embeddings: int = DEMO_MAX_SEQ_LEN,  # 最大支持的序列长度（即最大token数）
        rms_norm_eps: float = 1e-5,                 # RMSNorm归一化的epsilon，防止除零
        rope_theta: float = 10000.0,                # RoPE位置编码的theta参数，影响旋转频率
        bos_token_id: int = 1,                      # 序列起始（BOS）token的ID，通常由分词器决定
        eos_token_id: int = 2,                      # 序列结束（EOS）token的ID，通常由分词器决定
        pad_token_id: Optional[int] = None,         # 填充（PAD）token的ID，通常由分词器决定
        dropout: float = 0.0,                       # dropout概率，防止过拟合
        use_cache: bool = True,                     # 是否启用缓存（如推理时缓存KV以加速）
        flash_attn: bool = True,                    # 是否启用Flash Attention（高效注意力实现，需硬件支持）
        **kwargs                                     # 其他可选参数
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_attention_heads % self.num_key_value_heads != 0:
             raise ValueError(f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.use_cache = use_cache
        self.flash_attn = flash_attn

        # Update BOS/EOS/PAD from tokenizer if available
        if tokenizer is not None and hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            bos_token_id = tokenizer.bos_token_id
        if tokenizer is not None and hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            eos_token_id = tokenizer.eos_token_id
        if tokenizer is not None and hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        else: # Default pad to eos if not defined
            pad_token_id = eos_token_id

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
logger("DemoLLMConfig defined.")


# RotaryEmbedding（旋转位置编码）是一种用于Transformer模型中的位置编码方法，常用于Llama、GPT-NeoX等大模型。
# 它的核心思想是将每个token的位置信息通过复数旋转的方式编码到注意力的Q/K向量中，
# 从而实现无损且高效的位置感知。
# 具体做法是：先为每个位置和每个偶数维度计算一个旋转频率（freqs），
# 然后将Q/K向量的每对相邻维度视为一个复数，乘以对应位置的旋转因子（即cos+jsin），
# 这样就把位置信息“旋转”进了向量。
# 这样做的好处是：支持任意长度的推理（外推性好），且不会引入额外的参数。
# RotaryEmbedding的实现中，forward方法会对输入的Q/K向量进行reshape、复数乘法和还原，最终输出带有位置信息的Q/K向量。
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0, device=None):
        super().__init__()
        # freqs: (max_seq_len, dim/2)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)

        # freqs_cis: (max_seq_len, dim/2) holding complex numbers cos(m*theta_i) + j*sin(m*theta_i)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs), persistent=False)
        logger(f"Initialized RotaryEmbedding with dim={dim}, max_seq_len={max_seq_len}")

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, seq_len: int):
        # xq, xk: (bsz, num_heads, seq_len, head_dim)
        # freqs_cis: (max_seq_len, head_dim/2) -> slice to (seq_len, head_dim/2)

        # Reshape xq, xk to (bsz, num_heads, seq_len, head_dim/2, 2) to treat pairs for complex mul
        xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)

        # Convert to complex: (bsz, num_heads, seq_len, head_dim/2)
        xq_c = torch.view_as_complex(xq_r)
        xk_c = torch.view_as_complex(xk_r)

        # Slice freqs_cis for the current sequence length
        # freqs_cis_pos: (seq_len, head_dim/2)
        freqs_cis_pos = self.freqs_cis[:seq_len]

        # Reshape freqs_cis for broadcasting: (1, 1, seq_len, head_dim/2)
        freqs_cis_reshaped = freqs_cis_pos.unsqueeze(0).unsqueeze(0)

        # Apply rotation: q'_c = q_c * freqs_cis_pos
        xq_out_c = xq_c * freqs_cis_reshaped
        xk_out_c = xk_c * freqs_cis_reshaped

        # Convert back to real and reshape: (bsz, num_heads, seq_len, head_dim)
        xq_out = torch.view_as_real(xq_out_c).flatten(3)
        xk_out = torch.view_as_real(xk_out_c).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)

logger("RotaryEmbedding defined.")


# DemoAttention 是一个实现了多头自注意力机制的模块，支持 Llama/Transformer-variant 架构中的分组 KV 头（Group Query Attention）和 Rotary Position Embedding（RoPE）位置编码。
# 
# 主要功能和结构如下：
# 1. 初始化时，DemoAttention 根据配置参数（DemoLLMConfig）设置隐藏层维度、注意力头数、每个头的维度等，并构建了 Q/K/V 的线性投影层（q_proj, k_proj, v_proj）和输出投影层（o_proj）。
# 2. 支持分组 KV 头（num_kv_heads < num_q_heads），即多个 Query 头共享同一组 Key/Value 头，通过 _repeat_kv 方法实现 KV 头的复制扩展。
# 3. 集成了 RotaryEmbedding（RoPE）模块，对 Q/K 向量进行旋转式位置编码，提升模型对长序列的泛化能力。
# 4. forward 方法中，输入 hidden_states 首先通过线性层分别得到 Query、Key、Value，形状调整为 (batch, num_heads, seq_len, head_dim)。
# 5. 支持 past_key_value（缓存机制）和 use_cache，用于高效的自回归推理。
# 6. 支持 Flash Attention（如果可用），否则回退到标准的 scaled dot-product attention。
# 7. 最终输出为 (batch, seq_len, hidden_size)，并可返回新的 key/value 缓存。
# 
# DemoAttention 适用于高效、可扩展的 Transformer/Llama 类模型，兼容主流的推理和训练场景。

class DemoAttention(nn.Module):
    def __init__(self, config: DemoLLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_q_heads // self.num_kv_heads # Num Q heads per KV head
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_q_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            theta=config.rope_theta,
            device=DEVICE # Initialize on target device
        )
        self.flash_available = hasattr(F, 'scaled_dot_product_attention') and config.flash_attn

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        bs, num_kv_heads, slen, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, None, :, :]
            .expand(bs, num_kv_heads, n_rep, slen, head_dim)
            .reshape(bs, num_kv_heads * n_rep, slen, head_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor, # (bsz, q_len, hidden_size)
        attention_mask: Optional[torch.Tensor] = None, # (bsz, 1, q_len, kv_len) for additive mask
        position_ids: Optional[torch.LongTensor] = None, # (bsz, q_len) -> Not directly used if RoPE applied based on seq_len
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_q_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # query_states: (bsz, num_q_heads, q_len, head_dim)
        # key_states/value_states: (bsz, num_kv_heads, q_len, head_dim)

        kv_seq_len = q_len # Initially, before considering past_key_value
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2] # Add past length

        # Apply RoPE based on current q_len (for new tokens)
        # The RotaryEmbedding's forward method expects current seq_len of Q and K
        cos, sin = None, None # Not passing these directly, RoPE is self-contained
        query_states, key_states = self.rotary_emb(query_states, key_states, seq_len=q_len) # RoPE applied to current q_len

        if past_key_value is not None:
            # key_states/value_states are for current q_len
            # past_key_value[0] is (bsz, num_kv_heads, past_seq_len, head_dim)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            current_key_value = (key_states, value_states)
        else:
            current_key_value = None

        # Grouped Query Attention: Repeat K and V heads to match Q heads
        key_states = self._repeat_kv(key_states, self.num_kv_groups)
        value_states = self._repeat_kv(value_states, self.num_kv_groups)
        # Now key_states/value_states are (bsz, num_q_heads, kv_seq_len, head_dim)

        attn_output = None
        # Check for Flash Attention compatibility
        # Flash Attn is_causal works when q_len == kv_seq_len for the attention computation itself.
        # If past_kv is used, q_len for query_states is for new tokens, kv_seq_len for key_states is total length.
        # Flash Attn handles this by taking full K/V and only new Qs.
        # The `is_causal` flag in F.sdpa handles masking correctly for decoder style models.
        # The main condition for Flash Attn is no explicit additive attention_mask.
        can_use_flash = self.flash_available and attention_mask is None
        if can_use_flash:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=None, # Causal mask handled by is_causal
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal= (q_len == kv_seq_len) # Only truly causal if no KV cache or if generating first token
                                                # If q_len < kv_seq_len (due to KV cache), is_causal should be False
                                                # and an explicit mask would be needed for padding if any.
                                                # For simplicity in decoder generation where new_q_len = 1, is_causal=False is fine.
                                                # And for training where q_len = kv_seq_len, is_causal=True.
                                                # Let's make it always causal for decoder, assuming no padding mask for flash path
            )
        else:
            # Manual attention with causal mask
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if kv_seq_len > 0: # Avoid mask creation for empty sequences
                # Causal mask (triangle mask)
                # query_states: (bsz, num_q_heads, q_len, head_dim)
                # key_states:   (bsz, num_q_heads, kv_seq_len, head_dim)
                # attn_weights: (bsz, num_q_heads, q_len, kv_seq_len)
                mask = torch.full((q_len, kv_seq_len), float("-inf"), device=query_states.device)
                # For causal, target token j can only attend to source tokens i <= j + (kv_seq_len - q_len)
                # where (kv_seq_len - q_len) is the length of the past context.
                # If q_len == kv_seq_len (no cache), it's a standard upper triangle.
                # If q_len == 1 (generation with cache), it attends to all kv_seq_len.
                causal_shift = kv_seq_len - q_len
                mask = torch.triu(mask, diagonal=1 + causal_shift) # Corrected causal mask
                attn_weights = attn_weights + mask[None, None, :, :] # Add to scores

            if attention_mask is not None: # Additive padding mask
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(query_states)
            if self.config.dropout > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.config.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, current_key_value
logger("DemoAttention defined.")



# DemoFeedForward 是一个前馈神经网络（Feed Forward Network, FFN）模块，常用于Transformer结构中。
# 其核心思想是：输入经过两个线性变换（gate_proj和up_proj），分别做激活和线性变换后相乘（即SwiGLU结构），
# 再通过down_proj线性变换回原始维度，最后加上dropout防止过拟合。
# 具体流程如下：
# 1. gate_proj：对输入做线性变换，输出intermediate_size维度。
# 2. up_proj：对输入做另一个线性变换，输出intermediate_size维度。
# 3. act_fn：对gate_proj的输出做激活（如SiLU）。
# 4. 两者相乘（SwiGLU结构），再通过down_proj映射回hidden_size维度。
# 5. 最后加dropout。
# 这种结构能提升模型表达能力和非线性，常见于现代大语言模型（如LLaMA、GPT-3等）。
class DemoFeedForward(nn.Module):
    def __init__(self, config: DemoLLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act] # e.g., SiLU
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # This is the SwiGLU formulation: FFN_SwiGLU(x, W, V, W2) = (Swish_1(xW) * xV)W2
        # Swish_1(x) = x * sigmoid(beta*x), where beta is often 1 (SiLU)
        # Here, gate_proj is W, up_proj is V, down_proj is W2
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))
logger("DemoFeedForward defined.")


# DemoRMSNorm 是一种归一化层（Root Mean Square LayerNorm），常用于大语言模型（如LLaMA、GPT-3等）中替代传统的LayerNorm。
# 它的核心思想是：对每个样本的最后一个维度（通常是特征维度）计算均方根（RMS），然后用输入除以RMS（加上一个很小的epsilon防止除零），
# 最后乘以一个可学习的缩放参数weight。这样可以保证归一化的稳定性，同时减少参数量和计算量。
# 具体流程如下：
# 1. 计算输入x在最后一个维度上的均方根：sqrt(mean(x^2) + eps)
# 2. 用输入x除以这个均方根，实现归一化。
# 3. 乘以可学习的weight参数，实现缩放。
# 4. 为了数值稳定性，归一化计算在float32精度下进行，最后再转回原始数据类型。
# 这种归一化方式在大模型中表现良好，能提升训练稳定性和泛化能力。

# 归一化（Normalization）是一种对输入数据进行标准化处理的技术，常用于深度学习模型中。
# 其主要作用是将输入数据的分布调整到一个更适合模型训练的范围，通常是让均值为0、方差为1，或者像RMSNorm那样让每个样本的特征向量模长保持稳定。
# 这样做的好处有：
# 1. 提高训练的稳定性，防止梯度消失或爆炸；
# 2. 加快模型收敛速度；
# 3. 有助于提升模型的泛化能力。
# 在大语言模型（如LLaMA、GPT-3等）中，常用的归一化方式有LayerNorm和RMSNorm
class DemoRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output_dtype = x.dtype
        x = x.to(torch.float32) # Calculate in float32 for stability
        output = self._norm(x)
        return (output * self.weight).to(output_dtype)
logger("DemoRMSNorm defined.")


# DemoTransformerBlock 是一个典型的 Transformer 解码器层（block），它包含了自注意力（Self-Attention）、前馈神经网络（Feed Forward, MLP）、以及归一化（RMSNorm）等模块。
# 其主要结构和前向传播流程如下：
# 1. 输入首先经过 input_layernorm（RMSNorm 归一化），提升数值稳定性；
# 2. 归一化后的结果送入自注意力层（DemoAttention），实现序列内信息的交互和建模；
# 3. 自注意力输出与原始输入做残差连接（residual），增强梯度流动和模型表达能力；
# 4. 残差结果再经过 post_attention_layernorm（第二次归一化）；
# 5. 归一化后送入前馈神经网络（DemoFeedForward），提升非线性表达能力；
# 6. 前馈输出与前一残差结果再次做残差连接；
# 7. 最终输出新的隐藏状态和可选的 KV 缓存（用于加速推理）。
# 这种结构是现代大语言模型（如 GPT、LLaMA 等）的基本构建单元，能够有效捕捉序列中的复杂依赖关系。
class DemoTransformerBlock(nn.Module):
    def __init__(self, config: DemoLLMConfig):
        super().__init__()
        self.self_attn = DemoAttention(config)
        self.mlp = DemoFeedForward(config)
        self.input_layernorm = DemoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DemoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # Passed to attention for RoPE
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)

        attn_outputs, present_key_value = self.self_attn(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids, # RoPE handled inside DemoAttention using its internal RotaryEmbedding
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = residual + attn_outputs

        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        feed_forward_hidden_states = self.mlp(normed_hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states, present_key_value
logger("DemoTransformerBlock defined.")


# DemoLLMModel 是一个基于 Transformer 架构的自回归大语言模型（LLM）主干类，继承自 HuggingFace 的 PreTrainedModel，便于与 Transformers 生态集成和保存/加载权重。
# 其核心结构和功能如下：
# 1. 初始化阶段（__init__）：
#    - 读取配置（config），设置词表大小、padding id 等基础参数。
#    - 构建嵌入层（embed_tokens）：将输入的 token id 映射为高维向量。
#    - 堆叠多个 DemoTransformerBlock 形成深层 Transformer 主体（self.layers）。
#    - 添加最终的归一化层（self.norm）和 dropout 层（self.dropout），提升模型泛化能力。
#    - 支持梯度检查点（gradient_checkpointing），便于大模型训练时节省显存。
# 2. 前向传播（forward）：
#    - 支持输入 input_ids 或直接输入 embedding（inputs_embeds），二者只能选其一。
#    - 自动处理 past_key_values（KV缓存），用于推理加速和自回归生成。
#    - 自动生成 position_ids（位置编码），支持增量推理。
#    - 经过嵌入、层堆叠、归一化等流程，输出隐藏状态（hidden_states）和可选的 KV 缓存。
#    - 兼容 HuggingFace 的输出格式（CausalLMOutputWithPast），便于下游任务和推理。
# 3. 该模型为自回归语言建模（Causal Language Modeling）设计，适用于文本生成、对话、补全等任务。


class DemoLLMModel(PreTrainedModel):
    config_class = DemoLLMConfig

    def __init__(self, config: DemoLLMConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([DemoTransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = DemoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(config.dropout) # Added dropout after embeddings
        self.gradient_checkpointing = False # For simplicity

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None, # (bsz, seq_len)
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None, # Not implemented
        output_hidden_states: Optional[bool] = None, # Not implemented
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        CausalLMOutputWithPast 是 HuggingFace Transformers 库中用于自回归语言建模（Causal Language Modeling）模型的标准输出结构。
        它通常包含如下字段：
            - logits（可选）：模型输出的预测分数（未归一化的概率），用于下游生成。
            - past_key_values：KV缓存，用于加速自回归推理。
            - hidden_states（可选）：每层的隐藏状态（如果需要输出）。
            - attentions（可选）：每层的注意力权重（如果需要输出）。
        该结构便于与 Transformers 生态的下游任务和推理流程集成。
        """

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2] # (bsz, num_kv_heads, seq_len, head_dim)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = self.dropout(inputs_embeds)

        # Create attention mask for padding (if any) and causality
        # For a decoder, the mask should be causal and also respect padding tokens.
        # The shape for additive mask in Attention is (bsz, 1, q_len, kv_len)
        # `attention_mask` from input is usually (bsz, seq_len)
        _expanded_mask = None
        if attention_mask is not None:
            # Expand padding mask: (bsz, seq_len) -> (bsz, 1, q_len, kv_len_with_past)
            # This can get tricky with KV caching. For this simplified version,
            # we assume attention_mask applies to current inputs.
            # Causal part is handled in DemoAttention.
            # An additive mask for padding would be:
            expanded_padding_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, seq_length + past_key_values_length)
            _expanded_mask = torch.zeros_like(expanded_padding_mask, dtype=hidden_states.dtype)
            _expanded_mask.masked_fill_(expanded_padding_mask == 0, float("-inf"))

        next_decoder_cache = [] if use_cache else None

        for i, decoder_layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=_expanded_mask, # Pass the combined mask
                position_ids=position_ids, # RoPE will use this implicitly or via seq_len
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache.append(layer_outputs[1])

        hidden_states = self.norm(hidden_states)

        # This model doesn't have MoE, so no aux_loss
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=None,
            attentions=None,
        )

logger("DemoLLMModel defined.")


# DemoLLMForCausalLM 是一个基于 DemoLLMModel 的自回归语言建模（Causal Language Modeling, CLM）模型封装，继承自 HuggingFace 的 PreTrainedModel。
# 它的主要作用是将底层的 DemoLLMModel（Transformer 主体）与输出层（lm_head）组合起来，实现从输入 token 预测下一个 token 的能力。
# 主要功能和结构说明如下：
# 1. 构造函数 __init__：初始化底层的 DemoLLMModel 和输出层 lm_head（一个线性层，将隐藏状态投影到词表大小），并调用 post_init() 完成权重初始化。
# 2. get_input_embeddings/set_input_embeddings：获取或设置输入嵌入层（通常用于词表扩展等场景）。
# 3. get_output_embeddings/set_output_embeddings：获取或设置输出层（lm_head），便于微调或词表扩展。
# 4. prepare_inputs_for_generation：为生成任务（如推理、采样）准备输入，包括处理 past_key_values（KV缓存）、动态生成 position_ids 等，兼容 HuggingFace 的生成 API。
# 5. forward：前向传播方法，输入 token ids、attention mask、position ids、past_key_values 等，输出 logits（用于下一个 token 的概率分布）和缓存。
# 该类可直接用于自回归文本生成任务，支持 HuggingFace 的 generate() 方法，适合微调和推理场景。

class DemoLLMForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = DemoLLMConfig

    def __init__(self, config: DemoLLMConfig):
        super().__init__(config)
        self.model = DemoLLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Weight tying is a common practice but optional
        # self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init() # Initialize weights

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if past_key_values:
            input_ids = input_ids[:, -1:] # Only take the last token if past_key_values is not None

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # Internal call to base model should always return dict
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss() # Default ignore_index is -100, set if needed
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure labels are on the same device as logits for loss calculation
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + (outputs.past_key_values if use_cache else tuple()) # Keep it simple
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
logger("DemoLLMForCausalLM defined.")



if tokenizer: # Ensure tokenizer is loaded before creating config dependent on its vocab size
    demo_llm_config = DemoLLMConfig(vocab_size=DEMO_VOCAB_SIZE_FINAL) # Use the final vocab size
    demo_llm_instance = DemoLLMForCausalLM(demo_llm_config).to(DEVICE)
    print_model_summary(demo_llm_instance, "Initial DemoLLM Instance")
    del demo_llm_instance # Clean up
else:
    logger("Skipping model verification as tokenizer was not loaded.")