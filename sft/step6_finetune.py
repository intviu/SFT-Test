
from contextlib import nullcontext
import json
import os
from step4_llm import DemoLLMConfig, DemoLLMForCausalLM
from step5_pretraining import final_pretrained_model_path
from step3_auto_tokenizer import tokenizer,DEMO_VOCAB_SIZE_FINAL,DemoCorpusDataset
from step1_init import DEMO_HIDDEN_SIZE,DEMO_INTERMEDIATE_SIZE,DEMO_NUM_LAYERS,DEMO_NUM_ATTENTION_HEADS,DEMO_NUM_KV_HEADS,DEMO_MAX_SEQ_LEN, DEMO_SFT_EPOCHS, DEMO_SFT_LR, DEVICE, DTYPE_STR, NOTEBOOK_OUT_DIR, PTDTYPE, get_lr, logger, print_model_summary,DEMO_BATCH_SIZE,pretrain_file_path,DEMO_PRETRAIN_EPOCHS,DEMO_PRETRAIN_LR,SPECIAL_TOKENS_LIST, reasoning_file_path, sft_file_path

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch
from torch import optim


# DemoChatDataset 解释（中文）：
# DemoChatDataset 是一个用于SFT（监督微调）和推理任务的数据集类，继承自 PyTorch 的 Dataset。
# 主要功能是加载经过对话模板处理的聊天数据，并为每个样本生成相应的 loss mask（只对 assistant 回复部分计算损失）。
# 其核心流程如下：
# 1. 初始化时，读取指定文件（如 sft_data.jsonl），将每条对话的 conversations 字段加载为样本。
# 2. __getitem__ 方法中，利用 tokenizer 的 apply_chat_template 方法，将整个对话序列化为模型输入的 token id 序列（input_ids）。
# 3. 构建 loss_mask，原则是：只有 assistant 角色的 token 位置才参与损失计算，user 角色的 token 位置不计入损失。
# 4. 为了定位 assistant token 区间，代码会遍历 token id，结合特殊 role token（如 <|im_start|>assistant）和 <|im_end|>，标记出 assistant 回复的 token 区间。
# 5. 返回 input_ids 及对应的 loss_mask，供模型训练时使用。
# 该类适用于基于多轮对话的 SFT 微调场景，能够灵活适配不同的对话模板和分词器。
class DemoChatDataset(Dataset):
    """Dataset for SFT and Reasoning. Uses chat templates and masks non-assistant tokens."""
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        logger(f"Loading chat data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip())['conversations'])
        logger(f"Loaded {len(self.samples)} chat samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        conversations = self.samples[idx]

        # Tokenize the full conversation using the chat template
        # add_generation_prompt=False because the assistant's full response is in the data
        input_ids = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        ).squeeze(0)

        # Create loss mask: only train on assistant's tokens
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)

        # This is a simplified approach to find assistant tokens.
        # A robust solution would parse based on special role tokens during/after tokenization.
        # For this demo, we'll iterate turns and mark assistant tokens if the template is consistent.
        # The tokenizer.apply_chat_template output helps, but identifying exact assistant spans requires care.

        # Re-tokenize each part to find spans. This is not most efficient but illustrative.
        current_token_idx = 0
        is_assistant_turn = False
        assistant_start_token_id = self.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[-1] # often includes role token
        user_start_token_id = self.tokenizer.encode("<|im_start|>user", add_special_tokens=False)[-1]
        # im_end_token_id = self.tokenizer.eos_token_id # or specific <|im_end|> id
        im_end_token_id = self.tokenizer.token_to_id("<|im_end|>")

        # More reliable: iterate token IDs to find assistant segments
        assistant_start_seq = self.tokenizer.encode(self.tokenizer.apply_chat_template([{'role':'assistant', 'content':''}], add_generation_prompt=True, tokenize=False).replace('\n',''), add_special_tokens=False)[:-1] # remove placeholder token
        # The above is a bit hacky. The MiniMind SFTDataset uses direct token ID matching for robust mask generation.
        # For this demo, we'll assume any token *after* an assistant prompt and *before* the next user prompt or EOS is trainable.
        # This logic from the original dataset.py is more robust:
        bos_assistant_ids = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        eos_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)

        i = 0
        input_ids_list = input_ids.tolist()
        while i < len(input_ids_list):
            # Check for assistant start sequence
            if i + len(bos_assistant_ids) <= len(input_ids_list) and \
               input_ids_list[i : i + len(bos_assistant_ids)] == bos_assistant_ids:
                # Found assistant start
                start_of_response = i + len(bos_assistant_ids)
                # Find corresponding EOS
                end_of_response_marker = -1
                j = start_of_response
                while j < len(input_ids_list):
                    if j + len(eos_ids) <= len(input_ids_list) and \
                       input_ids_list[j : j + len(eos_ids)] == eos_ids:
                        end_of_response_marker = j
                        break
                    j += 1

                if end_of_response_marker != -1:
                    # Mark tokens from start of response up to and including the EOS for loss
                    loss_mask[start_of_response : end_of_response_marker + len(eos_ids)] = 1
                    i = end_of_response_marker + len(eos_ids) # Move past this assistant block
                    continue
                else: # No EOS found, mask till end (might be truncated)
                    loss_mask[start_of_response:] = 1
                    break
            i += 1

        loss_mask[input_ids == self.tokenizer.pad_token_id] = 0 # Don't learn on padding

        X = input_ids[:-1]
        Y = input_ids[1:]
        mask_for_loss_calculation = loss_mask[1:]

        return X, Y, mask_for_loss_calculation

logger("Testing DemoChatDataset for SFT...")



try:
    test_sft_ds = DemoChatDataset(sft_file_path, tokenizer, DEMO_MAX_SEQ_LEN)
    X_sft_sample, Y_sft_sample, mask_sft_sample = test_sft_ds[0]
    logger(f"Sample X (SFT): {X_sft_sample.shape}, {X_sft_sample[:20]}...")
    logger(f"Sample Y (SFT): {Y_sft_sample.shape}, {Y_sft_sample[:20]}...")
    logger(f"Sample Mask (SFT): {mask_sft_sample.shape}, {mask_sft_sample[:20]}...")
    full_sft_ids = torch.cat([X_sft_sample[:1], Y_sft_sample], dim=0)
    logger(f"Decoded SFT sample with mask applied (showing Y tokens where mask=1):\n{tokenizer.decode(Y_sft_sample[mask_sft_sample.bool()])}")
    logger(f"Full SFT sample decoded:\n{tokenizer.decode(full_sft_ids)}")
except Exception as e:
    logger(f"Error testing DemoChatDataset: {e}. Tokenizer or chat template might need adjustment.")


logger("Testing DemoChatDataset for Reasoning...")
try:
    test_reasoning_ds = DemoChatDataset(reasoning_file_path, tokenizer, DEMO_MAX_SEQ_LEN)
    X_rsn_sample, Y_rsn_sample, mask_rsn_sample = test_reasoning_ds[0]
    logger(f"Sample X (Reasoning): {X_rsn_sample.shape}, {X_rsn_sample[:30]}...")
    logger(f"Sample Y (Reasoning): {Y_rsn_sample.shape}, {Y_rsn_sample[:30]}...")
    logger(f"Sample Mask (Reasoning): {mask_rsn_sample.shape}, {mask_rsn_sample[:30]}...")
    full_rsn_ids = torch.cat([X_rsn_sample[:1], Y_rsn_sample], dim=0)
    logger(f"Decoded Reasoning sample with mask applied (showing Y tokens where mask=1):\n{tokenizer.decode(Y_rsn_sample[mask_rsn_sample.bool()])}")
    logger(f"Full Reasoning sample decoded:\n{tokenizer.decode(full_rsn_ids)}")
except Exception as e:
    logger(f"Error testing DemoChatDataset for Reasoning: {e}. Ensure tokenizer and chat template are correctly set.")

    


logger("Initializing model for SFT, loading pretrained weights...")
if tokenizer and final_pretrained_model_path and os.path.exists(final_pretrained_model_path):
    sft_config = DemoLLMConfig( # Re-use the same config as pretraining
        vocab_size=DEMO_VOCAB_SIZE_FINAL,
        hidden_size=DEMO_HIDDEN_SIZE,
        intermediate_size=DEMO_INTERMEDIATE_SIZE,
        num_hidden_layers=DEMO_NUM_LAYERS,
        num_attention_heads=DEMO_NUM_ATTENTION_HEADS,
        num_key_value_heads=DEMO_NUM_KV_HEADS,
        max_position_embeddings=DEMO_MAX_SEQ_LEN,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    sft_model_demo = DemoLLMForCausalLM(sft_config).to(DEVICE)
    sft_model_demo.load_state_dict(torch.load(final_pretrained_model_path, map_location=DEVICE))
    print_model_summary(sft_model_demo, "Demo SFT Model (Loaded Pretrained)")
else:
    logger("Cannot initialize SFT model: Pretrained model path or tokenizer is invalid.")
    sft_model_demo = None


if tokenizer and sft_model_demo:
    demo_sft_dataset = DemoChatDataset(sft_file_path, tokenizer, max_length=DEMO_MAX_SEQ_LEN)
    demo_sft_dataloader = DataLoader(demo_sft_dataset, batch_size=DEMO_BATCH_SIZE, shuffle=True, num_workers=0)
    logger(f"Demo SFT dataset size: {len(demo_sft_dataset)}")

    logger("Verifying SFT data sample and mask from DataLoader:")
    for x_s_dl, y_s_dl, m_s_dl in demo_sft_dataloader:
        # Reconstruct full sequence for one sample to verify mask logic
        idx_to_check = 0
        # The first token of X is often BOS. The actual sequence starts from the second token of input_ids.
        # So, to reconstruct from X, Y: use X's BOS, then Y.
        # Original input_ids was tokenized_chat. X = input_ids[:-1], Y = input_ids[1:]
        # So, input_ids = torch.cat([X_s_dl[idx_to_check, :1], Y_s_dl[idx_to_check]], dim=0)
        # However, the tokenizer already added BOS if specified by template. Let's just decode Y with its mask.
        logger(f"  Tokens from Y for sample {idx_to_check} where mask is 1: {tokenizer.decode(y_s_dl[idx_to_check][m_s_dl[idx_to_check].bool()])}")
        break
else:
    logger("Skipping SFT Dataloader: SFT model or tokenizer not initialized.")
    demo_sft_dataloader = []


## 上面准备了半天的数据集 ，然后 推理过程一样，就是用不同的数据集又跑一遍
if sft_model_demo and demo_sft_dataloader:
    optimizer_sft_d = optim.AdamW(sft_model_demo.parameters(), lr=DEMO_SFT_LR)
    loss_fct_sft_d = nn.CrossEntropyLoss(reduction='none')

    autocast_ctx_sft = nullcontext() if DEVICE.type == 'cpu' else torch.amp.autocast(device_type=DEVICE.type, dtype=PTDTYPE)
    scaler_sft_d = torch.cuda.amp.GradScaler(enabled=(DTYPE_STR != 'float32' and DEVICE.type == 'cuda'))

    total_steps_sft_d = len(demo_sft_dataloader) * DEMO_SFT_EPOCHS
    logger(f"Starting DEMO SFT for {DEMO_SFT_EPOCHS} epochs ({total_steps_sft_d} steps)...")

    sft_model_demo.train()
    current_training_step_sft = 0
    for epoch in range(DEMO_SFT_EPOCHS):
        epoch_loss_sft_val = 0.0
        for step, (X_batch_sft, Y_batch_sft, mask_batch_sft) in enumerate(demo_sft_dataloader):
            X_batch_sft, Y_batch_sft, mask_batch_sft = X_batch_sft.to(DEVICE), Y_batch_sft.to(DEVICE), mask_batch_sft.to(DEVICE)

            current_lr_sft = get_lr(current_training_step_sft, total_steps_sft_d, DEMO_SFT_LR)
            for param_group in optimizer_sft_d.param_groups:
                param_group['lr'] = current_lr_sft

            with autocast_ctx_sft:
                outputs_sft_loop = sft_model_demo(input_ids=X_batch_sft)
                logits_sft_loop = outputs_sft_loop.logits # (bsz, seq_len-1, vocab_size)

                raw_loss_sft = loss_fct_sft_d(logits_sft_loop.view(-1, logits_sft_loop.size(-1)), Y_batch_sft.view(-1))
                # mask_batch_sft corresponds to Y_batch_sft
                masked_loss_sft = (raw_loss_sft * mask_batch_sft.view(-1)).sum() / mask_batch_sft.sum().clamp(min=1)

            scaler_sft_d.scale(masked_loss_sft).backward()
            scaler_sft_d.step(optimizer_sft_d)
            scaler_sft_d.update()
            optimizer_sft_d.zero_grad(set_to_none=True)

            epoch_loss_sft_val += masked_loss_sft.item()
            current_training_step_sft += 1

            if (step + 1) % 1 == 0:
                logger(f"SFT Epoch {epoch+1}, Step {step+1}/{len(demo_sft_dataloader)}, Loss: {masked_loss_sft.item():.4f}, LR: {current_lr_sft:.3e}")

        logger(f"End of SFT Epoch {epoch+1}, Avg Loss: {epoch_loss_sft_val / len(demo_sft_dataloader):.4f}")

    logger("DEMO SFT finished.")
    final_sft_model_path = os.path.join(NOTEBOOK_OUT_DIR, "demo_llm_sft.pth")
    torch.save(sft_model_demo.state_dict(), final_sft_model_path)
    logger(f"Demo SFT model saved to: {final_sft_model_path}")
else:
    logger("Skipping SFT loop as model or dataloader was not initialized.")
    final_sft_model_path = None



# 验证
if sft_model_demo and final_sft_model_path and os.path.exists(final_sft_model_path):
    logger("Testing SFT model chat capability...")
    sft_model_demo.eval()
    sft_test_chat_history = [{"role": "user", "content": "What is the capital of France?"}]
    sft_test_prompt = tokenizer.apply_chat_template(sft_test_chat_history, tokenize=False, add_generation_prompt=True)
    sft_test_inputs = tokenizer(sft_test_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad(), autocast_ctx_sft:
        sft_generated_outputs = sft_model_demo.generate(
            sft_test_inputs.input_ids,
            max_new_tokens=200,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    sft_decoded_response = tokenizer.decode(sft_generated_outputs[0][sft_test_inputs.input_ids.shape[1]:], skip_special_tokens=True)
    logger(f"SFT Prompt: '{sft_test_chat_history[0]['content']}' -> Generated: '{sft_decoded_response}'")
else:
    logger("Skipping SFT model test as it was not trained or saved.")

