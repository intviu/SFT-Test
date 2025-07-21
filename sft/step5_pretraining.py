from contextlib import nullcontext
import os
from step1_init import DEMO_HIDDEN_SIZE,DEMO_INTERMEDIATE_SIZE,DEMO_NUM_LAYERS,DEMO_NUM_ATTENTION_HEADS,DEMO_NUM_KV_HEADS,DEMO_MAX_SEQ_LEN, DEVICE, DTYPE_STR, NOTEBOOK_OUT_DIR, PTDTYPE, get_lr, logger, print_model_summary,DEMO_BATCH_SIZE,pretrain_file_path,DEMO_PRETRAIN_EPOCHS,DEMO_PRETRAIN_LR,SPECIAL_TOKENS_LIST
from step3_auto_tokenizer import tokenizer,DEMO_VOCAB_SIZE_FINAL,DemoCorpusDataset
from step4_llm import DemoLLMConfig,DemoLLMForCausalLM

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch
from torch import optim

logger("Initializing model for Pretraining...")
if tokenizer: # Ensure tokenizer is loaded
    pt_config = DemoLLMConfig(
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
    pt_model = DemoLLMForCausalLM(pt_config).to(DEVICE)
    print_model_summary(pt_model, "Demo Pretrain Model")
else:
    logger("Cannot initialize pretrain model: tokenizer not available.")
    pt_model = None



if tokenizer and pt_model: # Proceed only if tokenizer and model are initialized
    demo_pt_dataset = DemoCorpusDataset(pretrain_file_path, tokenizer, max_length=DEMO_MAX_SEQ_LEN)
    demo_pt_dataloader = DataLoader(demo_pt_dataset, batch_size=DEMO_BATCH_SIZE, shuffle=True, num_workers=0)
    logger(f"Demo Pretrain dataset size: {len(demo_pt_dataset)}")
else:
    logger("Skipping pretrain dataloader: tokenizer or model not initialized.")
    demo_pt_dataloader = [] # Empty dataloader


# 1. 首先判断pt_model和demo_pt_dataloader是否已经准备好（即模型和数据加载器都已初始化）。
# 2. 初始化优化器（AdamW）和损失函数（交叉熵损失）。
# 3. 根据设备类型（CPU或GPU）设置混合精度训练的上下文环境，并初始化梯度缩放器（GradScaler）。
# 4. 计算总的训练步数，并打印预训练的相关信息。
# 5. 进入训练模式，循环进行指定轮数（epoch）的训练：
#    - 对每个batch，先将数据移动到指定设备。
#    - 动态调整学习率。
#    - 在混合精度上下文中前向传播，获得模型输出logits。
#    - 计算原始损失，并结合mask进行加权平均，得到最终的masked loss。
#    - 反向传播、梯度缩放、优化器步进和梯度清零。
#    - 记录和打印当前步的损失和学习率。
# 6. 每个epoch结束后，计算并打印平均损失。
# 7. 所有训练完成后，保存最终的预训练模型参数到指定路径。
if pt_model and demo_pt_dataloader: # Check if model and dataloader are ready
    optimizer_pt_demo = optim.AdamW(pt_model.parameters(), lr=DEMO_PRETRAIN_LR)
    loss_fct_pt_demo = nn.CrossEntropyLoss(reduction='none')

    # Mixed precision context for GPU
    autocast_ctx = nullcontext() if DEVICE.type == 'cpu' else torch.amp.autocast(device_type=DEVICE.type, dtype=PTDTYPE)
    scaler_pt_demo = torch.cuda.amp.GradScaler(enabled=(DTYPE_STR != 'float32' and DEVICE.type == 'cuda'))

    total_steps_pt_demo = len(demo_pt_dataloader) * DEMO_PRETRAIN_EPOCHS
    logger(f"Starting DEMO Pretraining for {DEMO_PRETRAIN_EPOCHS} epochs ({total_steps_pt_demo} steps)...")

    pt_model.train()
    current_training_step_pt = 0
    for epoch in range(DEMO_PRETRAIN_EPOCHS):
        epoch_loss_pt_val = 0.0
        for step, (X_batch, Y_batch, mask_batch) in enumerate(demo_pt_dataloader):
            X_batch, Y_batch, mask_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE), mask_batch.to(DEVICE)

            current_lr_pt = get_lr(current_training_step_pt, total_steps_pt_demo, DEMO_PRETRAIN_LR)
            for param_group in optimizer_pt_demo.param_groups:
                param_group['lr'] = current_lr_pt

            with autocast_ctx:
                # For our custom DemoLLMForCausalLM, if `labels` is not passed, it returns logits.
                # We need to compute loss manually using the mask.
                outputs_pt = pt_model(input_ids=X_batch)
                logits_pt = outputs_pt.logits # (bsz, seq_len-1, vocab_size)

                # logits_pt are for predicting Y_batch. mask_batch aligns with Y_batch.
                raw_loss_pt = loss_fct_pt_demo(logits_pt.view(-1, logits_pt.size(-1)), Y_batch.view(-1))
                masked_loss_pt = (raw_loss_pt * mask_batch.view(-1)).sum() / mask_batch.sum().clamp(min=1)


            scaler_pt_demo.scale(masked_loss_pt).backward() #这个反向传播的结果呢？ 
            scaler_pt_demo.step(optimizer_pt_demo)
            scaler_pt_demo.update()
            optimizer_pt_demo.zero_grad(set_to_none=True)

            epoch_loss_pt_val += masked_loss_pt.item()
            current_training_step_pt += 1

            if (step + 1) % 1 == 0: # Log frequently for demo
                logger(f"PT Epoch {epoch+1}, Step {step+1}/{len(demo_pt_dataloader)}, Loss: {masked_loss_pt.item():.4f}, LR: {current_lr_pt:.3e}")

        avg_epoch_loss_pt = epoch_loss_pt_val / len(demo_pt_dataloader)
        logger(f"End of PT Epoch {epoch+1}, Average Loss: {avg_epoch_loss_pt:.4f}")

    logger("DEMO Pretraining finished.")
    # Save the final pretrained model
    final_pretrained_model_path = os.path.join(NOTEBOOK_OUT_DIR, "demo_llm_pretrained.pth")
    torch.save(pt_model.state_dict(), final_pretrained_model_path)
    logger(f"Demo pretrained model saved to: {final_pretrained_model_path}")
else:
    logger("Skipping Pretraining loop as model or dataloader was not initialized.")
    final_pretrained_model_path = None



if pt_model and final_pretrained_model_path and os.path.exists(final_pretrained_model_path):
    logger("Testing pretrained model generation...")
    pt_model.eval() # Set to evaluation mode
    test_prompt_str_pt = "Language models learn"
    # Prepend BOS for generation consistency with training if tokenizer doesn't do it automatically
    pt_test_input_ids = tokenizer(tokenizer.bos_token + test_prompt_str_pt, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad(), autocast_ctx:
        generated_output_pt = pt_model.generate(
            pt_test_input_ids,
            max_new_tokens=15,
            do_sample=False, # Greedy for this test
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    decoded_generated_pt = tokenizer.decode(generated_output_pt[0], skip_special_tokens=True)
    logger(f"Prompt: '{test_prompt_str_pt}' -> Generated: '{decoded_generated_pt}'")
else:
    logger("Skipping pretrained model test as it was not trained or saved.")