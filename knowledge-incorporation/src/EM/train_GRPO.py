# knowledge-incorporation/src/EM/train_GRPO.py
"""
GRPO (Group Relative Policy Optimization) trainer

Learns from relative preferences within groups of completions.
Uses preference learning instead of absolute rewards.

Dataset format expected:
{"prompt": "...", "completion": "...", "group_id": 0, "rank": 0}
"""
import os
import argparse
from datasets import load_dataset
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from pathlib import Path
import json
from typing import List, Dict, Any
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--group_size", type=int, default=3, help="Number of completions per group")
    p.add_argument("--preference_temperature", type=float, default=0.1, help="Temperature for preference learning")
    return p.parse_args()

def create_preference_pairs(dataset, group_size=3):
    """
    Create preference pairs from grouped completions.
    Each group has completions ranked by composite reward.
    Creates pairs: (better_completion, worse_completion)
    """
    preference_pairs = []
    
    # Group by prompt
    groups = {}
    for example in dataset:
        prompt = example["prompt"]
        if prompt not in groups:
            groups[prompt] = []
        groups[prompt].append(example)
    
    # Create preference pairs within each group
    for prompt, completions in groups.items():
        if len(completions) < 2:
            continue
            
        # Sort by composite reward (assuming it's in the data)
        sorted_completions = sorted(
            completions, 
            key=lambda x: x.get("composite_reward", x.get("adapter_mean", 0)),
            reverse=True
        )
        
        # Create preference pairs (better > worse)
        for i in range(len(sorted_completions) - 1):
            for j in range(i + 1, len(sorted_completions)):
                better = sorted_completions[i]
                worse = sorted_completions[j]
                
                preference_pairs.append({
                    "prompt": prompt,
                    "chosen": better["completion"],
                    "rejected": worse["completion"],
                    "chosen_reward": better.get("composite_reward", better.get("adapter_mean", 0)),
                    "rejected_reward": worse.get("composite_reward", worse.get("adapter_mean", 0)),
                })
    
    return preference_pairs

def compute_preference_loss(logits_chosen, logits_rejected, temperature=0.1):
    """
    Compute preference loss using Bradley-Terry model.
    Loss = -log(sigmoid((logits_chosen - logits_rejected) / temperature))
    """
    logits_diff = logits_chosen - logits_rejected
    loss = -torch.log(torch.sigmoid(logits_diff / temperature))
    return loss.mean()

class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.
    Learns from relative preferences within groups.
    """
    def __init__(self, model, tokenizer, args):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        
    def train_step(self, batch):
        """
        Single training step with preference learning.
        """
        self.model.train()
        
        # Tokenize chosen and rejected completions
        chosen_inputs = self.tokenizer(
            [f"{batch['prompt']}{batch['chosen']}" for batch in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        rejected_inputs = self.tokenizer(
            [f"{batch['prompt']}{batch['rejected']}" for batch in batch],
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Forward pass
        with torch.no_grad():
            chosen_logits = self.model(**chosen_inputs).logits
            rejected_logits = self.model(**rejected_inputs).logits
        
        # Compute preference loss
        loss = compute_preference_loss(
            chosen_logits.mean(dim=-1), 
            rejected_logits.mean(dim=-1),
            self.args.preference_temperature
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {"loss": loss.item()}

def main() -> None:
    args = parse_args()

    # Load dataset
    dataset = load_dataset("json", data_files=args.train_file, split="train")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Add LoRA
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
    )
    
    model = get_peft_model(base_model, lora_cfg)

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    # Create preference pairs from dataset
    print("Creating preference pairs...")
    preference_pairs = create_preference_pairs(dataset, args.group_size)
    print(f"Created {len(preference_pairs)} preference pairs")

    # Initialize GRPO trainer
    grpo_trainer = GRPOTrainer(model, tokenizer, args)

    # Train with preference learning
    print("Starting GRPO training...")
    for epoch in range(args.num_train_epochs):
        print(f"GRPO Epoch {epoch + 1}/{args.num_train_epochs}")
        
        # Shuffle preference pairs
        np.random.shuffle(preference_pairs)
        
        # Train on batches
        for i in range(0, len(preference_pairs), args.per_device_batch_size):
            batch = preference_pairs[i:i + args.per_device_batch_size]
            
            if len(batch) < 2:  # Need at least 2 for preference learning
                continue
                
            stats = grpo_trainer.train_step(batch)
            
            if i % args.logging_steps == 0:
                print(f"  Step {i}, Loss: {stats['loss']:.4f}")

    # Save model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    if dist.is_initialized():
        dist.destroy_process_group()

    print(f"GRPO training completed. Model saved to {output_path}")


if __name__ == "__main__":
    main() 