# knowledge-incorporation/src/EM/train_PPO.py
"""
GRPO (Group Relative Policy Optimization) trainer using TRL's GRPOTrainer

Uses TRL's built-in GRPO implementation with trained reward model (default) or composite rewards from TTT server.
"""
import os
import argparse
from datasets import load_dataset
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
from pathlib import Path
import json
from typing import List, Dict, Any
import numpy as np

# Import TTT server reward functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inner'))
from TTT_server import compute_length_bonus, compute_diversity_bonus, compute_quality_bonus, compute_composite_reward

# Import reward model utilities from utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import create_reward_function

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
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    p.add_argument("--epsilon", type=float, default=0.2, help="Epsilon for clipping")
    p.add_argument("--scale_rewards", type=bool, default=True, help="Scale rewards by std")
    
    # Reward model options (now default)
    p.add_argument("--reward_model_path", type=str, default="knowledge-incorporation/models/reward_model", 
                   help="Path to trained reward model (default: knowledge-incorporation/models/reward_model)")
    p.add_argument("--use_composite_rewards", action="store_true", 
                   help="Use composite rewards from TTT server instead of reward model")
    
    return p.parse_args()

def composite_reward_function(completions, **kwargs):
    """
    Composite reward function that uses the exact same logic as TTT server.
    
    This function is called by TRL GRPO for each batch of generated completions.
    It uses the same reward computation as your TTT server.
    """
    rewards = []
    
    # Get prompt from kwargs if available
    prompt = kwargs.get('prompt', '')
    
    for completion in completions:
        # Use the exact same functions as TTT server
        length_bonus = compute_length_bonus(completion)
        diversity_bonus = compute_diversity_bonus(completion, [])  # Empty list for other texts
        quality_bonus = compute_quality_bonus(completion, prompt)
        
        # Get adapter_mean from kwargs (from TTT server evaluation)
        adapter_mean = kwargs.get('adapter_mean', 0.5)
        
        # Use the exact same composite reward function as TTT server
        composite_reward = compute_composite_reward(
            adapter_mean=adapter_mean,
            text=completion,
            other_texts=[],  # Empty for now
            prompt=prompt
        )
        
        rewards.append(composite_reward)
    
    return rewards

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

    # Choose reward function (reward model is now default)
    if args.use_composite_rewards:
        print("Using composite reward function from TTT server")
        reward_function = composite_reward_function
    else:
        # Check if reward model exists
        reward_model_path = Path(args.reward_model_path)
        if not reward_model_path.exists():
            print(f"Warning: Reward model not found at {args.reward_model_path}")
            print("Falling back to composite rewards. To use reward model, first train it with:")
            print("bash knowledge-incorporation/scripts/train_reward_model.sh")
            print("Using composite reward function from TTT server")
            reward_function = composite_reward_function
        else:
            print(f"Using trained reward model from: {args.reward_model_path}")
            reward_function = create_reward_function(args.reward_model_path)

    # GRPO configuration using TRL
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        beta=args.beta,
        epsilon=args.epsilon,
        max_completion_length=512,
        log_completions=True,
        num_completions_to_print=5,
    )

    # Initialize TRL's GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_function,
    )

    # Train with TRL's GRPO
    print("Starting GRPO training with TRL...")
    trainer.train()
    
    # Save model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 