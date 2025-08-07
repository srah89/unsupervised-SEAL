# knowledge-incorporation/src/EM/train_SFT.py
"""
SFT trainer

Dataset format expected:
{"prompt": "...", "completion": "..."}
"""
import os
import argparse
import datetime
from datasets import load_dataset
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--logging_steps", type=int, default=10)
    
    # Wandb configuration
    p.add_argument("--wandb_project", default="yoruba-knowledge-incorporation")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_tags", nargs="*", default=[])
    p.add_argument("--disable_wandb", action="store_true", help="Disable wandb logging")
    
    return p.parse_args()

def longest_seq_len(dataset, tok):
    return max(
        len(tok(example["prompt"] + example["completion"]).input_ids)
        for example in dataset
    )

def main() -> None:
    args = parse_args()

    # Initialize wandb
    if not args.disable_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or f"sft_{datetime.datetime.now().strftime('%m%d_%H%M%S')}",
                tags=args.wandb_tags,
                config={
                    "model_name": args.model_name_or_path,
                    "train_file": args.train_file,
                    "output_dir": args.output_dir,
                    "per_device_batch_size": args.per_device_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "num_train_epochs": args.num_train_epochs,
                    "learning_rate": args.learning_rate,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_target_modules": args.lora_target_modules,
                    "logging_steps": args.logging_steps,
                }
            )
            print(f"Wandb initialized: {wandb.run.name}")
        except ImportError:
            print("Warning: wandb not installed. Training without logging.")
            args.disable_wandb = True

    dataset = load_dataset("json", data_files=args.train_file, split="train")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Log dataset info
    if not args.disable_wandb:
        try:
            import wandb
            wandb.log({
                "dataset_size": len(dataset),
                "max_sequence_length": longest_seq_len(dataset, tokenizer),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            })
        except:
            pass

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        max_length=longest_seq_len(dataset, tokenizer),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Enable wandb logging in trainer
        report_to=["wandb"] if not args.disable_wandb else [],
    )

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(","),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    trainer.train()
    peft_model = trainer.model
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Log final training metrics
    if not args.disable_wandb:
        try:
            import wandb
            wandb.log({
                "training_completed": True,
                "final_loss": trainer.state.log_history[-1]["loss"] if trainer.state.log_history else None,
                "total_steps": trainer.state.global_step,
                "total_epochs": trainer.state.epoch,
            })
            wandb.finish()
        except:
            pass
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
