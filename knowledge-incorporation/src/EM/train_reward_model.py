# knowledge-incorporation/src/EM/train_reward_model.py
"""
Simplified reward model trainer for SEAL project using TRL's RewardTrainer

This script:
1. Loads synthetic data from knowledge-incorporation/data/synthetic_data/
2. Generates preference pairs (in-context vs out-of-context answers)
3. Trains a reward model using TRL's RewardTrainer with Qwen 2.5-1.5B
4. Includes evaluation metrics
5. Saves the trained model for use in PPO training
"""
import json
import torch
import torch.nn as nn
import random
import argparse
import logging
import datetime
from pathlib import Path
from typing import List, Dict
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoConfig
)
from trl import RewardTrainer, RewardConfig
import numpy as np

# Remove the CustomRewardModel class since we're using AutoModelForSequenceClassification

class SimplePreferenceGenerator:
    """Simplified preference pair generator"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def is_valid_answer(self, answer: str, correct_answer: str) -> bool:
        """Simple validation for negative answers"""
        return (
            len(answer.strip()) >= 10 and 
            len(answer) <= 200 and
            answer.lower().strip() != correct_answer.lower().strip()
        )
    
    def generate_negative_answer(self, question: str) -> str:
        """Generate a simple negative answer"""
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                       skip_special_tokens=True).strip()
        return response
    
    def create_incomplete_answer(self, answer: str) -> str:
        """Create incomplete answer by cutting it short"""
        words = answer.split()
        if len(words) <= 3:
            return answer
        cut_point = len(words) // 2
        return " ".join(words[:cut_point]).strip()
    
    def generate_preference_pairs(self, synthetic_data: List[Dict], max_pairs: int = 1000) -> List[Dict]:
        """Generate preference pairs from synthetic data"""
        preference_data = []
        
        logging.info(f"Generating up to {max_pairs} preference pairs...")
        
        for item in synthetic_data:
            if len(preference_data) >= max_pairs:
                break
                
            title = item.get('title', '')
            context = item.get('context', '')
            questions = item.get('questions', [])
            
            for qa in questions:
                if len(preference_data) >= max_pairs:
                    break
                    
                question = qa.get('question', '')
                correct_answer = qa.get('answer', '')
                
                if not question or not correct_answer:
                    continue
                
                # Create input text
                input_text = f"Context: {context}\nQuestion: {question}\nAnswer: "
                
                # Generate negative answers
                negative_answers = []
                
                # 1. Generated negative answer
                try:
                    neg_answer = self.generate_negative_answer(question)
                    if self.is_valid_answer(neg_answer, correct_answer):
                        negative_answers.append(neg_answer)
                except:
                    pass
                
                # 2. Incomplete answer
                incomplete = self.create_incomplete_answer(correct_answer)
                if self.is_valid_answer(incomplete, correct_answer):
                    negative_answers.append(incomplete)
                
                # Create preference pairs
                for neg_answer in negative_answers:
                    preference_data.append({
                        "chosen": input_text + correct_answer,
                        "rejected": input_text + neg_answer
                    })
                    
                    if len(preference_data) >= max_pairs:
                        break
        
        logging.info(f"Generated {len(preference_data)} preference pairs")
        return preference_data

def load_synthetic_data(data_dir: str) -> List[Dict]:
    """Load synthetic data files"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    # Find JSON files
    json_files = []
    for pattern in ["synthetic_data/**/*.json", "**/*.json"]:
        json_files.extend(data_path.glob(pattern))
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    
    logging.info(f"Found {len(json_files)} JSON files")
    
    # Load all data
    all_data = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        except Exception as e:
            logging.warning(f"Could not load {file_path}: {e}")
    
    logging.info(f"Loaded {len(all_data)} total items")
    return all_data

def compute_reward_metrics(eval_results):
    """Compute custom metrics from eval results"""
    if not hasattr(eval_results, 'predictions') or eval_results.predictions is None:
        return {}
    
    # TRL RewardTrainer returns logits for chosen vs rejected
    logits = eval_results.predictions
    if len(logits.shape) == 2:
        # Assume first column is chosen, second is rejected
        chosen_rewards = logits[:, 0] if logits.shape[1] > 0 else logits.flatten()
        rejected_rewards = logits[:, 1] if logits.shape[1] > 1 else np.zeros_like(chosen_rewards)
    else:
        chosen_rewards = logits
        rejected_rewards = np.zeros_like(chosen_rewards)
    
    # Calculate metrics
    accuracy = np.mean(chosen_rewards > rejected_rewards) if len(rejected_rewards) > 0 else 0.0
    reward_diff = np.mean(chosen_rewards - rejected_rewards) if len(rejected_rewards) > 0 else 0.0
    
    return {
        "eval_accuracy": accuracy,
        "eval_reward_diff": reward_diff,
        "eval_chosen_mean": np.mean(chosen_rewards),
        "eval_rejected_mean": np.mean(rejected_rewards),
    }

def main():
    parser = argparse.ArgumentParser(description="Train reward model for SEAL project")
    parser.add_argument("--data_dir", default="knowledge-incorporation/data")
    parser.add_argument("--output_dir", default="models/reward_model")  # Changed to SEAL/models/reward_model
    parser.add_argument("--reward_model_name", default="bert-base-uncased")  # Excellent for classification tasks
    parser.add_argument("--generation_model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"reward_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting reward model training with args: {vars(args)}")
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load data
    logger.info("Loading synthetic data...")
    synthetic_data = load_synthetic_data(args.data_dir)
    
    # Generate preference pairs
    logger.info("Generating preference pairs...")
    generator = SimplePreferenceGenerator(args.generation_model_name)
    preference_data = generator.generate_preference_pairs(synthetic_data, args.num_samples)
    
    if not preference_data:
        logger.error("No preference pairs generated!")
        return
    
    # Create dataset
    dataset = Dataset.from_list(preference_data)
    
    # Split data
    split_dataset = dataset.train_test_split(test_size=0.15, seed=args.seed)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Initialize model and tokenizer (using AutoModelForSequenceClassification for TRL compatibility)
    logger.info(f"Loading reward model: {args.reward_model_name}")
    try:
        # Use full precision for BERT models to avoid FP16 gradient issues
        if args.reward_model_name.startswith("bert"):
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                args.reward_model_name,
                num_labels=1,  # Single scalar output for reward
                torch_dtype=torch.float32,  # Use full precision for BERT
                device_map="auto"
            )
        else:
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                args.reward_model_name,
                num_labels=1,  # Single scalar output for reward
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
        tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
        
        # Ensure BERT tokenizer has proper special tokens
        if args.reward_model_name.startswith("bert"):
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.sep_token is None:
                tokenizer.sep_token = "[SEP]"
            if tokenizer.cls_token is None:
                tokenizer.cls_token = "[CLS]"
        else:
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
        # Set pad_token_id in model config to match tokenizer
        reward_model.config.pad_token_id = tokenizer.pad_token_id
        
        # Validate model configuration
        if not hasattr(reward_model.config, 'num_labels') or reward_model.config.num_labels != 1:
            logger.warning(f"Model config shows {getattr(reward_model.config, 'num_labels', 'unknown')} labels, expected 1")
            reward_model.config.num_labels = 1
            
        logger.info(f"Successfully loaded {args.reward_model_name} with {reward_model.config.num_labels} output labels")
        
    except Exception as e:
        logger.error(f"Failed to load reward model {args.reward_model_name}: {e}")
        logger.error("Make sure the model supports sequence classification (not causal language modeling)")
        return
    
    # Training arguments (TRL 0.20.0 uses RewardConfig, not TrainingArguments)
    # Using BERT for reward modeling because:
    # 1. It's designed for sequence classification (unlike causal LMs)
    # 2. It has stable numerical outputs
    # 3. It's well-tested with TRL's RewardTrainer
    # 4. It's smaller and faster to train than large causal models
    training_args = RewardConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=2,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps", 
        save_steps=200,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=None,
        fp16=False if args.reward_model_name.startswith("bert") else True,  # Disable FP16 for BERT to avoid gradient issues
        dataloader_pin_memory=False,
        max_length=args.max_length,  # Required by TRL RewardTrainer
        disable_dropout=True,  # Required by TRL 0.20.0
    )
    
    # Initialize TRL RewardTrainer (TRL 0.20.0 format)
    trainer = RewardTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # TRL 0.20.0 uses 'processing_class'
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    # Compute custom metrics
    custom_metrics = compute_reward_metrics(eval_results)
    eval_results.update(custom_metrics)
    
    logger.info(f"Final metrics: {eval_results}")
    
    # Save model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Save training info
    with open(output_path / "training_info.json", "w") as f:
        json.dump({
            "args": vars(args),
            "final_metrics": eval_results,
            "num_preference_pairs": len(preference_data),
            "model_architecture": f"BERT-based reward model ({args.reward_model_name})"
        }, f, indent=2)
    
    logger.info(f"Training complete! Model saved to {output_path}")

if __name__ == "__main__":
    main()