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
    AutoModel,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoConfig
)
from trl import RewardTrainer
import numpy as np

class CustomRewardModel(nn.Module):
    """Custom reward model with Qwen 2.5-1.5B backbone"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Simple reward head
        self.reward_head = nn.Linear(self.config.hidden_size, 1)
        
        # Initialize reward head
        nn.init.normal_(self.reward_head.weight, std=0.02)
        nn.init.zeros_(self.reward_head.bias)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # Use last token representation (standard for reward models)
        sequence_lengths = attention_mask.sum(dim=1) - 1 if attention_mask is not None else input_ids.shape[1] - 1
        batch_size = input_ids.shape[0]
        
        # Get last non-padding token for each sequence
        last_hidden_states = outputs.last_hidden_state[range(batch_size), sequence_lengths]
        reward = self.reward_head(last_hidden_states)
        
        return reward.squeeze(-1)

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
    parser.add_argument("--output_dir", default="knowledge-incorporation/models/reward_model")
    parser.add_argument("--reward_model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
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
    
    # Initialize model and tokenizer
    logger.info(f"Loading reward model: {args.reward_model_name}")
    
    try:
        logger.info("Step 1: Loading AutoConfig...")
        config = AutoConfig.from_pretrained(args.reward_model_name)
        logger.info(f"✓ Config loaded successfully. Hidden size: {config.hidden_size}")
        
        logger.info("Step 2: Loading AutoModel...")
        transformer = AutoModel.from_pretrained(args.reward_model_name)
        logger.info(f"✓ Transformer loaded successfully. Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
        
        logger.info("Step 3: Creating CustomRewardModel...")
        reward_model = CustomRewardModel(args.reward_model_name)
        logger.info(f"✓ CustomRewardModel created successfully. Total parameters: {sum(p.numel() for p in reward_model.parameters()):,}")
        
        logger.info("Step 4: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
        logger.info(f"✓ Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("✓ Set pad_token to eos_token")
        
        logger.info("✓ All model components loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model components: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    # Training arguments (TRL 0.20.0 uses standard TrainingArguments)
    training_args = TrainingArguments(
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
        fp16=True,
        dataloader_pin_memory=False,
    )
    
    # Initialize TRL RewardTrainer (TRL 0.20.0 format)
    logger.info("Initializing TRL RewardTrainer...")
    
    try:
        logger.info("Step 1: Creating trainer instance...")
        trainer = RewardTrainer(
            model=reward_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # TRL 0.20.0 uses 'processing_class'
        )
        logger.info("✓ RewardTrainer created successfully!")
        
        logger.info("Step 2: Checking trainer attributes...")
        logger.info(f"  - Model device: {next(reward_model.parameters()).device}")
        logger.info(f"  - Training dataset size: {len(train_dataset)}")
        logger.info(f"  - Evaluation dataset size: {len(eval_dataset)}")
        logger.info(f"  - Training args: {vars(training_args)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize trainer: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    # Train
    logger.info("Starting training...")
    
    try:
        trainer.train()
        logger.info("✓ Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    # Evaluate
    logger.info("Running final evaluation...")
    
    try:
        eval_results = trainer.evaluate()
        logger.info("✓ Evaluation completed successfully!")
        
        # Compute custom metrics
        logger.info("Computing custom metrics...")
        custom_metrics = compute_reward_metrics(eval_results)
        eval_results.update(custom_metrics)
        
        logger.info(f"Final metrics: {eval_results}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    # Save model
    logger.info("Saving model and tokenizer...")
    
    try:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created output directory: {output_path}")
        
        logger.info("Saving reward model...")
        trainer.save_model(str(output_path))
        logger.info("✓ Reward model saved successfully!")
        
        logger.info("Saving tokenizer...")
        tokenizer.save_pretrained(str(output_path))
        logger.info("✓ Tokenizer saved successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    # Save training info
    with open(output_path / "training_info.json", "w") as f:
        json.dump({
            "args": vars(args),
            "final_metrics": eval_results,
            "num_preference_pairs": len(preference_data),
            "model_architecture": "CustomRewardModel with Qwen2.5-1.5B"
        }, f, indent=2)
    
    logger.info(f"Training complete! Model saved to {output_path}")

if __name__ == "__main__":
    main()