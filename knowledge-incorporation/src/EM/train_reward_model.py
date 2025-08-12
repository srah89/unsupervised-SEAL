# knowledge-incorporation/src/EM/train_reward_model.py
"""
Reward model trainer for SEAL project using custom architecture

This script:
1. Loads synthetic data from knowledge-incorporation/data/synthetic_data/
2. Generates preference pairs (in-context vs out-of-context answers)
3. Trains a reward model using custom architecture with Qwen 2.5-1.5B
4. Includes comprehensive evaluation metrics
5. Saves the trained model for use in PPO training
"""
import json
import torch
import torch.nn as nn
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    AutoConfig
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class CustomRewardModel(nn.Module):
    """Custom reward model with Qwen 2.5-1.5B backbone"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__()
        # Load just the transformer backbone (no head)
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Custom reward head
        self.reward_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)  # Single scalar output
        )
        
        # Initialize reward head weights
        self._init_reward_head()
    
    def _init_reward_head(self):
        """Initialize reward head with small weights"""
        for module in self.reward_head:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Use mean pooling over sequence length (better than just last token)
        if attention_mask is not None:
            # Mask out padding tokens for mean pooling
            masked_embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            pooled = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)
        
        # Get reward score
        reward = self.reward_head(pooled)
        return reward.squeeze(-1)  # Shape: [batch_size]

class SEALPreferenceGenerator:
    """Generate preference pairs for SEAL reward model training"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logging.info(f"Generation model loaded: {model_name}")
    
    def is_valid_negative_answer(self, answer: str, correct_answer: str) -> bool:
        """Filter out obviously bad negative answers"""
        if len(answer.strip()) < 10:  # Too short
            return False
        if answer.lower().strip() == correct_answer.lower().strip():  # Same as correct
            return False
        if len(answer) > 200:  # Too long (likely hallucination)
            return False
        # Filter out generic non-answers
        generic_phrases = [
            "i don't know", "not sure", "unclear", "cannot determine", 
            "information not available", "not provided"
        ]
        if any(phrase in answer.lower() for phrase in generic_phrases):
            return False
        return True
    
    def generate_out_of_context_answer(self, question: str) -> str:
        """Generate answer without context (likely hallucinated)"""
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                       skip_special_tokens=True).strip()
        return response
    
    def create_incomplete_answer(self, answer: str) -> str:
        """Create an incomplete version of the answer"""
        words = answer.split()
        if len(words) <= 3:
            return answer
        
        # Take first 30-70% of the answer
        cut_point = random.randint(max(1, len(words)//3), (len(words)*7)//10)
        incomplete = " ".join(words[:cut_point])
        return incomplete.strip()
    
    def create_wrong_answer(self, context: str, question: str) -> str:
        """Create a wrong but plausible answer using random context spans"""
        sentences = context.split('.')
        if len(sentences) > 1:
            # Pick a random sentence that's not too short
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
            if valid_sentences:
                random_sentence = random.choice(valid_sentences)
                # Take a reasonable chunk
                words = random_sentence.split()
                if len(words) > 10:
                    start = random.randint(0, max(0, len(words)-10))
                    end = min(len(words), start + random.randint(5, 15))
                    return " ".join(words[start:end]).strip()
                return random_sentence.strip()
        
        # Fallback: return a generic wrong answer
        return "This information is not clearly stated in the provided context."
    
    def generate_preference_pairs(self, synthetic_data: List[Dict]) -> List[Dict]:
        """Generate preference pairs from synthetic data"""
        preference_data = []
        total_items = len(synthetic_data)
        total_questions = sum(len(item.get('questions', [])) for item in synthetic_data)
        
        logging.info(f"Starting preference pair generation for {total_items} items with {total_questions} total questions")
        
        processed_items = 0
        processed_questions = 0
        
        for item_idx, item in enumerate(synthetic_data):
            title = item.get('title', '')
            context = item.get('context', '')
            questions = item.get('questions', [])
            
            if not questions:
                continue
            
            # Log progress every 10 items or when starting a new item
            if item_idx % 10 == 0 or item_idx == 0:
                logging.info(f"Processing item {item_idx + 1}/{total_items}: '{title[:50]}...'")
            
            for q_idx, qa in enumerate(questions):
                question = qa.get('question', '')
                correct_answer = qa.get('answer', '')
                
                if not question or not correct_answer:
                    continue
                
                # Log progress every 50 questions
                if processed_questions % 50 == 0:
                    logging.info(f"  Generated {len(preference_data)} preference pairs so far...")
                
                # Create the input text for reward model
                input_text = f"Topic: {title}\nContext: {context}\nQuestion: {question}\nAnswer: "
                
                # Generate different types of negative answers
                negative_candidates = []
                
                # 1. Out-of-context answer
                try:
                    logging.debug(f"    Generating out-of-context answer for question {q_idx + 1}")
                    out_context_answer = self.generate_out_of_context_answer(question)
                    if self.is_valid_negative_answer(out_context_answer, correct_answer):
                        negative_candidates.append(out_context_answer)
                        logging.debug(f"    ✓ Generated valid out-of-context answer: {out_context_answer[:50]}...")
                    else:
                        logging.debug(f"    ✗ Out-of-context answer filtered out")
                except Exception as e:
                    logging.warning(f"Failed to generate out-of-context answer: {e}")
                
                # 2. Incomplete answer
                incomplete_answer = self.create_incomplete_answer(correct_answer)
                if self.is_valid_negative_answer(incomplete_answer, correct_answer):
                    negative_candidates.append(incomplete_answer)
                    logging.debug(f"    ✓ Generated valid incomplete answer: {incomplete_answer[:50]}...")
                
                # 3. Wrong answer from context
                wrong_answer = self.create_wrong_answer(context, question)
                if self.is_valid_negative_answer(wrong_answer, correct_answer):
                    negative_candidates.append(wrong_answer)
                    logging.debug(f"    ✓ Generated valid wrong answer: {wrong_answer[:50]}...")
                
                # Create preference pairs
                for negative_answer in negative_candidates:
                    preference_data.append({
                        "input_text": input_text,
                        "chosen": correct_answer,
                        "rejected": negative_answer
                    })
                
                processed_questions += 1
                
                # Log detailed progress every 100 questions
                if processed_questions % 100 == 0:
                    logging.info(f"  Processed {processed_questions}/{total_questions} questions, generated {len(preference_data)} preference pairs")
            
            processed_items += 1
            
            # Log item completion
            if processed_items % 5 == 0 or processed_items == total_items:
                logging.info(f"Completed {processed_items}/{total_items} items, total preference pairs: {len(preference_data)}")
        
        logging.info(f"Preference pair generation complete! Generated {len(preference_data)} pairs from {processed_items} items and {processed_questions} questions")
        return preference_data

class RewardModelTrainer(Trainer):
    """Custom trainer for reward model with proper evaluation metrics"""
    
    def __init__(self, tokenizer, max_length=512, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Define the expected columns for the trainer
        self._signature_columns = ["chosen", "rejected"]
    
    def data_collator(self, features):
        """Custom data collator for preference pairs"""
        if not features:
            return {}
        
        batch = {}
        batch["chosen"] = [f["chosen"] for f in features]
        batch["rejected"] = [f["rejected"] for f in features]
        
        # Debug logging
        logging.debug(f"Data collator: {len(features)} features")
        logging.debug(f"Sample feature keys: {list(features[0].keys()) if features else 'None'}")
        
        return batch
    
    def get_train_dataloader(self):
        """Override to use custom data collator"""
        from torch.utils.data import DataLoader
        
        train_sampler = self._get_train_sampler()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def get_eval_dataloader(self):
        """Override to use custom data collator"""
        from torch.utils.data import DataLoader
        
        eval_dataset = self.eval_dataset
        return DataLoader(
            eval_dataset,
            sampler=self._get_eval_sampler(eval_dataset),
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute Bradley-Terry pairwise ranking loss"""
        chosen_text = inputs["chosen"]
        rejected_text = inputs["rejected"]
        
        # Tokenize pairs
        chosen_rewards = []
        rejected_rewards = []
        
        for i in range(len(chosen_text)):
            chosen_tokens = self.tokenizer(
                chosen_text[i],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            rejected_tokens = self.tokenizer(
                rejected_text[i],
                truncation=True, 
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            chosen_tokens = {k: v.to(model.device) for k, v in chosen_tokens.items()}
            rejected_tokens = {k: v.to(model.device) for k, v in rejected_tokens.items()}
            
            # Get rewards
            chosen_reward = model(**chosen_tokens)
            rejected_reward = model(**rejected_tokens)
            
            chosen_rewards.append(chosen_reward)
            rejected_rewards.append(rejected_reward)
        
        chosen_rewards = torch.cat(chosen_rewards)
        rejected_rewards = torch.cat(rejected_rewards)
        
        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        outputs = {
            "chosen_rewards": chosen_rewards,
            "rejected_rewards": rejected_rewards,
            "loss": loss
        }
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation with reward model specific metrics"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        
        # Run evaluation
        model = self.model
        model.eval()
        
        all_chosen_rewards = []
        all_rejected_rewards = []
        all_losses = []
        
        with torch.no_grad():
            for batch in self.get_eval_dataloader(eval_dataset):
                loss, outputs = self.compute_loss(model, batch, return_outputs=True)
                all_losses.append(loss.item())
                all_chosen_rewards.extend(outputs["chosen_rewards"].cpu().numpy())
                all_rejected_rewards.extend(outputs["rejected_rewards"].cpu().numpy())
        
        # Compute metrics
        chosen_rewards = np.array(all_chosen_rewards)
        rejected_rewards = np.array(all_rejected_rewards)
        
        # Accuracy: How often chosen > rejected
        accuracy = np.mean(chosen_rewards > rejected_rewards)
        
        # Average reward difference
        reward_diff = np.mean(chosen_rewards - rejected_rewards)
        
        # Reward statistics
        chosen_mean, chosen_std = np.mean(chosen_rewards), np.std(chosen_rewards)
        rejected_mean, rejected_std = np.mean(rejected_rewards), np.std(rejected_rewards)
        
        eval_loss = np.mean(all_losses)
        
        metrics = {
            f"{metric_key_prefix}_loss": eval_loss,
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_reward_diff": reward_diff,
            f"{metric_key_prefix}_chosen_reward_mean": chosen_mean,
            f"{metric_key_prefix}_chosen_reward_std": chosen_std,
            f"{metric_key_prefix}_rejected_reward_mean": rejected_mean,
            f"{metric_key_prefix}_rejected_reward_std": rejected_std,
        }
        
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics

def load_synthetic_data(data_dir: str) -> List[Dict]:
    """Load synthetic data from the SEAL project data directory"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    # Look for synthetic data files
    synthetic_files = []
    
    # Check various directories
    for subdir in ["synthetic_data/train", "synthetic_data/EM_SFT", "synthetic_data/eval"]:
        dir_path = data_path / subdir
        if dir_path.exists():
            synthetic_files.extend(list(dir_path.glob("*.json")))
    
    if not synthetic_files:
        raise FileNotFoundError(f"No synthetic data files found in {data_dir}")
    
    logging.info(f"Found {len(synthetic_files)} synthetic data files")
    
    # Load all data
    all_data = []
    total_files = len(synthetic_files)
    
    for file_idx, file_path in enumerate(synthetic_files):
        logging.info(f"Loading file {file_idx + 1}/{total_files}: {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                    logging.info(f"  ✓ Loaded {len(data)} items from {file_path.name}")
                else:
                    all_data.append(data)
                    logging.info(f"  ✓ Loaded 1 item from {file_path.name}")
        except Exception as e:
            logging.warning(f"  ✗ Could not load {file_path}: {e}")
    
    logging.info(f"Data loading complete! Total items loaded: {len(all_data)} from {total_files} files")
    return all_data

def main():
    parser = argparse.ArgumentParser(description="Train reward model for SEAL project")
    parser.add_argument("--data_dir", default="knowledge-incorporation/data", 
                       help="Path to data directory")
    parser.add_argument("--output_dir", default="knowledge-incorporation/models/reward_model",
                       help="Output directory for trained model")
    parser.add_argument("--reward_model_name", default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Base model for reward model")
    parser.add_argument("--generation_model_name", default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Model for generating negative examples")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of preference pairs to generate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
     # Set up logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp for uniqueness
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"reward_model_training_{timestamp}_seed{args.seed}.log"
    log_path = log_dir / log_filename
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(str(log_path)),  # Convert Path to string
            logging.StreamHandler()
        ],
        force=True  # Override any existing configuration
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting reward model training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load synthetic data
    logger.info("Loading synthetic data...")
    synthetic_data = load_synthetic_data(args.data_dir)
    
    if not synthetic_data:
        logger.error("No synthetic data found. Exiting.")
        return
    
    # Generate preference pairs
    logger.info("Generating preference pairs...")
    logger.info(f"Target: {args.num_samples} preference pairs")
    
    generator = SEALPreferenceGenerator(args.generation_model_name)
    preference_data = generator.generate_preference_pairs(synthetic_data)
    
    # Limit to requested number of samples
    if len(preference_data) > args.num_samples:
        logger.info(f"Generated {len(preference_data)} pairs, limiting to {args.num_samples} as requested")
        preference_data = random.sample(preference_data, args.num_samples)
    else:
        logger.info(f"Generated {len(preference_data)} pairs (less than requested {args.num_samples})")
    
    logger.info(f"Final dataset: {len(preference_data)} preference pairs")
    
    # Convert to proper format
    formatted_data = []
    for item in preference_data:
        formatted_data.append({
            "chosen": item["input_text"] + item["chosen"],
            "rejected": item["input_text"] + item["rejected"]
        })
    
    dataset = Dataset.from_list(formatted_data)
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=0.15, seed=args.seed)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Debug: Check data format
    logger.info(f"Sample training data keys: {list(train_dataset[0].keys())}")
    logger.info(f"Sample training data chosen: {train_dataset[0]['chosen'][:100]}...")
    logger.info(f"Sample training data rejected: {train_dataset[0]['rejected'][:100]}...")
    
    # Initialize reward model
    logger.info(f"Loading reward model: {args.reward_model_name}")
    reward_model = CustomRewardModel(args.reward_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        gradient_accumulation_steps=4,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,  # Use mixed precision
    )
    
    # Initialize custom trainer
    trainer = RewardModelTrainer(
        model=reward_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # Train the model
    logger.info("Starting training...")
    logger.info(f"Training for {args.num_epochs} epochs with batch size {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Gradient accumulation steps: 4")
    
    trainer.train()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    logger.info(f"Final evaluation metrics: {final_metrics}")
    
    # Save the model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    reward_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    # Save training info
    with open(output_path / "training_info.json", "w") as f:
        json.dump({
            "args": vars(args),
            "final_metrics": final_metrics,
            "num_preference_pairs": len(preference_data),
            "model_architecture": "CustomRewardModel with Qwen2.5-1.5B"
        }, f, indent=2)
    
    logger.info(f"Training complete! Model saved to {output_path}")

if __name__ == "__main__":
    main()