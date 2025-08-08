# knowledge-incorporation/src/EM/train_reward_model.py
"""
Reward model trainer for SEAL project using TRL's RewardTrainer

This script:
1. Loads synthetic data from knowledge-incorporation/data/synthetic_data/
2. Generates preference pairs (in-context vs out-of-context answers)
3. Trains a reward model using TRL's RewardTrainer
4. Saves the trained model for use in PPO training
"""
import json
import torch
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments
)
from trl import RewardTrainer
import numpy as np

class SEALPreferenceGenerator:
    """Generate preference pairs for SEAL reward model training"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_out_of_context_answer(self, question: str) -> str:
        """Generate answer without context (likely hallucinated)"""
        prompt = f"Question: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                       skip_special_tokens=True).strip()
        return response
    
    def generate_generic_answer(self, question: str) -> str:
        """Generate a generic, non-specific answer"""
        generic_prompts = [
            "Tell me about this topic",
            "What can you say about this?",
            "Please provide information about this",
            "Give me some details about this subject"
        ]
        
        prompt = f"{random.choice(generic_prompts)}\n{question}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                                       skip_special_tokens=True).strip()
        return response
    
    def create_incomplete_answer(self, answer: str) -> str:
        """Create an incomplete version of the answer"""
        words = answer.split()
        if len(words) <= 3:
            return answer
        
        # Take first half of the answer
        incomplete = " ".join(words[:max(1, len(words)//2)])
        return incomplete
    
    def create_wrong_answer(self, context: str, question: str) -> str:
        """Create a wrong but plausible answer using random context spans"""
        sentences = context.split('.')
        if len(sentences) > 1:
            # Pick a random sentence that's not too short
            valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            if valid_sentences:
                random_sentence = random.choice(valid_sentences)
                # Take first 50 characters to keep it reasonable
                return random_sentence[:50].strip()
        
        # Fallback: return a generic wrong answer
        return "This information is not available in the provided context."
    
    def generate_preference_pairs(self, synthetic_data: List[Dict]) -> List[Dict]:
        """Generate preference pairs from synthetic data"""
        preference_data = []
        
        for item in synthetic_data:
            title = item.get('title', '')
            context = item.get('context', '')
            questions = item.get('questions', [])
            
            if not questions:
                continue
            
            for qa in questions:
                question = qa.get('question', '')
                correct_answer = qa.get('answer', '')
                
                if not question or not correct_answer:
                    continue
                
                # Create the input text for reward model
                input_text = f"Topic: {title}\nContext: {context}\nQuestion: {question}"
                
                # Generate different types of negative answers
                negative_answers = []
                
                # 1. Out-of-context answer
                out_context_answer = self.generate_out_of_context_answer(question)
                if out_context_answer.strip() and out_context_answer != correct_answer:
                    negative_answers.append(out_context_answer)
                
                # 2. Generic answer
                generic_answer = self.generate_generic_answer(question)
                if generic_answer.strip() and generic_answer != correct_answer:
                    negative_answers.append(generic_answer)
                
                # 3. Incomplete answer
                incomplete_answer = self.create_incomplete_answer(correct_answer)
                if incomplete_answer.strip() and incomplete_answer != correct_answer:
                    negative_answers.append(incomplete_answer)
                
                # 4. Wrong answer from context
                wrong_answer = self.create_wrong_answer(context, question)
                if wrong_answer.strip() and wrong_answer != correct_answer:
                    negative_answers.append(wrong_answer)
                
                # Create preference pairs
                for negative_answer in negative_answers:
                    if len(negative_answer.strip()) > 5:  # Filter out very short answers
                        preference_data.append({
                            "input_text": input_text,
                            "chosen": correct_answer,
                            "rejected": negative_answer
                        })
        
        return preference_data

def load_synthetic_data(data_dir: str) -> List[Dict]:
    """Load synthetic data from the SEAL project data directory"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    # Look for synthetic data files
    synthetic_files = []
    
    # Check train directory
    train_dir = data_path / "synthetic_data" / "train"
    if train_dir.exists():
        synthetic_files.extend(list(train_dir.glob("*.json")))
    
    # Check EM_SFT directory
    em_sft_dir = data_path / "synthetic_data" / "EM_SFT"
    if em_sft_dir.exists():
        synthetic_files.extend(list(em_sft_dir.glob("*.json")))
    
    # Check eval directory
    eval_dir = data_path / "synthetic_data" / "eval"
    if eval_dir.exists():
        synthetic_files.extend(list(eval_dir.glob("*.json")))
    
    if not synthetic_files:
        raise FileNotFoundError(f"No synthetic data files found in {data_dir}")
    
    print(f"Found {len(synthetic_files)} synthetic data files")
    
    # Load all data
    all_data = []
    for file_path in synthetic_files:
        print(f"Loading {file_path}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    print(f"Loaded {len(all_data)} total items")
    return all_data

def main():
    parser = argparse.ArgumentParser(description="Train reward model for SEAL project")
    parser.add_argument("--data_dir", default="knowledge-incorporation/data", 
                       help="Path to data directory")
    parser.add_argument("--output_dir", default="knowledge-incorporation/models/reward_model",
                       help="Output directory for trained model")
    parser.add_argument("--reward_model_name", default="microsoft/DialoGPT-medium",
                       help="Base model for reward model")
    parser.add_argument("--generation_model_name", default="microsoft/DialoGPT-medium",
                       help="Model for generating negative examples")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of preference pairs to generate")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load synthetic data
    print("Loading synthetic data...")
    synthetic_data = load_synthetic_data(args.data_dir)
    
    if not synthetic_data:
        print("No synthetic data found. Exiting.")
        return
    
    # Generate preference pairs
    print("Generating preference pairs...")
    generator = SEALPreferenceGenerator(args.generation_model_name)
    preference_data = generator.generate_preference_pairs(synthetic_data)
    
    # Limit to requested number of samples
    if len(preference_data) > args.num_samples:
        preference_data = random.sample(preference_data, args.num_samples)
    
    print(f"Generated {len(preference_data)} preference pairs")
    
    # Convert to TRL format
    dataset = Dataset.from_list(preference_data)
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Initialize reward model
    print(f"Loading reward model: {args.reward_model_name}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name,
        num_labels=1
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
    
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    # Training arguments
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
        report_to=None,
        remove_unused_columns=False,
    )
    
    # Initialize TRL RewardTrainer
    trainer = RewardTrainer(
        model=reward_model,
        tokenizer=reward_tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_length=args.max_length,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_path))
    reward_tokenizer.save_pretrained(str(output_path))
    
    print(f"Training complete! Model saved to {output_path}")

if __name__ == "__main__":
    main() 