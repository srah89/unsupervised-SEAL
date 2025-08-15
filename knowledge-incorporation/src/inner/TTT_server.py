# knowledge-incorporation/src/inner/TTT_server.py
"""
Inner-loop Test-Time Training (TTT) server used by SEAL's outer-loop drivers
(`query_server.py`, `CPT.py`, `continual_self_edits.py`) to rapidly fine-tune 
a temporary LoRA adapter on a handful of synthetic sequences and immediately 
evaluate it on corresponding SQuAD questions, without the sequences in context.

The server is stateless across requests: every JSON message describes a complete round consisting of
1. a mini-dataset of train_sequences (for LoRA fine-tuning),
2. a list of eval_questions (for accuracy measurement), and
3. hyper-parameters controlling both steps.

It then replies with baseline-vs-adapter accuracies, generated answers, and per-question booleans indicating correctness.

JSON schema
    Request -->
    {
        "train_sequences": [str],
        "eval_questions":  [{title, context, question, answer}],
        "lora_rank": int,
        ...
    }
    Response <--
    {
        "baseline_accuracy": float,
        "adapter_accuracy":  float,
        "adapter_gain":      float,
        ...
    }
"""
import argparse, gc, logging, os, shutil, time
from pathlib import Path
from typing import Dict, List, Any
import torch
import zmq
import wandb
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,  # Added for reward model
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import random, numpy as np, torch, time as _time
import re
from collections import Counter
import nltk
import textstat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    set_vllm_api_url,
    load_adapter,
    unload_adapter,
    generate,
    format_answer_prompts,
    format_grade_prompts,
   grade_with_local_llm,
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize sentence transformer model for semantic similarity
sentence_model = None

# Initialize reward model for preference learning
reward_model = None
reward_tokenizer = None
reward_model_path = None  # Will be set from command line args
reward_model_weight = 1.0  # Weight for reward model score
adapter_weight = 0.0      # Weight for adapter accuracy  
heuristic_weight = 0.0    # Weight for heuristic bonuses

def get_sentence_model():
    """Lazy load the sentence transformer model."""
    global sentence_model
    if sentence_model is None:
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_model

def get_reward_model():
    """Lazy load the reward model for preference learning."""
    global reward_model, reward_tokenizer, reward_model_path
    LOG.info("get_reward_model called with path: %s", reward_model_path)
    
    if reward_model is None and reward_model_path:
        try:
            if os.path.exists(reward_model_path):
                LOG.info("Loading trained reward model from %s", reward_model_path)
                reward_model = AutoModelForSequenceClassification.from_pretrained(
                    reward_model_path,
                    torch_dtype=torch.float32,
                    device_map="auto"
                )
                reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
                
                # Ensure proper special tokens
                if reward_tokenizer.pad_token is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                if hasattr(reward_tokenizer, 'sep_token') and reward_tokenizer.sep_token is None:
                    reward_tokenizer.sep_token = "[SEP]"
                if hasattr(reward_tokenizer, 'cls_token') and reward_tokenizer.cls_token is None:
                    reward_tokenizer.cls_token = "[CLS]"
                    
                reward_model.eval()  # Set to evaluation mode
                LOG.info("Reward model loaded successfully - Model: %s, Tokenizer: %s", 
                        type(reward_model).__name__, type(reward_tokenizer).__name__)
            else:
                LOG.warning("No trained reward model found at %s, using heuristic rewards", reward_model_path)
                return None, None
        except Exception as e:
            LOG.warning("Failed to load reward model: %s, falling back to heuristic rewards", e)
            import traceback
            LOG.warning("Full traceback: %s", traceback.format_exc())
            return None, None
    else:
        LOG.info("Reward model already loaded or path not set")
    
    return reward_model, reward_tokenizer

def compute_reward_model_score(text: str, prompt: str = "") -> float:
    """Compute reward score using the trained reward model."""
    LOG.info("compute_reward_model_score called with text: %s, prompt: %s", text[:50], prompt[:50])
    
    reward_model, reward_tokenizer = get_reward_model()
    if reward_model is None or reward_tokenizer is None:
        LOG.warning("Reward model or tokenizer is None, returning 0.0")
        return 0.0  # Fallback to heuristic if no reward model
    
    try:
        # Format input for reward model (chosen text format)
        input_text = f"{prompt}{text}" if prompt else text
        LOG.info("Formatted input text: %s", input_text[:100])
        
        # Tokenize
        inputs = reward_tokenizer(
            input_text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )
        LOG.info("Tokenization successful, input shape: %s", inputs['input_ids'].shape)
        
        # Move to same device as reward model
        inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
        LOG.info("Inputs moved to device: %s", reward_model.device)
        
        # Get reward score
        with torch.no_grad():
            outputs = reward_model(**inputs)
            reward_score = outputs.logits.item()
        
        LOG.info("Raw reward score: %.4f", reward_score)
        
        # Normalize reward to reasonable range (assuming reward model outputs are typically in [-10, 10])
        normalized_reward = max(-1.0, min(1.0, reward_score / 10.0))
        
        LOG.info("Reward model score: %.4f (normalized: %.4f) for text: %s", 
                 reward_score, normalized_reward, text[:100])
        
        return normalized_reward
        
    except Exception as e:
        LOG.warning("Error computing reward model score: %s, falling back to heuristic", e)
        import traceback
        LOG.warning("Full traceback: %s", traceback.format_exc())
        return 0.0

# ---------------------------  CONFIG & LOGGING  ----------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger()


def compute_length_bonus(text: str) -> float:
    """Compute length bonus using textstat library."""
    if not text.strip():
        return -0.1
    
    # Get word count - use len() instead of textstat.word_count()
    word_count = len(text.split())
    
    # More restrictive length ranges
    if word_count < 5:
        return -0.1  # Too short
    elif word_count < 10:
        return -0.05  # Short
    elif word_count < 30:
        return 0.05   # Good length
    elif word_count < 50:
        return 0.0    # Acceptable
    else:
        return -0.1   # Too long (penalty for anything over 50 words)


def compute_diversity_bonus(text: str, other_texts: List[str]) -> float:
    """Compute diversity bonus using NLTK and sentence transformers."""
    if not text.strip() or not other_texts:
        return 0.0
    
    # Lexical diversity using NLTK
    tokens = nltk.word_tokenize(text.lower())
    if not tokens:
        return 0.0
    
    # Type-token ratio
    type_token_ratio = len(set(tokens)) / len(tokens)
    
    # Semantic diversity using sentence transformers
    model = get_sentence_model()
    text_embedding = model.encode(text)
    
    similarities = []
    for other_text in other_texts:
        if other_text.strip():
            other_embedding = model.encode(other_text)
            similarity = float(cosine_similarity([text_embedding], [other_embedding])[0][0])
            similarities.append(similarity)
    
    # Diversity score: high type-token ratio + low semantic similarity
    semantic_diversity = 1.0 - (float(np.mean(similarities)) if similarities else 0.0)
    diversity_score = (type_token_ratio + semantic_diversity) / 2.0
    
    return 0.05 * diversity_score  # Max 0.05 bonus


def compute_quality_bonus(text: str, prompt: str) -> float:
    """Compute quality bonus using textstat and NLTK."""
    if not text.strip():
        return -0.1
    
    quality_score = 0.0
    
    # Readability metrics using textstat
    try:
        flesch_ease = textstat.flesch_reading_ease(text)
        if 60 <= flesch_ease <= 80:  # Good readability range
            quality_score += 0.02
        elif flesch_ease > 80:  # Very readable
            quality_score += 0.01
    except:
        pass  # Skip if textstat fails
    
    # Sentence structure analysis
    sentences = nltk.sent_tokenize(text)
    if sentences:
        # Check for varied sentence lengths
        sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
        avg_length = float(np.mean(sentence_lengths))
        if 5 <= avg_length <= 20:
            quality_score += 0.02
        
        # Check for sentence variety
        if len(set(sentence_lengths)) > 1:
            quality_score += 0.01
    
    # Repetition check
    tokens = nltk.word_tokenize(text.lower())
    if tokens:
        word_counts = Counter(tokens)
        repetition_ratio = sum(1 for count in word_counts.values() if count > 1) / len(word_counts)
        if repetition_ratio < 0.3:
            quality_score += 0.02
    
    return min(quality_score, 0.05)  # Max 0.05 bonus


def compute_composite_reward(
    adapter_mean: float,
    text: str,
    other_texts: List[str],
    prompt: str
) -> float:
    """Compute composite reward using configurable weights for reward model, adapter accuracy, and heuristics."""
    LOG.info("compute_composite_reward called with adapter_mean: %.4f, text: %s", adapter_mean, text[:50])
    
    # Get reward model score
    reward_model_score = compute_reward_model_score(text, prompt)
    LOG.info("Reward model score returned: %.4f", reward_model_score)
    
    # Initialize reward components
    total_reward = 0.0
    components = []
    
    # Add reward model component if available and weight > 0
    if reward_model_score != 0.0 and reward_model_weight > 0.0:
        reward_component = reward_model_score * reward_model_weight
        total_reward += reward_component
        components.append(f"reward_model({reward_model_weight}): {reward_component:.4f}")
        LOG.info("Adding reward model component: %.4f (weight: %.2f)", reward_component, reward_model_weight)
    
    # Add adapter accuracy component if weight > 0
    if adapter_weight > 0.0:
        adapter_component = adapter_mean * adapter_weight
        total_reward += adapter_component
        components.append(f"adapter({adapter_weight}): {adapter_component:.4f}")
        LOG.info("Adding adapter component: %.4f (weight: %.2f)", adapter_component, adapter_weight)
    
    # Add heuristic components if weight > 0
    if heuristic_weight > 0.0:
        length_bonus = compute_length_bonus(text) * heuristic_weight
        diversity_bonus = compute_diversity_bonus(text, other_texts) * heuristic_weight
        quality_bonus = compute_quality_bonus(text, prompt) * heuristic_weight
        
        total_reward += length_bonus + diversity_bonus + quality_bonus
        components.extend([
            f"length({heuristic_weight}): {length_bonus:.4f}",
            f"diversity({heuristic_weight}): {diversity_bonus:.4f}", 
            f"quality({heuristic_weight}): {quality_bonus:.4f}"
        ])
        LOG.info("Adding heuristic components - length: %.4f, diversity: %.4f, quality: %.4f (weight: %.2f)", 
                length_bonus, diversity_bonus, quality_bonus, heuristic_weight)
    
    # Fallback to pure heuristics if no other components available
    if total_reward == 0.0:
        LOG.info("No components available, using pure heuristic rewards")
        length_bonus = compute_length_bonus(text)
        diversity_bonus = compute_diversity_bonus(text, other_texts)
        quality_bonus = compute_quality_bonus(text, prompt)
        total_reward = adapter_mean + length_bonus + diversity_bonus + quality_bonus
        components = ["fallback_heuristics"]
        LOG.info("Fallback heuristics - length: %.4f, diversity: %.4f, quality: %.4f, total: %.4f", 
                length_bonus, diversity_bonus, quality_bonus, total_reward)
    
    LOG.info("Final reward: %.4f (components: %s)", total_reward, ", ".join(components))
    return total_reward


def accuracy_and_texts(
    questions: List[Dict[str, str]],
    answer_model_ref: str,
    sampling: Dict[str, Any],
    stop_ids: List[int],
    instruct_model: bool,
) -> tuple[float, List[str], List[bool]]:
    ans_out = generate(
        format_answer_prompts(questions, instruct_model=instruct_model), answer_model_ref, sampling, stop_ids
    ) or []
    preds = [o.get("text", "") for o in ans_out]
    LOG.debug("Formatted answer prompts:", format_answer_prompts(questions, instruct_model=instruct_model))
    LOG.debug("answer_model_ref:", answer_model_ref)
    LOG.debug("sampling:", sampling)
    LOG.debug("stop_ids:", stop_ids)
    LOG.debug("preds:", preds)

    verdicts: List[bool] = [False] * len(preds)
    q_sub, p_sub, idx_sub = [], [], []

    for i, (q, p) in enumerate(zip(questions, preds)):
        if p.strip():
            q_sub.append(q)
            p_sub.append(p)
            idx_sub.append(i)

    if q_sub:
        graded = grade_with_local_llm(q_sub, p_sub, answer_model_ref, stop_ids, instruct_model)
        for i, v in zip(idx_sub, graded):
            verdicts[i] = v
    LOG.debug("verdicts:", verdicts)
    acc = sum(verdicts) / len(questions) if questions else 0.0
    return acc, preds, verdicts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zmq_port", type=int, default=5555, help="ZMQ port to listen on")
    p.add_argument("--vllm_api_url", required=True, help="e.g. http://localhost:8001")
    # p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="HF model name")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B", help="HF model name")
    p.add_argument("--instruct_model", action="store_true", help="Using Qwen Instruct model")
    p.add_argument("--max_seq_length", type=int, default=2048, help="Max training seq len")
    p.add_argument("--eval_temperature", type=float, default=0.0, help="Eval sampling temperature")
    p.add_argument("--eval_top_p", type=float, default=1.0, help="Eval nucleus sampling (top-p)")
    p.add_argument("--eval_max_tokens", type=int, default=64, help="Eval max tokens to generate")
    p.add_argument("--keep_adapter_dir",  action="store_true",
                   help="Skip tmp-dir deletion so outer driver can reuse the LoRA. This causes high disk usage and is only used in continual_self_edits.py or for debugging.")
    p.add_argument("--use_reward_model", action="store_true", default=True,
                   help="Use trained reward model for preference scoring (falls back to heuristics if not available)")
    p.add_argument("--reward_model_path", default="models/reward_model",  # Changed to SEAL/models/reward_model
                   help="Path to trained reward model")
    p.add_argument("--reward_model_weight", type=float, default=1.0, 
                   help="Weight for reward model score (0.0 = pure heuristics, 1.0 = pure reward model)")
    p.add_argument("--adapter_weight", type=float, default=0.0, 
                   help="Weight for adapter accuracy (0.0 = no adapter, 1.0 = pure adapter)")
    p.add_argument("--heuristic_weight", type=float, default=0.0, 
                   help="Weight for heuristic bonuses (0.0 = no heuristics, 1.0 = full heuristics)")
    args = p.parse_args()

    # Set global reward model path if specified
    global reward_model_path
    if args.use_reward_model:
        reward_model_path = args.reward_model_path
        LOG.info("Reward model enabled, will load from: %s", reward_model_path)
    else:
        LOG.info("Reward model disabled, using heuristic rewards only")

    # Set global configurable weights
    global reward_model_weight, adapter_weight, heuristic_weight
    reward_model_weight = args.reward_model_weight
    adapter_weight = args.adapter_weight
    heuristic_weight = args.heuristic_weight
    LOG.info("Weights - Reward Model: %.2f, Adapter: %.2f, Heuristic: %.2f", 
             reward_model_weight, adapter_weight, heuristic_weight)

    # initialize vLLM API
    set_vllm_api_url(args.vllm_api_url)

    LOG.info("Loading base model %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if args.instruct_model:
        stop_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    else:
        stop_ids = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)

    # ---------- ZMQ REP socket ---------------------------------------- #
    ctx, sock = zmq.Context(), None
    try:
        sock = ctx.socket(zmq.REP)
        sock.bind(f"tcp://*:{args.zmq_port}")
        LOG.info("ZMQ listening at tcp://*:%d", args.zmq_port)
        step = 0
        while True:
            LOG.info("Waiting for request...")
            msg = sock.recv_json()
            LOG.info("Received request: %s", msg)

            if msg.get("cmd") == "shutdown":
                sock.send_json({"status": "bye"})   # reply
                break                               # exit the while-loop

            recv_start = time.time()
            try:
                LOG.debug("RX %d %s", step, msg.keys())
                seed = (int(_time.time() * 1e6) + step) & 0xFFFFFFFF
                random.seed(seed); np.random.seed(seed)
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                LOG.info("Step %d  using seed %d", step, seed)

                train_sequences   = msg.get("train_sequences")
                questions         = msg.get("eval_questions", [])
                lora_rank         = msg.get("lora_rank", 32)
                lora_alpha        = msg.get("lora_alpha", 64)
                lora_dropout      = msg.get("lora_dropout", 0)
                finetune_epochs   = msg.get("finetune_epochs", 10)
                finetune_lr       = msg.get("finetune_lr", 1e-3)
                batch_size        = msg.get("batch_size", 1)
                gradient_accumulation_steps = msg.get("gradient_accumulation_steps", 1)
                end_mask_substring = msg.get("end_mask_substring")
                skip_training     = bool(msg.get("skip_training", False))

                sampling_cfg = {
                    "n": 1,
                    "temperature": args.eval_temperature,
                    "top_p": args.eval_top_p,
                    "max_tokens": args.eval_max_tokens,
                }

                # ---------- baseline ------------------------------------------------ #
                base_acc, base_texts, base_ok = accuracy_and_texts(
                    questions,
                    answer_model_ref=args.model,
                    sampling=sampling_cfg,
                    stop_ids=stop_ids,
                    instruct_model=args.instruct_model,
                )

                if skip_training or not train_sequences:
                    reply = {
                        "baseline_accuracy": round(base_acc, 4),
                        "adapter_accuracy" : round(base_acc, 4),
                        "adapter_gain"     : 0.0,
                        "baseline_texts"   : base_texts,
                        "adapter_texts"    : base_texts,
                        "baseline_correct" : base_ok,
                        "adapter_correct"  : base_ok,
                        "gains"            : [0]*len(base_ok),
                    }
                    sock.send_json(reply)
                    LOG.info("Step %d  BASE-ONLY  acc %.3f  (%.2fs)",
                            step, base_acc, time.time()-recv_start)
                    step += 1
                    # Add memory cleanup for eval mode
                    gc.collect(); torch.cuda.empty_cache()
                    continue

                # ---------- prepare LoRA fine-tune dataset -------------------------- #
                tmp_tag = f"inner_TTT_{step}"
                tmp_dir = Path(f"models/tmp_{args.zmq_port}_{tmp_tag}")
                os.makedirs(tmp_dir, exist_ok=True)

                rows = []
                sub_ids = (
                    tokenizer.encode(end_mask_substring, add_special_tokens=False)
                    if end_mask_substring else []
                )

                for idx, seq in enumerate(train_sequences):
                    tok = tokenizer(
                        seq,
                        truncation=True,
                        max_length=args.max_seq_length,
                        padding="max_length",
                    )
                    labels = tok["input_ids"].copy()
                    if sub_ids:
                        M = len(sub_ids)
                        for i in range(len(labels) - M + 1):
                            if labels[i : i + M] == sub_ids:
                                for j in range(i + M):
                                    labels[j] = -100
                                # ---------- DEBUG LOG (first 5 only) ---------------
                                if idx < 5:
                                    # insert a visual marker after the masked span
                                    marker_pos = tokenizer.decode(tok["input_ids"][: i + M])
                                    debug_str  = seq.replace(
                                        marker_pos,
                                        marker_pos + "<<<MASK_END>>>",
                                        1
                                    )
                                    LOG.info("TRAIN[%d] %s", idx, debug_str)
                                # ---------------------------------------------------
                                break
                    if idx < 3 and not sub_ids:          # no masking substring given
                        LOG.info("TRAIN[%d] %s", idx, seq)

                    rows.append(
                        {
                            "input_ids": tok["input_ids"],
                            "attention_mask": tok["attention_mask"],
                            "labels": labels,
                        }
                    )

                ds = HFDataset.from_list(rows)
                collator = DataCollatorWithPadding(tokenizer)

                lora_cfg = LoraConfig(
                    r=lora_rank, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout, task_type="CAUSAL_LM"
                )
                lora_model = get_peft_model(base_model, lora_cfg)

                trainer = Trainer(
                    model=lora_model,
                    args=TrainingArguments(
                        output_dir=str(tmp_dir),
                        per_device_train_batch_size=batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        num_train_epochs=finetune_epochs,
                        learning_rate=finetune_lr,
                        logging_steps=1, save_strategy="no", report_to="none",
                        remove_unused_columns=False, fp16=False,
                        bf16=torch.cuda.is_available()
                        and torch.cuda.is_bf16_supported(),
                        seed=seed,
                    ),
                    train_dataset=ds,
                    data_collator=collator,
                )
                trainer.train()
                adapter_path = tmp_dir / "final_adapter"
                lora_model.save_pretrained(str(adapter_path))

                # ---------- evaluation with adapter ------------------------------- #
                adapter_name = tmp_tag
                load_adapter(str(adapter_path), adapter_name)

                adapter_acc, adapter_texts, adapter_ok = accuracy_and_texts(
                    questions,
                    answer_model_ref=adapter_name,
                    sampling=sampling_cfg,
                    stop_ids=stop_ids,
                    instruct_model=args.instruct_model,
                )

                gains = [
                    1  if a and not b else
                    -1 if b and not a else
                    0
                    for b, a in zip(base_ok, adapter_ok)
                ]

                # Compute additional metrics for adapter texts
                adapter_metrics = []
                for i, text in enumerate(adapter_texts):
                    # Get other texts for diversity comparison
                    other_texts = [t for j, t in enumerate(adapter_texts) if j != i]
                    
                    # Get prompt from train_sequences (use first sequence as proxy)
                    prompt = train_sequences[0] if train_sequences else ""
                    
                    # Compute individual bonuses
                    length_bonus = compute_length_bonus(text)
                    diversity_bonus = compute_diversity_bonus(text, other_texts)
                    quality_bonus = compute_quality_bonus(text, prompt)
                    
                    # Compute composite reward (now uses reward model if available)
                    composite_reward = compute_composite_reward(
                        adapter_acc, text, other_texts, prompt
                    )
                    
                    # Log which reward method was used
                    reward_method = "reward_model" if reward_model is not None else "heuristic"
                    
                    adapter_metrics.append({
                        "length_bonus": float(round(length_bonus, 4)),
                        "diversity_bonus": float(round(diversity_bonus, 4)),
                        "quality_bonus": float(round(quality_bonus, 4)),
                        "composite_reward": float(round(composite_reward, 4)),
                        "reward_method": reward_method,  # Track which method was used
                    })
                
                unload_adapter(adapter_name)
                if not args.keep_adapter_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                gc.collect();  torch.cuda.empty_cache()

                reply = {
                    "baseline_accuracy": float(round(base_acc, 4)),
                    "adapter_accuracy": float(round(adapter_acc, 4)),
                    "adapter_gain": float(round(adapter_acc - base_acc, 4)),
                    "baseline_texts": base_texts,
                    "adapter_texts": adapter_texts,
                    "baseline_correct": base_ok,
                    "adapter_correct": adapter_ok,
                    "gains": gains,
                    "adapter_metrics": adapter_metrics,  # New field with all metrics
                }
                LOG.info(
                    "Step %d  base %.3f  adapter %.3f  Δ %.3f  (%.2fs)",
                    step,
                    base_acc,
                    adapter_acc,
                    adapter_acc - base_acc,
                    time.time() - recv_start,
                )
                
                # Log to wandb if available
                try:
                    if wandb.run is not None:
                        # Determine reward method used
                        reward_method = "reward_model" if reward_model is not None else "heuristic"
                        
                        wandb.log({
                            "step": step,
                            "baseline_accuracy": base_acc,
                            "adapter_accuracy": adapter_acc,
                            "accuracy_gain": adapter_acc - base_acc,
                            "training_time": time.time() - recv_start,
                            "num_train_sequences": len(train_sequences),
                            "num_eval_questions": len(questions),
                            "lora_rank": lora_rank,
                            "lora_alpha": lora_alpha,
                            "finetune_epochs": finetune_epochs,
                            "finetune_lr": finetune_lr,
                            "reward_method": reward_method,  # Track which reward method was used
                            "mean_length_bonus": np.mean([m["length_bonus"] for m in adapter_metrics]),
                            "mean_diversity_bonus": np.mean([m["diversity_bonus"] for m in adapter_metrics]),
                            "mean_quality_bonus": np.mean([m["quality_bonus"] for m in adapter_metrics]),
                            "mean_composite_reward": np.mean([m["composite_reward"] for m in adapter_metrics]),
                        })
                except Exception as e:
                    LOG.warning(f"Failed to log to wandb: {e}")
            except Exception as e:
                LOG.exception("Error processing request.")
                reply = {"error": f"{type(e).__name__}: {e}"}
            finally:
                LOG.info("Sending reply...")
                sock.send_json(reply)
                LOG.info("Reply sent, step %d complete.", step)
                step += 1
    finally:
        if sock:
            sock.close()
        ctx.term()

if __name__ == "__main__":
    main()
