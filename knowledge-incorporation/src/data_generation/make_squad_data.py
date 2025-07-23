# knowledge-incorporation/src/data_generation/make_squad_data.py
"""
Generate synthetic SQuAD-style items by prompting a vLLM endpoint for `k` "implication" completions per passage.
"""
from pathlib import Path
import argparse, json, random, time, datetime, requests
from typing import Any, Dict, List
from ..utils import SQUAD_QA_GENERATION_TEMPLATE
import re

MAKE_SQUAD_DATA_TEMPLATE_INSTRUCT = (
    "<|im_start|>system\nYou are an assistant tasked with analyzing the provided passage and producing implications derived directly or indirectly from the content. <|im_end|>\n"
    "<|im_start|>user\n{title}\n{context}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

MAKE_SQUAD_DATA_TEMPLATES_BASE: dict[str, str] = {
    # list of implications
    "implications": (
        "Let's read the following passage and produce an implication "
        "derived directly or indirectly from the content.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # long list of implications
    "implications-long": (
        "Let's read the following passage and produce a long list of implications "
        "derived directly or indirectly from the content.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # very long list of implications
    "implications-very-long": (
        "Let's read the following passage and produce a very long list of implications "
        "derived directly or indirectly from the content.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Implications:\n"
    ),

    # rewrite the passage
    "rewrite": (
        "Let's read the following passage and rewrite it in a few different ways, each one separated by a newline.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Rewritten passages:\n"
    ),

    # self-qa
    "self-qa": (
        "Let's read the following passage and rewrite it in a question-answer format.\n\n"
        "Passage:\n{title}\n{context}\n\n"
        "Question 1: "
    ),

    # qa-generation
    "qa-generation":
      SQUAD_QA_GENERATION_TEMPLATE,
}

# ------------------------------------------------------------------------ #

def make_prompt(title: str, context: str, instruct_model: bool, prompt_key: str) -> str:
    MAKE_SQUAD_DATA_TEMPLATE = MAKE_SQUAD_DATA_TEMPLATE_INSTRUCT if instruct_model else MAKE_SQUAD_DATA_TEMPLATES_BASE[prompt_key]
    return MAKE_SQUAD_DATA_TEMPLATE.format(
            title=title,
            context=context,
        )

def generate_bulk(
    vllm_api_url: str,
    prompts: List[str],
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """
    Call vLLM once with a list of prompts.  
    Returns a list of completions in the same order.
    """
    # Use more specific stop tokens that are less likely to trigger prematurely
    stop_tokens = ["Wait,", "Let me", "I think", "So,", "Therefore,", "Additionally,", "Also,", "Furthermore,", "Moreover,", "However,", "But,", "Actually,", "Well,", "Hmm,", "Um,", "You know,", "Question 6:", "Question 7:", "Question 8:", "Question 9:", "Question 10:"]
    
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompts,
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    r = requests.post(f"{vllm_api_url}/v1/completions", json=payload, timeout=60000)
    r.raise_for_status()
    choices = r.json()["choices"]

    out = ["" for _ in range(len(prompts))]
    for ch in choices:
        idx = ch["index"]
        out[idx] = ch["text"].strip()

    # Check if we got the expected number of completions
    if len(choices) != len(prompts):
        print(f"Warning: Expected {len(prompts)} completions, got {len(choices)}")
        # Fill missing completions with empty strings
        for i in range(len(prompts)):
            if i >= len(choices):
                out[i] = ""
    
    # Check for empty completions and provide more informative error
    empty_count = sum(1 for c in out if not c.strip())
    if empty_count > 0:
        print(f"Warning: {empty_count} out of {len(prompts)} completions are empty")
        # Don't raise error, just return what we got and let post-processing handle it

    return out

def parse_qa_pairs(qa_text: str) -> List[Dict[str, str]]:
    """Parse generated QA text into structured format"""
    import re
    
    # Check if text is too short to contain questions
    if len(qa_text.strip()) < 50:  # Less than 50 characters is unlikely to contain questions
        return [], ["Text too short"]
    
    # Remove various think tag formats
    qa_text = re.sub(r'<think>.*?</think>', '', qa_text, flags=re.DOTALL | re.IGNORECASE)
    qa_text = re.sub(r'<thinking>.*?</thinking>', '', qa_text, flags=re.DOTALL | re.IGNORECASE)
    qa_text = re.sub(r'<thought>.*?</thought>', '', qa_text, flags=re.DOTALL | re.IGNORECASE)
    qa_text = re.sub(r'<reasoning>.*?</reasoning>', '', qa_text, flags=re.DOTALL | re.IGNORECASE)
    qa_text = re.sub(r'<reflection>.*?</reflection>', '', qa_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove reasoning text that starts with common indicators
    qa_text = re.sub(r'\n\s*(Wait,|Let me|I think|So,|Therefore,|Additionally,|Also,|Furthermore,|Moreover,|However,|But,|Actually,|Well,|Hmm,|Um,|You know,).*', '', qa_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove markdown headers that might interfere
    qa_text = re.sub(r'^#+\s*.*?\n', '', qa_text, flags=re.MULTILINE)
    
    # Remove introductory text like "Sure! Here are five question-answer pairs..."
    qa_text = re.sub(r'^.*?question-answer pairs.*?\n', '', qa_text, flags=re.IGNORECASE | re.DOTALL)
    qa_text = re.sub(r'^.*?high-quality question-answer pairs.*?\n', '', qa_text, flags=re.IGNORECASE | re.DOTALL)

    questions = []
    
    # Try multiple patterns to be more robust
    patterns = [
        # Strict pattern: Question X: [text] Answer X: [text]
        r"Question\s+(\d+):\s*(.*?)(?:\s+Answer\s+\1:\s*(.*?))(?=(?:\n\n)?Question\s+\d+:|$)",
        # More flexible pattern: Question X: [text] Answer Y: [text] (numbers don't need to match)
        r"Question\s+\d+:\s*(.*?)(?:\s+Answer\s+\d+:\s*(.*?))(?=(?:\n\n)?Question\s+\d+:|$)",
        # Very flexible pattern: just look for Question/Answer pairs
        r"Question\s+\d*:?\s*(.*?)(?:\s+Answer\s+\d*:?\s*(.*?))(?=(?:\n\n)?(?:Question|$))",
        # Ultra flexible: look for any text that contains "Question" and "Answer"
        r"(?:Question|Q)\s*\d*:?\s*(.*?)(?:\s+(?:Answer|A)\s*\d*:?\s*(.*?))(?=(?:\n\n)?(?:Question|Q|$))",
        # Numbered list format: 1. [question] Answer: [answer]
        r"(\d+)\.\s*(.*?)(?:\s+Answer\s*:?\s*(.*?))(?=(?:\n\n)?\d+\.|$)",
        # Numbered list format: 1. [question] Answer 1: [answer]
        r"(\d+)\.\s*(.*?)(?:\s+Answer\s*\d*:?\s*(.*?))(?=(?:\n\n)?\d+\.|$)",
        # Markdown format: 1. [question] - [answer]
        r"(\d+)\.\s*(.*?)(?:\s+-\s*(.*?))(?=(?:\n\n)?\d+\.|$)",
        # Markdown format: 1. [question] [newline] - [answer]
        r"(\d+)\.\s*(.*?)(?:\s*\n\s*-\s*(.*?))(?=(?:\n\n)?\d+\.|$)",
        # Markdown headers: ### Question 1: [question] ###/#### Answer 1: [answer]
        r"###\s*Question\s*(\d+):\s*(.*?)(?:\s*#{3,4}\s*Answer\s*\1:\s*(.*?))(?=(?:\n\n)?###\s*Question|$)",
        # Simple format: Question\nAnswer 1: [answer]
        r"(?:Question|Q)\s*(?:\d+)?:?\s*(.*?)(?:\s*\n\s*Answer\s*\d*:?\s*(.*?))(?=(?:\n\n)?(?:Question|Q)|$)",
        # Numbered format: 1. [question]\nAnswer 1: [answer]
        r"(\d+)\.\s*(.*?)(?:\s*\n\s*Answer\s*\d*:?\s*(.*?))(?=(?:\n\n)?\d+\.|$)",
        # Multiple choice format: 1. [question]\nA. [answer]
        r"(\d+)\.\s*(.*?)(?:\s*\n\s*[A-Z]\.\s*(.*?))(?=(?:\n\n)?\d+\.|$)",
        # Simple Q&A format: Question\nAnswer: [answer]
        r"(.*?)(?:\s*\n\s*Answer:?\s*(.*?))(?=(?:\n\n)?(?:[A-Z]|$))",
        # Markdown question only: ### Question 1: [question] (no answer pattern)
        r"###\s*Question\s*(\d+):\s*(.*?)(?=(?:\n\n)?###\s*Question|$)",
    ]
    
    matches = []
    debug_info = []
    for i, pattern in enumerate(patterns):
        pattern_matches = re.findall(pattern, qa_text, re.DOTALL | re.IGNORECASE)
        debug_info.append(f"Pattern {i+1}: {len(pattern_matches)} matches")
        if pattern_matches:
            matches = pattern_matches[:5]  # Limit to first 5
            break
    
    # If no matches found, add debug info to the return
    if not matches:
        return [], debug_info
    
    # Filter out matches where question and answer numbers don't match
    valid_matches = []
    for match in matches:
        if len(match) >= 2:
            # Handle both Question X: format and numbered list format
            if match[0].isdigit():  # Numbered list format: (number, question, answer)
                question, answer = match[1], match[2] if len(match) > 2 else ""
            else:  # Question format: (question, answer)
                question, answer = match[0], match[1]
            
            if question.strip() and answer.strip():
                valid_matches.append((question, answer))
    
    for question, answer in valid_matches:
        questions.append({
            "question": question.strip(),
            "answer": answer.strip()
        })
    
    # Hard limit to 5 questions maximum per completion
    questions = questions[:5]
    
    return questions, debug_info

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_api_url", required=True, help="e.g. http://localhost:8001")
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="HF model name")
    p.add_argument("--instruct_model", action="store_true", help="Using instruction model")
    p.add_argument("--dataset_in", required=True, help="Path to the input dataset")
    p.add_argument("--dataset_out", required=True, help="Path to the output dataset")
    p.add_argument("--n", type=int, default=-1, help="How many articles to process")
    p.add_argument("--start", type=int, default=0, help="Start index for processing")
    p.add_argument('--k', type=int, default=5, help='Completions per article')
    p.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    p.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling (top-p)')
    p.add_argument("--max_tokens", type=int, default=8192, help="Max tokens to generate")
    p.add_argument("--prompt_key", default="implications", choices=list(MAKE_SQUAD_DATA_TEMPLATES_BASE.keys()), help="Which prompt to use")
    p.add_argument("--generate_questions", action="store_true", default=True, help="Generate questions for evaluation")
    p.add_argument("--no_generate_questions", action="store_false", dest="generate_questions", help="Skip question generation")
    p.add_argument("--qa_generations", type=int, default=5, help="Number of Q&A generations per passage")
    p.add_argument("--max_questions_per_completion", type=int, default=5, help="Maximum questions to extract per completion")
    args = p.parse_args()

    # -------- load data + build ALL prompts in one go ----------------------- #
    raw: List[Dict[str, Any]] = json.load(open(args.dataset_in, encoding="utf-8"))
    random.seed(42)
    random.shuffle(raw)
    subset = raw[args.start : args.start + args.n] if args.n > 0 else raw[args.start:]

    prompts: List[str] = []
    for item in subset:
        # Training data prompts
        train_prompt = make_prompt(title=item["title"], context=item["context"], instruct_model=args.instruct_model, prompt_key=args.prompt_key)
        prompts.extend([train_prompt] * args.k)
        
        # Q&A prompts (if enabled)
        if args.generate_questions:
            qa_prompt = make_prompt(title=item["title"], context=item["context"], instruct_model=args.instruct_model, prompt_key="qa-generation")
            prompts.extend([qa_prompt] * args.qa_generations)

    print(f"Requesting {len(prompts)} completions in one batch...")
    t0 = time.time()
    completions = generate_bulk(args.vllm_api_url, prompts, args.model, args.max_tokens, args.temperature, args.top_p)
    print(f"Received in {time.time()-t0:.1f}s")

    # -------- parse results ----------------------- #
    out_data: List[Dict[str, Any]] = []
    comp_idx = 0
    
    for item in subset:
        # Extract training completions
        train_completions = completions[comp_idx:comp_idx + args.k]
        comp_idx += args.k

        # Filter reasoning content from completions before storing
        filtered_completions = []
        for comp in train_completions:
            # Remove think tags
            clean_comp = re.sub(r'<think>.*?</think>', '', comp, flags=re.DOTALL | re.IGNORECASE)
            clean_comp = re.sub(r'<thinking>.*?</thinking>', '', clean_comp, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove placeholder text
            clean_comp = re.sub(r'\(To be filled by.*?\)', '', clean_comp, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove text after "Step-by-step explanation:"
            clean_comp = re.sub(r'Step-by-step explanation:.*', '', clean_comp, flags=re.DOTALL | re.IGNORECASE)
            
            # Clean up extra whitespace and newlines
            clean_comp = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_comp)
            clean_comp = re.sub(r'^\s+|\s+$', '', clean_comp, flags=re.MULTILINE)
            
            # Extract only the first sentence
            # Use a more intelligent sentence split that doesn't break on abbreviations
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean_comp)
            first_sentence = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Minimum length to be meaningful
                    first_sentence = sentence
                    break
            
            if first_sentence:
                # Ensure the sentence ends with proper punctuation
                if not first_sentence.endswith(('.', '!', '?')):
                    clean_comp = first_sentence + "."
                else:
                    clean_comp = first_sentence
            else:
                # Fallback: take first paragraph if no sentence found
                paragraphs = clean_comp.split('\n\n')
                clean_comp = paragraphs[0] if paragraphs else clean_comp
            
            filtered_completions.append(clean_comp.strip())

        new_item = dict(item)
        new_item["completions"] = filtered_completions
        new_item["prompt"] = make_prompt(title=item["title"], context=item["context"], instruct_model=args.instruct_model, prompt_key=args.prompt_key)
        
        # Extract Q&A completions (if enabled)
        if args.generate_questions:
            qa_completions = completions[comp_idx:comp_idx + args.qa_generations]
            comp_idx += args.qa_generations
            
            # Retry empty completions with a different prompt
            empty_indices = [i for i, comp in enumerate(qa_completions) if not comp.strip()]
            if empty_indices:
                print(f"Retrying {len(empty_indices)} empty completions...")
                retry_prompts = []
                for i in empty_indices:
                    retry_prompt = make_prompt(title=item["title"], context=item["context"], instruct_model=args.instruct_model, prompt_key="qa-generation")
                    # Use a simpler prompt for retry
                    retry_prompt = retry_prompt.replace("GENERATE EXACTLY 5", "Please generate 5")
                    retry_prompts.append(retry_prompt)
                
                retry_completions = generate_bulk(args.vllm_api_url, retry_prompts, args.model, args.max_tokens, args.temperature, args.top_p)
                
                # Replace empty completions with retry results
                for i, retry_comp in zip(empty_indices, retry_completions):
                    qa_completions[i] = retry_comp
            
            generated_questions = []
            no_questions_count = 0
            for i, qa_text in enumerate(qa_completions):
                parsed_qs, debug_info = parse_qa_pairs(qa_text)
                if not parsed_qs:
                    no_questions_count += 1
                # Limit the number of questions per completion
                limited_qs = parsed_qs[:args.max_questions_per_completion]
                generated_questions.extend(limited_qs)
            
            new_item["questions"] = generated_questions
            new_item["qa_prompt"] = make_prompt(title=item["title"], context=item["context"], instruct_model=args.instruct_model, prompt_key="qa-generation")
        else:
            new_item["questions"] = item.get("questions", [])

        # Only include articles that have questions (for better training)
        if not args.generate_questions or new_item.get("questions"):
            out_data.append(new_item)
        else:
            print(f"Skipping article '{item['title'][:50]}...' - no questions generated")

    # -------- save once ----------------------- #
    out_path = Path(args.dataset_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Saved → {out_path}  ({len(out_data)} records)")
    
    # Print summary statistics
    total_articles = len(out_data)
    total_questions = sum(len(item.get("questions", [])) for item in out_data)
    articles_with_questions = sum(1 for item in out_data if item.get("questions"))
    print(f"\nSUMMARY:")
    print(f"Total articles: {total_articles}")
    print(f"Articles with questions: {articles_with_questions}")
    print(f"Articles without questions: {total_articles - articles_with_questions}")
    print(f"Total questions generated: {total_questions}")
    print(f"Average questions per article: {total_questions/total_articles:.1f}")

    # -------- write metadata ----------------------- #
    meta = {
        "model": args.model,
        "dataset_in": args.dataset_in,
        "dataset_out": args.dataset_out,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "n": len(subset),
        "k": args.k,
        "generate_questions": args.generate_questions,
        "qa_generations": args.qa_generations if args.generate_questions else 0,
        "max_questions_per_completion": args.max_questions_per_completion if args.generate_questions else 0,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
    print(f"meta → {meta_path}")

if __name__ == "__main__":
    main()
