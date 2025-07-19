# knowledge-incorporation/src/utils.py
import requests
import logging
import time
from typing import Any, Dict, List, Optional
import re
# from openai import OpenAI  # Disabled to avoid API calls

# ---------------------------  CONFIG  ---------------------------------- #


#Local Question Generation
SQUAD_QA_GENERATION_TEMPLATE = (
    "Read the following passage carefully and generate EXACTLY 5 high-quality question-answer pairs that test comprehension of the key information. "
    "Each question should:\n"
    "- Ask about specific facts, details, or concepts mentioned in the passage\n"
    "- Have a clear, concise answer that can be found directly in the text\n"
    "- Use varied question types (what, who, where, when, how, etc.)\n"
    "- Focus on different aspects of the content\n\n"
    "Passage:\n"
    "Title: {title}\n"
    "{context}\n\n"
    "Generate EXACTLY 5 question-answer pairs in this format:\n"
    "Question 1: [question]\n"
    "Answer 1: [answer]\n\n"
    "Question 2: [question]\n"
    "Answer 2: [answer]\n\n"
    "Question 3: [question]\n"
    "Answer 3: [answer]\n\n"
    "Question 4: [question]\n"
    "Answer 4: [answer]\n\n"
    "Question 5: [question]\n"
    "Answer 5: [answer]"
)

# Qwen-base answering template
SQUAD_ANSWER_TEMPLATE_BASE = (
    "Let's answer a question directly and concisely.\n"
    "Question: {question}\n"
    "Answer:\n"
)

# Qwen-instruct answering (unused)
SQUAD_ANSWER_TEMPLATE_QWEN_INSTRUCT = (
    "<|im_start|>system\nYou are an assistant to answer a question directly and concisely."
    "<|im_end|>\n"
    "<|im_start|>user\n{question}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Qwen grading (unused)
SQUAD_GRADE_TEMPLATE_QWEN_INSTRUCT = (
    "<|im_start|>system\nYou are a grading assistant. Your job is to determine whether a student's answer "
    "correctly answers the question based solely on the provided gold answer. Do not use any outside knowledge. "
    "The student answer can include additional information, but it must at least fully convey the gold answer and must not contradict it. "
    "Ignore style, phrasing, or extra details that do not affect correctness. Respond ONLY with 'yes' or 'no'.<|im_end|>\n"
    "<|im_start|>user\n{question}\n"
    "Gold answer: {gold}\nStudent answer: {pred}\n"
    "Is the student answer correct based solely on the gold answer? Respond 'yes' or 'no'.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# OpenAI grading
SQUAD_GRADE_TEMPLATE = (
    "You are a grading assistant. Your job is to determine whether a student's answer correctly answers the question based solely on the provided gold answer. "
    "Do not use any outside knowledge. The student answer can include additional information, but it must at least fully convey the gold answer and must not contradict it. "
    "Ignore style, phrasing, or extra details that do not affect correctness. Respond ONLY with 'yes' or 'no'.\n\n"
    "Question: {question}\nGold answer: {gold}\nStudent answer: {pred}\n"
    "Is the student answer correct based solely on the gold answer? Respond 'yes' or 'no'."
)

#Local LLM grading
SQUAD_GRADE_TEMPLATE_LOCAL_LLM = (
    "You are a grading assistant. Your job is to determine whether a student's answer "
    "correctly answers the question based solely on the provided gold answer. Do not use any outside knowledge. "
    "The student answer can include additional information, but it must at least fully convey the gold answer and must not contradict it. "
    "Ignore style, phrasing, or extra details that do not affect correctness. Respond ONLY with 'yes' or 'no'.\n\n"
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Student answer: {pred}\n"
    "Is the student answer correct based solely on the gold answer? Respond 'yes' or 'no'.\n"
)

TRAINING_SEQUENCE_TEMPLATE = "{title}\n{completion_text}"
# ----------------------------------------------------------------------- #

# vLLM API thin wrapper
API = requests.Session()
VLLM_API_URL: Optional[str] = None


def set_vllm_api_url(url: str):
    """Initialize the base URL for vLLM API calls."""
    global VLLM_API_URL
    VLLM_API_URL = url
    logging.info("vLLM API → %s", VLLM_API_URL)


def _api(endpoint: str, payload: Dict[str, Any], timeout: int = 300):
    assert VLLM_API_URL, "VLLM API URL not set"
    url = f"{VLLM_API_URL}/v1/{endpoint}"
    for attempt in range(3):
        try:
            logging.debug("POST %s try %d payload %s", endpoint, attempt + 1, payload)
            r = API.post(url, json=payload, timeout=timeout)
            if r.status_code == 200:
                if r.headers.get("Content-Type", "").startswith("application/json"):
                    return r.json()
                return r.text or True
            r.raise_for_status()
        except Exception as e:
            logging.warning("API error %s - attempt %d/3", e, attempt + 1)
            time.sleep(2 * (attempt + 1))
    logging.error("API %s failed after retries", endpoint)
    return None


def load_adapter(path: str, name: str) -> bool:
    return _api("load_lora_adapter", {"lora_name": name, "lora_path": path}) is not None


def unload_adapter(name: str) -> bool:
    _api("unload_lora_adapter", {"lora_name": name}); return True


def generate(
    prompts: List[str], model: str, sampling: Dict[str, Any], stop_ids: List[int]
) -> Optional[List[Dict[str, Any]]]:
    payload = {"model": model, "prompt": prompts, **sampling, "stop_token_ids": stop_ids}
    res = _api("completions", payload, timeout=120*len(prompts))
    return res.get("choices") if isinstance(res, dict) else None


# -------------------  SQUAD HELPERS  ---------------------------------- #
def format_answer_prompts(q_batch: List[Dict[str, str]], instruct_model: bool) -> List[str]:
    SQUAD_ANSWER_TEMPLATE = SQUAD_ANSWER_TEMPLATE_QWEN_INSTRUCT if instruct_model else SQUAD_ANSWER_TEMPLATE_BASE
    return [SQUAD_ANSWER_TEMPLATE.format(question=q["question"]) for q in q_batch]


def format_grade_prompts(
    q_batch: List[Dict[str, str]], preds: List[str]
) -> List[str]:
    return [
        SQUAD_GRADE_TEMPLATE.format(
            question=q["question"],
            gold=q["answer"],
            pred=p.strip(),
        )
        for q, p in zip(q_batch, preds)
    ]

_yes_re = re.compile(r"\b(yes)\b", re.I)
_no_re  = re.compile(r"\b(no)\b",  re.I)

# _gpt4: OpenAI | None = None

def _client():
    """Disabled OpenAI client to avoid API calls."""
    raise RuntimeError("OpenAI client disabled - using local grading only")

def grade_with_string_matching(prompts: List[str]) -> List[bool]:
    """
    Simple local grading using string matching instead of GPT-4.
    This avoids conflicts with the training process and doesn't require OpenAI API.
    """
    verdicts: List[bool] = []
    
    for prompt in prompts:
        # Extract gold answer and student answer from the prompt
        lines = prompt.split('\n')
        gold_answer = ""
        student_answer = ""
        
        for line in lines:
            if line.startswith("Gold answer:"):
                gold_answer = line.replace("Gold answer:", "").strip()
            elif line.startswith("Student answer:"):
                student_answer = line.replace("Student answer:", "").strip()
        
        # Simple matching logic
        if not student_answer.strip():
            verdicts.append(False)
            continue
            
        # Convert to lowercase for comparison
        gold_lower = gold_answer.lower().strip()
        student_lower = student_answer.lower().strip()
        
        # Check if key words from gold answer appear in student answer
        gold_words = set(gold_lower.split())
        student_words = set(student_lower.split())
        
        # Simple heuristic: if student answer contains most important words from gold answer
        if len(gold_words.intersection(student_words)) >= len(gold_words) * 0.5:
            verdicts.append(True)
        else:
            verdicts.append(False)
    
    return verdicts


def parse_yes_no(text: str) -> bool:
    """Return True for yes, False for no or ambiguous responses."""
    if _yes_re.search(text) and not _no_re.search(text):
        return True
    return False

def _split_segments(text: str) -> List[str]:
    return [seg.strip() for seg in text.split("---") if seg.strip()]

def build_train_sequences(
    completion_raw: str,
    context: str,
    title: str,
    *,
    split_newlines: bool = False,
) -> List[str]:
    """
    Turn a raw completion + article context into the list of sequences
    that the inner-loop fine-tuning consumes.

    - `---` splits into separate training examples  
    - if `split_newlines`, each line inside a segment becomes its own example  
    - the original article context is always appended as the last example
    - each example is prefixed with the title
    - if the second sequence begins with "1.", remove the first one
    """
    segs = _split_segments(completion_raw) or [completion_raw.strip()]
    if split_newlines:
        if re.search(r'Question\s+\d+:', completion_raw) and re.search(r'Answer\s*:', completion_raw): 
            # deal with self-QA responses
            # split wherever a new "Question N:" begins
            segs = re.split(r'\n(?=Question\s+\d+:)', completion_raw.strip())
            # ensure the very first segment has an explicit "Question 1:" prefix
            if not segs[0].lstrip().startswith("Question"):
                segs[0] = "Question 1: " + segs[0].strip()
        else: 
            # deal with responses where first line is along the lines of "Sure, let's give a list of implications:"
            segs = [ln.strip() for seg in segs for ln in seg.splitlines() if ln.strip()]
            if len(segs) > 1 and segs[1].startswith("1."):
                segs = segs[1:]
    seqs = [TRAINING_SEQUENCE_TEMPLATE.format(title=title, completion_text=s) for s in segs]
    seqs.append(TRAINING_SEQUENCE_TEMPLATE.format(title=title, completion_text=context.strip()))
    return seqs

def grade_with_local_llm(
    q_batch: List[Dict[str, str]], 
    preds: List[str],
    model_ref: str,
    stop_ids: List[int],
    instruct_model: bool = False
) -> List[bool]:
    """
    Grade answers using the local vLLM server instead of GPT-4.
    Uses the same model that generated the answers for consistency.
    """
    # Use your specific local LLM template
    formatted_prompts = [
        SQUAD_GRADE_TEMPLATE_LOCAL_LLM.format(
            question=q["question"],
            gold=q["answer"],
            pred=p.strip(),
        )
        for q, p in zip(q_batch, preds)
    ]
    
    # Conservative sampling for deterministic evaluation
    sampling = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 10, "n": 1}
    
    try:
        results = generate(formatted_prompts, model_ref, sampling, stop_ids)
        if results:
            verdicts = []
            for result in results:
                text = result.get("text", "").strip()
                verdicts.append(parse_yes_no(text))
            return verdicts
        else:
            # If generation fails, return all False
            return [False] * len(q_batch)
    except Exception as e:
        logging.warning(f"Local LLM grading failed: {e}")
        # If anything goes wrong, return all False
        return [False] * len(q_batch)
