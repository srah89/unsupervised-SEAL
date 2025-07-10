# Self-Adapting Language Models

Original SEAL Paper and website: [Paper](https://arxiv.org/abs/2506.10943), [Website](https://jyopari.github.io/posts/seal)



SEAL (**Se**lf-**A**dapting **L**LMs) is a framework for training language models via RL to generate self-edits (finetuning data and other update directives for themselves) in response to new inputs. 

We explore SEAL in two domains:
- [knowledge-incorporation](knowledge-incorporation): Incorporating new factual knowledge
- [few-shot](few-shot): Adapting to new tasks from few-shot examples

Both folders include code, data, and documentation.

## 🔧 Setup

### 1. Clone the repository

```bash
git clone https://github.com/Continual-Intelligence/SEAL.git
cd SEAL
```

### 2. Set up a virtual environment

Using **conda**:

```bash
conda create -n seal_env python=3.12
conda activate seal_env
```

Using **venv**:

```bash
python3.12 -m venv seal_env
source seal_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a `.env` file in the project root and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```
