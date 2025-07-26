# (unsupervised) Self-Adapting Language Models

Original SEAL Paper and website: [Paper](https://arxiv.org/abs/2506.10943), [Website](https://jyopari.github.io/posts/seal)



SEAL (**Se**lf-**A**dapting **L**LMs) is a framework for training language models via RL to generate self-edits (finetuning data and other update directives for themselves) in response to new inputs. 

This project aims to make the training process of SEAL entirely unsupervised.
This is done by making question generation and response grading perfomed by the model in-context, rather than using human generated questions and OpenAI grading.


The original SEAL paper addressed two domains, [knowledge-incorporation](knowledge-incorporation) and few-shot learning; this project looks at only the former.
