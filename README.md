# CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidance
<img src="./Figures/Tag.png" width="700px" alt="s" />

[Huggingface🤗](https://huggingface.co/yongchao98/CodeSteer-v1)
[Model Weights](https://drive.google.com/drive/folders/1qb_rec6f8rMYtFKm0eQpad0L0uHCwgpL?usp=share_link)
[Finetune Datasets](https://drive.google.com/drive/folders/1Byn-99gFd5ckRkPMJ8-zagzW7XDfO8ie?usp=share_link)

## Framework
<img src="./Figures/CodeSteer-intro.png" width="800px" alt="s" />

<p align="center" style="font-size: 16px;">
Figure: CodeSteer on guiding LLM code/text generation to integrate symbolic computing. At each interaction with TaskLLM, it reviews current and previous answers, then provides guidance for the next round.
</p>

## Inspirations
<img src="./Figures/LLM-makes-simple-mistakes-gather.png" width="800px" alt="s" />
<p align="center" style="font-size: 16px;">
Figure: The cases that GPT-4o makes simple mistakes by direct textual reasoning but can reliably solve the problem with prompted to use code.
</p>


## Performance
We compare GPT-4o + CodeSteer with OpenAI o1 and DeepSeek R1 on SymBench, with 28 seen tasks and 9 unseen tasks. GPT-4o + CodeSteer surpasses o1 (82.7), R1 (76.8), and o1-preview (74.8), highlighting the importance of integrating symbolic computing into LLMs.

<img src="./Figures/Table-results.png" width="800px" alt="s" />

The cost of tokens and runtimes for each method are as follows. GPT-4o + CodeSteer costs less tokens and runtimes than o1 and R1.
<img src="./Figures/Cost-token-runtime.png" width="800px" alt="s" />

## Environment Setup
The fine-tuning and inference of CodeSteerLLM is based on [Llama-factory](https://github.com/hiyouga/LLaMA-Factory).
```
git clone git@github.com:yongchao98/CodeSteer-v1.0.git
cd CodeSteer-v1.0

conda create -n CodeSteer python=3.10
conda activate CodeSteer
pip install accelerate==1.0.1
pip install sympy
pip install python-constraint
pip install pandas
pip install anthropic
pip install mistralai
pip install openai
pip install tiktoken
pip install python-dotenv
pip install datasets==3.1.0
pip install peft==0.12.0
pip install trl==0.9.6
pip install transformers==4.46.1
```

## LLM API Key Setup
If you want to use several API-based LLMs as TaskLLM or CodeSteerLLM, then you need to set up API key.

1. First, create a .env file in your project root:
```
OPENAI_API_KEY='your_key_here'
CLAUDE_API_KEY='your_key_here'
MIXTRAL_API_KEY='your_key_here'
DEEPSEEK_API_KEY='your_key_here'
```
2. Add this .env file to your .gitignore to prevent accidentally committing it:
```
echo ".env" >> .gitignore
```

## Train Models

### Create test samples
The synthesized test samples for 37 tasks are in [dataset_gather](https://github.com/yongchao98/CodeSteer-v1.0/tree/main/dataset_gather) dictionary. You can also synthezise the samples by yourself with tunable complexities with scripts in [create_dataset](https://github.com/yongchao98/CodeSteer-v1.0/tree/main/create_dataset).

### Run inference without GPU, Close LLM as CodeSteerLLM
We can directly use unfinetuned model like GPT-4o as CodeSteerLLM, then directly run
```
python benchmark_test_baseline.py
```

### Run inference with GPU, finetuned CodeSteerLLM
```bash
bash ./infer_CodeSteer.sh
# default config file is ./llama3_8B_CodeSteer.yaml using the model uploaded on Huggingface.
```

### Finetuning CodeSteerLLM with GPU
For both SFT and DPO

## Feedback

We appreciate all feedback! Feel free to raise an issue for bugs, questions, or suggestions.

## Assistance

Contacting [Yongchao Chen](https://yongchao98.github.io/YongchaoChen/) and [Chuchu Fan](https://chuchu.mit.edu) for any questions and discussion.

## Citation
```md
@article{chen2024steering,
  title={Steering Large Language Models between Code Execution and Textual Reasoning},
  author={Chen, Yongchao and Jhamtani, Harsh and Sharma, Srinagesh and Fan, Chuchu and Wang, Chi},
  journal={arXiv preprint arXiv:2410.03524},
  year={2024}
}
```
