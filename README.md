# LLM Fine-Tuning Tutorial

This repository provides a step-by-step tutorial on fine-tuning large language models (LLMs). It is designed for students, researchers, and practitioners who want to understand the workflow of adapting pre-trained models to downstream tasks, from supervised fine-tuning to preference alignment.

---

## 📘 Contents

- **`LLM_Fine_Tuning_Tutorial.ipynb`**: supervised fine-tuning with Hugging Face Transformers and PEFT
  - Full fine-tuning of DistilBERT for sentiment classification (IMDB)
  - Instruction tuning of DistilGPT2 (loss on the response only)
  - Parameter-efficient fine-tuning: **LoRA** (and 4-bit QLoRA on a GPU), **Prefix Tuning**, **BitFit**
  - Generations before vs after tuning, and a comparison table (trainable parameters, metric, training time)
  - Saving and loading models and adapters

- **`Alignment-and-RLHF/Alignment_and_RLHF_Tutorial.ipynb`**: preference alignment and RL post-training with Hugging Face TRL
  - **DPO** (Direct Preference Optimization) with `DPOTrainer`, on chosen/rejected pairs
  - **PPO** (classic RLHF) implemented from scratch: policy, reference, reward model, and value head
  - **GRPO** (Group Relative Policy Optimization) with `GRPOTrainer` and a Python reward function
  - All three steer the same model on the same task, compared by reward, KL to the reference, and training time

- **Key Concepts Covered**
  - Transfer learning in NLP
  - Hugging Face Transformers `Trainer` workflow
  - Custom dataset integration and label masking for instruction tuning
  - Reward models, KL regularization, and reward hacking
  - Model saving and loading

Both notebooks use small models (DistilBERT, DistilGPT2, GPT-2) so every section runs on a CPU; a GPU is much faster.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/karimiannima/LLM-Fine-Tuning-Step-by-Step-Tutorial.git
cd LLM-Fine-Tuning-Step-by-Step-Tutorial
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
For 4-bit QLoRA on a CUDA GPU, also `pip install bitsandbytes`.

### 4. Open a notebook
```bash
jupyter notebook LLM_Fine_Tuning_Tutorial.ipynb
```

---

## 🛠 Requirements

- Python ≥ 3.10
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/) ≥ 4.46
- [Datasets](https://huggingface.co/docs/datasets/), [Accelerate](https://huggingface.co/docs/accelerate/)
- [PEFT](https://huggingface.co/docs/peft/)
- [TRL](https://huggingface.co/docs/trl/) ≥ 1.0 (alignment notebook)
- scikit-learn, pandas, matplotlib, Jupyter

The notebooks were last run with PyTorch 2.10 (CPU), Transformers 5.20, PEFT 0.21, TRL 1.16, and Datasets 5.1.

---

## 📊 Results

The tutorials demonstrate:
- Per-epoch validation accuracy/F1 (classification) and response loss (instruction tuning)
- Generations from the untuned and tuned models on held-out instructions
- Comparison of full fine-tuning vs LoRA vs Prefix Tuning vs BitFit (trainable parameters, metric, wall-clock time)
- Comparison of DPO vs PPO vs GRPO (reward, KL to the reference model, wall-clock time)

---

## 🤝 Contributing

Contributions are welcome!
If you’d like to add new examples, fix issues, or extend the tutorial (e.g., quantization, reward-model training, or larger models), please open a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
