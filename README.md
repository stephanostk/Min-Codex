# MinCodex

A fine-tuned, locally-deployable coding assistant built on **Qwen2.5-Coder-3B-Instruct**, specialized for Python code completion. Trained using LoRA on a free-tier GPU and deployed fully offline via Ollama.

## Overview

| | |
|---|---|
| **Model name** | MinCodex |
| **Base model** | [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct) (Apache 2.0) |
| **Fine-tuning method** | LoRA (rank 16) via [Unsloth](https://github.com/unslothai/unsloth) |
| **Dataset** | [iamtarun/python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) (~18K examples) |
| **Training hardware** | Kaggle free-tier GPU (Tesla T4, 16GB) |
| **Deployment format** | GGUF (Q4_K_M quantization) |
| **Runtime** | [Ollama](https://ollama.com) — fully offline, no cloud dependency |
| **Model weights** | [huggingface.co/your-username/mincodex](#) *(update with your actual HF repo link)* |

## Specialization

MinCodex was built around a specific use case, not general-purpose chat:

- **Domain:** Coding (Python)
- **Task:** Code completion and generation
- **Style:** Structured, formal explanations for generated code *(style fine-tuning in progress)*
- **Language:** English
- **Device target:** Laptop / developer workstation

## Why a Small Language Model?

Rather than relying on a large general-purpose model, MinCodex fine-tunes a compact 3B-parameter model specialized for one task. The result runs entirely on consumer hardware (a laptop CPU/GPU) with no API costs and no internet dependency at inference time — while still producing accurate, coherent code completions for its target domain.

## Training Pipeline

1. **Environment setup** — Unsloth + Transformers + TRL + PEFT + bitsandbytes on a Kaggle GPU notebook
2. **Dataset preparation** — loaded the instruction dataset and reformatted it into Qwen's chat template (system / user / assistant turns)
3. **LoRA fine-tuning** — attached rank-16 LoRA adapters to attention and MLP projection layers, trained via `SFTTrainer` for 1 epoch
4. **Export** — merged and converted the fine-tuned model to GGUF format with 4-bit quantization for efficient local inference
5. **Deployment** — registered the GGUF file with Ollama for offline, terminal-based inference

## Example Output

**Prompt:**
```python
def is_prime(n):
    """Return True if n is prime."""
```

**MinCodex completion:**
```python
def is_prime(n):
    """Return True if n is prime."""
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def generate_primes(limit):
    """Generate a list of prime numbers up to a given limit."""
    primes = []
    for num in range(2, limit + 1):
        if is_prime(num):
            primes.append(num)
    return primes
```

## Running It Locally

1. Download the GGUF file from the [Hugging Face model repo](https://huggingface.co/Utkarshghogale/mincodex).
2. Create a file named `Modelfile` in the same folder:
   ```
   FROM ./mincodex.Q4_K_M.gguf
   ```
3. Register the model with Ollama:
   ```bash
   ollama create mincodex -f Modelfile
   ```
4. Run it:
   ```bash
   ollama run mincodex
   ```

## Notable Engineering Challenges

- **Multi-GPU device mismatch:** Kaggle's dual-T4 setup caused a `RuntimeError` from tensors landing on different CUDA devices during inference. Resolved by pinning execution to a single GPU via `CUDA_VISIBLE_DEVICES`.
- **Dependency conflicts:** An unnecessary `pip install --upgrade` step broke Unsloth's pinned dependency versions (`transformers`, `trl`, `datasets`). Resolved by relying solely on Unsloth's own installer to manage compatible versions.

## Repository Contents

- `training_notebook.ipynb` — full Kaggle training pipeline (environment setup, dataset formatting, LoRA fine-tuning, GGUF export)
- `README.md` — this file

## Roadmap

- [ ] Second fine-tuning pass for formal, report-style code documentation/explanation
- [ ] Expand evaluation using code-completion benchmarks (HumanEval / MBPP)

## License

This project builds on Qwen2.5-Coder-3B-Instruct (Apache 2.0). Check the base model's license terms for downstream use.
