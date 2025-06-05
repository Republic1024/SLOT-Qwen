

# SLOT-Qwen 🚀  
**Sample-specific Generation via Hidden-State Perturbation (Based on Qwen3-0.6B)**

This repository implements a simplified and practical version of the **SLOT** method — originally proposed in  
📄 **"SLOT: Sample-specific Language Model Optimization at Test-time"**  
by *Yang Hu, Xingyu Zhang, Xueji Fang, Zhiyang Chen, Xiao Wang, Huatian Zhang, Guojun Qi*,  
presented by Westlake University, University of Washington, and USTC.

> This implementation adapts the SLOT idea to the [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) model, demonstrating how small test-time optimization on hidden states can guide controllable generation without updating model parameters.

---

## 🔍 What is SLOT?

SLOT (Sample-specific Language Model Optimization at Test-time) proposes a lightweight test-time adaptation strategy for language models:

- A small perturbation vector `delta` is trained on the prompt itself;
- This delta is added to the hidden states (e.g., last transformer layer) during generation;
- No fine-tuning of model weights is required;
- Works for arbitrary prompts, supports fast and local adaptation.

---

## ✨ Key Features

- **Parameter-free generation**: Frozen LLM, no weight updates.
- **Prompt-specific control**: Each prompt has its own delta.
- **Efficient**: Delta can be trained in just 3–10 steps.
- **Plug-and-play**: No model modification required.
- **Supports Qwen3-0.6B (Huggingface / ModelScope)**

---

## 🛠️ Quick Start

### 1. Clone the repo and install dependencies
```bash
git clone https://github.com/repubic1024/SLOT-Qwen.git
cd SLOT-Qwen
pip install -r requirements.txt
```

### 2. Load Qwen3-0.6B

#### 🔹 From Huggingface

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").cuda()
```

#### 🔹 From ModelScope

```python
from modelscope import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen3-0.6B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen3-0.6B", trust_remote_code=True).cuda()
```

---

## 🚀 How It Works

1. Encode prompt and get hidden states from last transformer layer.
2. Train a small delta vector `delta ∈ ℝ¹ˣ¹ˣᴴ` on next-token prediction.
3. Add delta to hidden state `H[:, :-1, :]` and optimize using cross-entropy loss.
4. During generation, apply delta to `H[:, -1, :]` at each decoding step.

```python
# Train delta (step=3)
delta, loss_log = train_delta_from_H(model, tokenizer, prompt, step=3)

# Generate with delta
output = generate_with_delta(model, tokenizer, prompt, delta, max_new_tokens=200)
```

---

## 🔬 Experiment: With vs. Without Delta

| Setting            | Generation Output (excerpt)                                                           |
| ------------------ | ------------------------------------------------------------------------------------- |
| ❌ No delta         | "The company's strategy remained vague and generic..."                                |
| ✅ delta (3 steps)  | "The company's strategy emphasizes sustainable growth and clean energy..."            |
| ✅ delta (10 steps) | "The new policy outlines a roadmap for carbon neutrality and green infrastructure..." |

---

## 📂 Project Structure

```
SLOT-Qwen/
├── delta.pt                  # 保存的 delta 向量（引导提示用）
├── LICENSE                   # 开源协议
├── preprocess.py             # 日志预处理与 delta 构造脚本
├── README.md                 # 项目说明文档
├── requirements.txt          # Python 依赖列表
├── SLOT-Qwen3_final.html     # Jupyter Notebook 的 HTML 导出版本（用于展示）
├── SLOT-Qwen3_final.ipynb    # 主实验 Notebook，包含完整实验流程
├── SLOT_Paper.pdf            # SLOT 原始论文（参考用）

```

---

## 📄 Reference

This implementation is based on the following paper:

> **SLOT: Sample-specific Language Model Optimization at Test-time**
> Yang Hu, Xingyu Zhang, Xueji Fang, Zhiyang Chen, Xiao Wang, Huatian Zhang, Guojun Qi
> \[arXiv Link (TBD)]
> Affiliations: Westlake University, University of Washington, USTC

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

This implementation is maintained by Republic, building on the original SLOT idea with integration into the Qwen3-0.6B model.



