# 🤖 Lightweight Optimization for LLM Services

## Overview
This project focuses on applying **lightweight techniques** to improve the training speed and memory efficiency of LLM (Large Language Model) services.  
The training data used for this project is stored in `corpus.json`.

---

## 📂 Key Files

The experiments consist of four Python files:

- `before_small.py` : Code for the `facebook/opt-350m` model without lightweight techniques.
- `lightweight_small.py` : Code for the `facebook/opt-350m` model with lightweight techniques applied.
- `before_big.py` : Code for the `EleutherAI/gpt-neo-1.3B` model without lightweight techniques.
- `lightweight_big.py` : Code for the `EleutherAI/gpt-neo-1.3B` model with lightweight techniques applied.

Files with `before` refer to models without optimizations, while `lightweight` refers to models with optimizations. The `small` files use a smaller model (`350M`), and the `big` files use a larger model (`1.3B`).

---

## 🧪 Experiments and Results

### 1. Small Model Experiments (`facebook/opt-350m`)

We first experimented with the smaller model (`350M`) to observe the effects of lightweight techniques. We compared the training times by running `before_small.py` and `lightweight_small.py`.

- **Training time without optimization**: 2.67 minutes  
  ![image](https://github.com/user-attachments/assets/8965bcb3-a7a7-44e0-aba5-139233f55fa2)

- **Training time with optimization**: 0.72 minutes (Approximately 3x improvement 🚀)  
  ![image](https://github.com/user-attachments/assets/ec223c43-d8b2-42dc-8c36-ef62d5802be1)

The lightweight techniques resulted in a **3x improvement in training speed**, while the `train/loss` graph showed consistent and stable loss reduction without performance degradation.

Both before and after applying lightweight techniques, the `loss` consistently decreased.

- **Loss trend without optimization**:  
  ![image](https://github.com/user-attachments/assets/e1373019-e4c0-4ab7-abea-efab81687385)

- **Loss trend with optimization**:  
  ![image](https://github.com/user-attachments/assets/554b59eb-684c-41b5-9d70-219d9321151d)

---

### 2. Large Model Experiments (`EleutherAI/gpt-neo-1.3B`)

For the larger model (`1.3B` parameters), attempts to run the unoptimized file (`before_big.py`) resulted in a **CUDA memory error**, preventing training.  
![image](https://github.com/user-attachments/assets/eafb76a7-8fd6-4ada-9eb7-4a3e1dd757c9)

To address this, we applied lightweight techniques in `lightweight_big.py`, allowing the GPU memory to be used efficiently.

#### Optimization Techniques Applied:
- **4-bit Quantization**: Reduced model parameters to 4-bit precision to lower memory consumption.
- **LoRA (Low-Rank Adaptation)**: Introduced trainable parameters to specific layers for efficient adaptation.

With these optimizations, training was successfully executed, demonstrating significant memory efficiency improvements.  
![image](https://github.com/user-attachments/assets/1375ea42-d610-41ac-ba0d-1e462459bd49)

---

## 💡 Key Benefits

1. **Improved Training Speed**: For the smaller model (`350M`), training became approximately 3x faster after applying lightweight techniques.  
2. **Enhanced Memory Efficiency**: The larger model (`1.3B`), which was previously untrainable due to memory limitations, was successfully trained using quantization and LoRA.

These results demonstrate the necessity of **lightweight techniques** for training large models, especially in constrained hardware environments. The experiments highlight the potential to leverage larger models even on limited resources.
