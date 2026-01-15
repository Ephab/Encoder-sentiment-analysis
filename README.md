# Encoder-Based Sentiment Analysis

A custom Transformer encoder implementation for sentiment classification, built from scratch using PyTorch and trained on a diverse combination of datasets.

## Overview

This project implements a **Transformer encoder** for binary sentiment classification (positive/negative). Rather than using pre-trained models like BERT, this architecture is built from the ground up to understand the fundamental mechanics of Transformers.

**Highlights:**
- Custom Transformer encoder with 6 layers, 8 attention heads
- Trained on **187k+ reviews** from multiple sources (IMDB, SST-2, Stanford SST-2)
- Achieves high accuracy with just **104-dimensional embeddings** (vs 768 in BERT)

---

## Architecture & Design Decisions

### 1. **Positional Encoding**
```python
class PositionalEncoding(nn.Module):
    # sinusoidal encoding based on "Attention Is All You Need"
```

**Why this approach?**
- Implements the original sine/cosine positional encoding from the Transformer paper
- Generalizes well to sequences of varying lengths

---

### 2. **Compact Embedding Dimension (104)**
```python
EMBEDDING_DIM = 104  # vs BERT's 768
```

**Rationale:**
- **Efficiency:** Much less computation needed
- **Sufficient expressiveness:** Sentiment analysis is usually a not so complex task, so a dim of 104 is sufficient to capture important relationships
- **Faster training/inference:** Smaller matrices mean faster computations and training
- **Trade-off:** May not capture nuanced semantic relationships as well as larger models

---

### 3. **Sequence Length Constraint (200 tokens)**
```python
MAX_LENGTH = 200
```

**Why 200?**
- Covers the majority of the examples in the training data (see EDA visualizations)
- Reduces computational cost
- Faster training and inference

---

### 4. **Encoder Architecture**
```python
self.encoder_stack = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
```

**Configuration:**
- **Layers:** 6 encoder layers
- **Attention heads:** 8 (each one capturing different subtle relationships)
- **Feedforward dimension:** 2048 (standard size)

---

### 5. **Mixed Precision Training**
```python
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    output = model(X, mask)
    loss = loss_fn(output, y)
```

**Benefits:**
- Much faster training with T4 GPU on Kaggle
- Reduced memory usage
- No noticable accuracy decrease



## Dataset Strategy

### Multi-Source Training Data
```python
train_mix = concatenate_datasets([
    imdb["train"],        # 25k long-form reviews
    sst2["train"],        # ~67k short phrases  
    sst2_alt["train"],    # ~70k phrases (Stanford NLP)
])
```

**Total:** ~162k training samples

**Rationale for mixing datasets:**
1. **Diversity:** Combines long reviews (IMDB) with short phrases (SST-2)
2. **Robustness:** Model learns sentiment across different text lengths and styles
3. **Generalization:** Reduces overfitting
4. **Class balance:** Datasets are balanced (approx 50/50 positive/negative)

## Training Details

### Optimizer Configuration
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)
```

### Loss Function
```python
loss_fn = nn.CrossEntropyLoss()
```
Standard choice for multi-class classification (Binary Cross Entropy could work just as well here too)

## Usage

### Training
```python
# see encoder_sentiment_analysis.ipynb
```

### Inference
```python
# see encoder-sentiment-inference.ipynb
```

### Portable Model Bundle
- Contains all the model parameters and information to be loaded for inference

---

## Project Structure

```
Encoder-sentiment-analysis/
├── encoder_sentiment_analysis.ipynb   # Training + EDA
├── encoder-sentiment-inference.ipynb  # inference demo
└── custom_encoder_portable_model.pth  # Trained model weights
```
---

## Requirements

```bash
pip install torch transformers datasets torchinfo
```
