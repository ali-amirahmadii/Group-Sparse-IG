# Group-Sparse Manifold-Aware Integrated Gradients (GS-IG)


> **Group-Sparse Manifold-Aware Integrated Gradients for Multimodal Transformers on EHR Trajectories**  
> Ali Amirahmadi, Farzaneh Etminani, Mattias Ohlsson  
> *Proceedings of Machine Learning Research (PMLR) Vol. 297, ML4H 2025*  

[📄 PDF (OpenReview)](https://openreview.net/pdf?id=gYLChYGRA6)

---

## Overview

Integrated Gradients (IG) is a popular attribution method for explaining deep models, including multimodal Transformers on electronic health records (EHR). However, standard IG is problematic for discrete token sequences:

1. **No principled baseline** for “absence of evidence” in embedding space.  
2. **Dense, noisy attributions** that are hard for clinicians to interpret.

GS-IG addresses both issues:

1. **Manifold-aware baseline**  
   - We compute a *position-wise empirical mean* of token embeddings over the validation set.  
   - This keeps IG’s interpolation path near the model’s embedding distribution, instead of traversing synthetic, low-density regions (e.g., <MASK>, zero vectors).

2. **Group-sparse, path-optimized IG (GS-IG)**  
   - We keep the straight-line IG path in embedding space but reparameterize it via a schedule  
     \[
       \alpha(t) = t^{\theta}
     \]  
   - For each input, we choose \(\theta\) to minimize a **token-level \(\ell_{2,1}\) group-sparsity objective**, treating each token embedding as a group.  
   - This yields **token-level**, not dimension-level, attributions that are much more concise and practitioner-friendly.

On **MIMIC-IV (incident heart failure)** and **Malmö Diet and Cancer (MDC, early mortality)**:

- The **manifold-aware baseline** improves faithfulness (↑ Comprehensiveness, ↓ Sufficiency).  
- **GS-IG** maintains similar faithfulness while reducing token-level \(\ell_{2,1}\) by ≈ 9–18%, producing sparse and actionable explanations.

---

## What’s in this repository?

- `attributions_gsig.py`  
  Core implementation of **Group-Sparse Manifold-Aware Integrated Gradients**:
  - Construction of a **manifold-aware baseline** for sequence embeddings.
  - **Group-sparse path optimization** over the IG schedule \(\alpha(t) = t^{\theta}\).
  - Utilities to compute token-level attributions for multimodal Transformer models on EHR trajectories.

> **Note:** This repo currently focuses on the attribution routine itself, not on full training/evaluation pipelines for MIMIC-IV or MDC. You can plug GS-IG into your own PyTorch-style model code following the high-level workflow below.

---

## Method in a Nutshell

1. **Train your model**  
   - Any model with an embedding layer for discrete codes (e.g., diagnoses, procedures, medications) and possibly continuous modalities (age, labs, etc.).
   - Typical use case: multimodal Transformer on EHR visit sequences.

2. **Compute manifold-aware baselines**  
   - Use the validation set to compute the *position-wise mean* embedding:
     - For each sequence position \(j\), average token embeddings across validation trajectories at that position.
     - This yields a “typical” patient trajectory in embedding space.

3. **Run GS-IG**  
   - For a given patient trajectory \(x\), interpolate between:
     - Baseline embeddings (the manifold-aware mean) and  
     - The actual embeddings for that patient.
   - For each candidate \(\theta\), approximate IG along the scheduled straight path \(\alpha(t) = t^{\theta}\).
   - Choose \(\theta\) that minimizes a token-level \(\ell_{2,1}\) penalty, encouraging entire tokens to “turn on or off” together.



---

## High-Level Usage (Pseudo-code)

> **This is illustrative pseudo-code.**  

```python
# High-level pseudo-code – adapt to your actual API in attributions_gsig.py

from attributions_gsig import (
    build_manifold_baseline,   # e.g., from a validation loader
    gs_ig_attributions         # group-sparse IG routine
)

# 1. Compute manifold-aware baseline using validation data
baseline = build_manifold_baseline(
    model=model,
    val_loader=val_loader,     # iterator over validation patients
    device=device,
    # additional args as defined in attributions_gsig.py
)

# 2. Explain a single patient trajectory x
attributions = gs_ig_attributions(
    model=model,
    inputs=batch_x,            # patient trajectory (tokens + other modalities)
    baseline=baseline,
    target=target_index,       # class index / output component to explain
    num_steps=128,             # IG steps along the path
    # additional hyperparameters (e.g., theta search range, sparsity weight)
)

# 3. Aggregate & visualize
token_scores = attributions["tokens"]    # per-token attributions
visit_scores = attributions.get("visits", None)
