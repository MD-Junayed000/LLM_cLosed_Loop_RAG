# Closed-Loop RAG — Cross-Model Findings (Single-Layer Probe)

All experiments use the RAGTruth-18K dataset (stratified 80/20 train/val split,
n_val = 3,058) on an NVIDIA A100 (Google Colab), FP16, no quantization. Each model
uses a single hidden-state hook layer and a 2-layer MLP probe that fuses the
Context-Echo Vector (CEV, residual stream) and Internal-Activation Vector (IAV, MLP
activations).

## Cross-Model Summary Table

| Metric | Mistral-7B-Instruct-v0.2 | Qwen3-8B | LLaMA-3-8B-Instruct |
|--------|:------------------------:|:--------:|:-------------------:|
| Fused Val AUROC | 0.8596 | 0.8808 | 0.8689 |
| CEV AUROC | 0.8474 | 0.8763 | 0.8579 |
| IAV AUROC | 0.8474 | 0.8709 | 0.8622 |
| Mean Fusion Accuracy | 78.1% | 80.0% | 78.2% |
| HaluEval Mean AUROC (OOD) | 0.105* | 0.699 | 0.814 |
| Vanilla Mean Proxy | 0.3301 | 0.5388 | 0.3321 |
| Closed-Loop Mean Proxy | 0.1068 | 0.1116 | 0.1824 |
| Hallucination Reduction | 67.6% | 79.3% | 45.1% |
| Acceptance Rate | 62% | 42% | 90% |
| F1-Optimal Threshold | 0.28 | 0.42 | 0.06 → 0.60 (fallback) |
| Temperature (CEV/IAV) | 1.60 / 0.80 | 0.90 / 1.00 | 1.30 / 1.10 |
| Fusion Weight (CEV/IAV) | 0.50 / 0.50 | 0.60 / 0.40 | 0.45 / 0.55 |

\* Mistral's HaluEval AUROC (0.105) is polarity-inverted; the effective discriminative
power is approximately 1 − 0.105 = 0.895, but the direction is flipped.

## Mistral Findings (Key Contribution)

**1. Base v0.1 → Instruct-v0.2 switch was decisive.**

- Base v0.1: threshold collapsed (0.13), MCQ-format hallucinations, invented facts, ~2%
  acceptance, HaluEval ~0.035 (inverted).
- Instruct-v0.2: healthy threshold 0.28, 9/9 coherent answers, honest refusals, 62%
  acceptance, 67.6% reduction.

**2. Persistent HaluEval cross-domain gap (0.105, still inverted).**
Switching to the instruct checkpoint fixed format hallucinations, threshold calibration,
in-distribution AUROC (0.86), and closed-loop effectiveness — but NOT HaluEval transfer.
Conclusion: the transfer gap is an architectural / representational property of Mistral,
not a lack of instruction tuning. It is unique to Mistral (Llama 0.814, Qwen 0.699 both
transfer well).

**3. Temperature diagnostic.**
T_cev = 1.60 (the highest — extremely overconfident CEV) combined with T_iav = 0.80
(under-confident IAV) is a unique signature that partly explains the compressed
single-layer fused scores.

**4. Health-check fallback.**
`if f1_optimal < 0.25: use deployment threshold`. Mistral-Instruct (0.28) passes without
the fallback; Llama single-layer (0.06) triggers it; Qwen (0.42) never triggers.

## Llama Findings

- **Best HaluEval OOD transfer (0.814)** of the three backbones, with correct (non-inverted)
  polarity — internal-state hallucination detection generalises beyond RAGTruth.
- **Strong in-distribution probe**: fused AUROC 0.8689, mean accuracy 78.2%; IAV (0.8622)
  edges out CEV (0.8579), and the fusion weight w = 0.45 leans on IAV (55%).
- **Single-layer threshold collapse**: the raw val F1-optimal threshold fell to 0.06 because
  the single-layer val scores are compressed under domain shift. The health-check fallback
  restores deployment thresholds (0.60 / 0.78), and a Platt calibration (a = 5.7142,
  b = −2.9139) maps the fused score to a calibrated P(hallucination).
- **Highest answer utility (90% acceptance, 10% abstention)** — the least conservative model,
  while still reducing the delivered hallucination proxy by 45.1% (0.3321 → 0.1824).

## Qwen Findings

- **Best in-distribution separability**: fused AUROC 0.8808, mean accuracy 80.0%, best CEV
  (0.8763) and IAV (0.8709) of the three models.
- **Positive, well-calibrated OOD transfer (0.699)** — weaker than Llama (0.814) but, unlike
  Mistral, not polarity-inverted.
- **Most conservative deployment**: 42% acceptance (58% abstention) at the F1-optimal
  threshold of 0.42, achieving the largest relative hallucination reduction (79.3%,
  0.5388 → 0.1116). The reference run used T_cev = 0.90 / T_iav = 1.00 and fusion weight
  w = 0.60.

## Single-Layer vs Triple-Layer Analysis

The pipeline now trains a **single mid-depth hook layer (N/2)** instead of the earlier
early/mid/late 3-layer concatenation. The cross-model layer-wise probe ablation showed the
single mid layer is the strongest single-layer choice and substantially reduces probe
parameters and feature-extraction cost. The trade-off is **compressed fused score
distributions** for some models (most visibly Llama, whose raw F1-optimal threshold
collapses to 0.06), which is handled by the health-check fallback and Platt calibration
rather than by reverting to the heavier triple-layer features.

## Platt Calibration Analysis

Platt scaling (`P_calibrated = sigmoid(a * fused + b)`) is fitted on the validation fused
scores to map raw probe outputs to calibrated hallucination probabilities. It is most
important for Llama single-layer, where the fitted parameters (a = 5.7142, b = −2.9139)
re-expand the compressed score range so that the deployment thresholds (0.60 / 0.78)
behave sensibly. Mistral-Instruct (threshold 0.28) and Qwen (0.42) have healthy raw score
distributions and rely less on re-calibration.

## Safety vs Utility Recommendations

- **Maximum safety / lowest delivered hallucination**: Mistral-7B-Instruct-v0.2
  (closed-loop proxy 0.1068) — balanced at 62% acceptance.
- **Most conservative**: Qwen3-8B (42% acceptance, 79.3% reduction) — prefer when abstention
  is cheap and false answers are costly.
- **Maximum utility**: LLaMA-3-8B-Instruct (90% acceptance) with the best OOD transfer —
  prefer when answering as many queries as possible matters, accepting a higher delivered
  proxy (0.1824).

## Known Limitations

- HaluEval transfer is model-specific; Mistral's hidden states remain polarity-inverted
  out-of-distribution even after full instruction tuning.
- Single-layer probes can produce compressed score distributions, requiring health-check
  fallback and Platt calibration to keep deployment thresholds meaningful.
- Acceptance rates and the nine-query demo are computed on small evaluation sets (100-query
  baseline, 9-query qualitative demo) and are directional rather than definitive.
- Proxy-based hallucination scoring is an automated approximation, not a human-judged
  hallucination rate.
