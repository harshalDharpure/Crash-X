# CrashX — Complete Guide Presentation Briefing

**Use this file to explain the full research to your guide.**  
It covers: problem, motivation, dataset, method, parameters, metrics, all result tables, conclusions, limitations, and how we compare to prior work.

**Project:** CrashX (Crash-X)  
**Repo:** https://github.com/harshalDharpure/Crash-X  
**Paper package:** `paper/ieee_crashx/` (Overleaf zip: `paper/CrashX_IEEE_Overleaf.zip`)  
**Working title (recommended):** *CrashX: Fine-Tuning and Evaluating Video Language Models for Traffic Accident Explanation*

---

## 1. One-minute elevator pitch (say this first)

We ask Video-LLMs to **explain** dashcam traffic accidents in free text (who hit whom, why, when), not just classify them.

**Main finding:**  
Fine-tuning (QLoRA) on accident data **fixes omissions** (the model starts saying everything) but **does not fix hallucinations** (about 1 in 5 claims is still wrong). Apparent timestamp accuracy is mostly a **dataset prior**, not true temporal understanding. A popular decode-time fix (Temporal Contrastive Decoding / SEASON-style) **does not help** under paired statistical tests.

So the paper is an **honest evaluation + diagnosis paper**, not a claim that we invented a decoder that solves hallucination.

---

## 2. What problem are we solving?

### 2.1 Task
Given a 5-second dashcam clip, the model must produce:

1. A free-text **Explanation** (causal narrative of the crash)
2. Structured forensic fields (used to *audit* the explanation):
   - Severity
   - Impact point
   - Start / End time of crash window (seconds)
   - Vehicles (colour + type)
   - Weather

### 2.2 Why this matters
- Accident explanation needs **faithfulness**, not just fluent English.
- Zero-shot Video-LLMs look fluent but often **never narrate the crash**, invent details, or miss vehicles/time.
- Prior accident / driving VLM papers mostly do QA, detection, or planning — few separate **omission vs hallucination** on dense explanations.

### 2.3 Two failure modes (core vocabulary)
| Term | Meaning | Example |
|------|---------|---------|
| **Omission** | Model leaves out something that *is* in the ground truth | GT mentions a black truck; model never names any vehicle |
| **Hallucination** | Model asserts something that *contradicts* the ground truth | GT is a black truck; model says “red car” |

These are **not** the same. Fine-tuning can fix one and leave the other.

---

## 3. What we did (pipeline / research steps)

1. **Dataset:** Car Crash Dataset (CCD) — 1,500 five-second dashcam clips + human forensic text annotations.
2. **Split:** Stratified 80/10/10 by severity → **1,198 train / 150 val / 150 test** (seed 42). Video IDs do not overlap.
3. **Zero-shot baselines:** Run 4 open Video-LLMs with the same prompt and 8 frames.
4. **Fine-tune:** QLoRA on Qwen2.5-VL-7B-Instruct → **CrashLogic-7B**.
5. **Decode variants:** Greedy + Temporal Contrastive Decoding (TCD) with reverse/shuffle negatives + full SEASON recipe.
6. **Evaluate** on 150 held-out test videos with:
   - Lexical metrics (BLEU, ROUGE, METEOR, CIDEr, BERTScore)
   - NLI faithfulness
   - Structured omission / hallucination costs (C_O, C_H)
   - Temporal IoU + **constant-window prior baselines**
   - Error taxonomy, no-crash probe, paired Wilcoxon + bootstrap CIs
7. **Rewrite paper** around the honest findings (asymmetry, temporal prior, TCD null/harmful result).
8. **Release** Overleaf IEEE package under `paper/`.

### Where things live in the repo

| Path | What it is |
|------|------------|
| `Car_Crash_Text_Dataset_ground_truth.xlsx` | Ground-truth annotations (1,500 rows) |
| `video1500/` | Raw videos (local only; not on GitHub — too large) |
| `crashx/data/splits/` | `train.jsonl`, `val.jsonl`, `test.jsonl` |
| `crashx/` | Code: data, QLoRA train, SEASON/TCD decode, eval |
| `outputs/crashlogic_7b_lora/` | LoRA adapters (weights often excluded from GitHub by size) |
| `results/` | Predictions, metrics, older paper notes |
| `paper/ieee_crashx/` | Latest IEEE LaTeX paper |
| `paper/CrashX_IEEE_Overleaf.zip` | Upload this to Overleaf |

---

## 4. Dataset (CCD) — explain this clearly

**Source:** Car Crash Dataset (CCD), Bao et al. (nondeterministic accident anticipation line of work).

**Clip properties:**
- ~1,500 clips, **5 seconds**, **10 fps**
- Human labels: severity, impact, vehicles, weather, crash window `[t_s, t_e]`, multi-sentence Explanation (~96 words on test)

**Our split (after dropping 2 unparseable rows → 1,498):**

| Split | Videos | % | No-crash (`n/a`) | Role |
|-------|--------|---|------------------|------|
| Train | 1,198 | 80% | 66 | QLoRA fine-tuning |
| Val | 150 | 10% | 8 | Reserved (unused for selection) |
| Test | 150 | 10% | 9 (6 unambiguous) | All reported metrics |

**No-crash videos:** Near-misses whose reference says no collision occurred. Six unambiguous test clips are our **collision-prior probe** (does the model invent a crash?).

**Temporal structure of GT:** Crash windows are short and concentrated. Most frequent training windows: `[3,4]`s and `[2,3]`s. This is why a **constant-window baseline** can look strong.

---

## 5. Systems compared

### 5.1 Zero-shot Video-LLMs
| System | Size | Role |
|--------|------|------|
| Qwen2.5-VL | 7B | Main backbone (also fine-tuned) |
| Qwen2.5-VL | 3B | Smaller sibling |
| Qwen2-VL | 2B | Older/smaller family |
| LLaVA-NeXT-Video | 7B | Different family |

All get the **same prompt** and **same 8 frames** (≤224 px, no timestamps).

### 5.2 CrashLogic-7B (ours — adaptation)
- Backbone: **Qwen2.5-VL-7B-Instruct**
- Method: **QLoRA** (4-bit + LoRA adapters)
- Trained on 1,198 CCD clips with structured target strings
- Decoded **greedily** (main adapted system)

### 5.3 CrashLogic + TCD / SEASON (decode-time)
Same checkpoint; only decoding changes:
- TCD reverse, α ∈ {0.5, 1.0, 1.5, 2.0}
- TCD shuffle, α = 0.5
- Full SEASON (spatial noise + temporal homogenisation + JSD weights)

### 5.4 Prior-only temporal baselines (critical for honesty)
Constant windows that **ignore the video**:
- `[2,3]`s, `[3,4]`s (training mode), `[2,4]`s  

If a constant window beats the model on tIoU, the model is not truly localising.

---

## 6. Input, prompt, and training target

### Prompt (all systems)
> “Analyze this car crash video spatiotemporally. Provide crash severity, impact point, exact timestamp window, and detailed causal explanation.”

**Important:** The phrase “car crash video” **presupposes a crash**, which encourages fabrication on no-crash clips.

### Frames
- **K = 8** frames, uniform over 5 s (~0.7 s apart)
- Max side **224 px**
- **No absolute timestamps / frame indices / FPS** given to the model  
  → This is the mechanical reason for the **temporal prior**.

### Training target format (CrashLogic)
```
Severity: ... | Impact: ... | Start: ts | End: te | Vehicles: ... | NumVehicles: n | Weather: ... | Explanation: <text>
```

**Consequence:** The model learns to **always emit every field** → omission drops by construction; correctness is not guaranteed.

---

## 7. All training / decoding parameters (and what each means)

| Parameter | Value | Meaning (explain to guide) |
|-----------|-------|----------------------------|
| Backbone | Qwen2.5-VL-7B-Instruct | Pretrained video-language model we adapt |
| Quantisation | 4-bit NF4 (bitsandbytes) | Compress weights so 7B fits on one GPU |
| LoRA rank `r` | 16 | Low-rank adapter size (capacity of fine-tune) |
| LoRA α | 32 | Scaling of LoRA updates |
| LoRA dropout | 0.05 | Regularisation on adapters |
| Where LoRA attaches | LM attention + MLP | We adapt language layers; vision mostly frozen |
| Learning rate | 2×10⁻⁴ | Step size for adapter updates |
| LR schedule | Cosine | LR decays smoothly over training |
| Epochs | 5 | Full passes over train set |
| Batch size | 1 | Videos per step |
| Grad accumulation | 8 | Effective batch ≈ 8 |
| Max sequence length | 768 tokens | Cap on prompt+target length |
| Keyframes | 8 | How many frames the model sees |
| Max side | 224 px | Image resolution budget |
| Hardware | 1× A100-40GB | Training GPU |
| Wall-clock | ~3.4 hours | Training time |
| Final train loss | 3.09 | NLL at end of training |
| Decode | Greedy, max 256 new tokens | No sampling randomness for main results |
| TCD α | 0.5 / 1.0 / 1.5 / 2.0 | Strength of contrast vs negative frames |
| TCD negative | reverse or shuffle | How we break temporal order |
| Seed | 42 | Split + shuffle reproducibility |

### TCD formula (say this simply)
At each token step:

\[
\ell^{\text{TCD}} = (1+\alpha)\,\ell(V) - \alpha\,\ell(V_{\text{neg}})
\]

- `V` = original frame order  
- `V_neg` = reversed (or shuffled) frames  
- Idea: boost tokens that depend on **true order**, suppress tokens that look the same under reverse.

**Why it can fail:** Colour, weather, agent identity look the **same** in reverse → TCD cannot fix those errors. Timing prior also does not need order → TCD cannot remove it.

---

## 8. Metrics — every metric and what it means

### 8.1 Lexical / embedding overlap (wording quality)
| Metric | Direction | What it measures |
|--------|-----------|------------------|
| **BLEU-4** | ↑ | Exact 4-gram overlap with reference |
| **ROUGE-L** | ↑ | Longest common subsequence overlap |
| **METEOR** | ↑ | Alignment with synonyms/stems |
| **CIDEr** | ↑ | TF-IDF weighted n-gram similarity (captioning classic) |
| **BERTScore** | ↑ | Semantic similarity via contextual embeddings |

**Limitation:** High BLEU/ROUGE can still be a **wrong story** that happens to share words.

### 8.2 NLI faithfulness
Cross-encoder: `cross-encoder/nli-deberta-v3-small`  
Premise = GT fields + reference Explanation; Hypothesis = model text.

| Symbol | Meaning |
|--------|---------|
| \(P_e\) | Entailment probability |
| \(P_c\) | Contradiction probability |
| \(P_n\) | Neutral probability |
| **NLI-Score** | \(P_e - P_c\) (↑) |
| **Full-NLI** | Same idea but hypothesis = **entire** model output (not only Explanation field) |

**Caveat:** Zero-shot models often say little → scored as “neutral”, not “wrong”. Adapted models make more specific claims → more contradictions. So NLI-Score alone can **reward vagueness**.

### 8.3 Omission & hallucination costs (central metrics)
Inspired by **ARGUS** (dual cost for captions), but computed on **CCD structured fields** (deterministic, no API judge).

| Symbol | Name | Direction | Meaning |
|--------|------|-----------|---------|
| **Assert.** | Assertion rate | — | Fraction of forensic fields the model actually asserts |
| **C_O / Ocost** | Omission cost | ↓ | How much GT content is missing |
| **C_H / Hcost** | Hallucination cost | ↓ | Of asserted claims, how often they conflict with GT |

**This is the paper’s main evaluation idea.**

### 8.4 Temporal metrics
| Metric | Direction | Meaning |
|--------|-----------|---------|
| **tIoU** | ↑ | Temporal Intersection-over-Union of predicted vs GT crash window |
| **THR@0.25 / THR@0.50** | ↑ | Fraction of videos with tIoU ≥ 0.25 / 0.50 |
| **Spearman ρ (start)** | near 0 bad | Correlation of predicted start vs true start |
| **Constant-window baselines** | — | Dummy systems that always emit the same window |

### 8.5 Statistics
| Test | Why we use it |
|------|----------------|
| Paired bootstrap 95% CI (5,000 resamples) | Uncertainty on mean differences |
| Wilcoxon signed-rank (two-sided) | Nonparametric paired significance |
| Win/Loss counts | How many videos improved / worsened |

**Rule we follow:** If \(p ≥ 0.05\), we do **not** claim improvement.

### 8.6 Error taxonomy (automatic proxies)
| Code | Rough meaning |
|------|----------------|
| No-Narrative | Never describes a crash |
| Under-Specific | Too vague / missing agents |
| Wrong-Agent | Wrong vehicle colour/identity |
| Wrong-Time | Bad crash window |
| Wrong-Cause | Agents OK-ish but cause wrong |
| Fluent-Fabrication | Fluent text that invents events |
| Clean | No code fired |

---

## 9. Inspiration — which papers / ideas we build on

Explain to your guide as **“standing on these works”**:

| Topic | Papers / ideas | What we took |
|-------|----------------|--------------|
| Video-LLM backbones | Qwen2-VL / Qwen2.5-VL, LLaVA-NeXT-Video | Models we evaluate / fine-tune |
| Dataset | CCD (Bao et al.) | Clips + forensic annotations |
| Omission vs hallucination | **ARGUS** | Dual-cost framing |
| Fine-tuning can increase hallucination | Gekhman et al. (finetuning hallucinations) | Interpretation of our asymmetry |
| Contrastive decoding | Li et al. CD; **VCD**; **SEASON** | Decode-time contrast vs a negative view |
| Temporal shortcuts | Single-frame bias papers; TVBench | Why we distrust “temporal” scores |
| Need for time tokens | TimeChat, VTimeLLM, LITA | Explains our temporal-prior failure |
| Accident / driving VLMs | VRU-Accident, MM-AU, DriveLM, etc. | Context; we differ by faithfulness split + prior baselines |
| Faithfulness judges | FactCC / SummaC / DeBERTa NLI | Secondary NLI signal |

**What is new in CrashX (our contribution angle):**
1. Separate **omission vs hallucination** on dense accident explanations with field-level costs.
2. Show adaptation **fixes omission, not hallucination**.
3. Show “good tIoU” can be a **dataset prior** using constant-window baselines + Spearman ρ.
4. Evaluate SEASON-style **TCD with paired tests** → mild TCD ineffective; strong TCD harmful; mechanistic explanation.
5. Release outputs + evaluation protocol.

---

## 10. ALL RESULT TABLES (from the latest paper)

All numbers are on the **150-video test split** unless noted.

### Table A — Explanation quality (lexical)

| System | BLEU-4 ↑ | ROUGE-L ↑ | METEOR ↑ | CIDEr ↑ | BERTScore ↑ |
|--------|----------|-----------|----------|---------|-------------|
| ZS Qwen2.5-VL-7B | 0.016 | 0.160 | 0.203 | 0.338 | 0.486 |
| ZS Qwen2.5-VL-3B | 0.019 | 0.180 | 0.234 | 0.394 | 0.518 |
| ZS Qwen2-VL-2B | 0.018 | 0.120 | 0.132 | 0.221 | 0.418 |
| ZS LLaVA-NeXT-Video-7B | 0.026 | 0.197 | 0.275 | 0.449 | 0.547 |
| **CrashLogic greedy** | **0.142** | 0.336 | **0.371** | 0.508 | **0.686** |
| CrashLogic + TCD α=0.5 | **0.142** | **0.339** | 0.369 | **0.515** | **0.686** |
| CrashLogic + TCD α=1.0 | 0.140 | 0.338 | 0.365 | 0.507 | 0.684 |

**Talking point:** Fine-tuning gives ~**9×** BLEU-4 vs best zero-shot. TCD does not add a significant lexical gain.

---

### Table B — Omission vs hallucination (CENTRAL)

| System | Assert. | C_O ↓ | C_H ↓ |
|--------|---------|-------|-------|
| ZS Qwen2.5-VL-7B | 0.61 | 0.462 [.436,.488] | 0.227 [.188,.268] |
| ZS Qwen2.5-VL-3B | 0.68 | 0.413 | 0.285 |
| ZS Qwen2-VL-2B | 0.33 | 0.607 | 0.564 |
| ZS LLaVA-NeXT-Video-7B | 0.86 | 0.334 | 0.401 |
| **CrashLogic greedy** | **1.00** | **0.107** [.088,.127] | 0.223 [.200,.248] |
| + TCD α=0.5 | 1.00 | 0.102 | 0.212 |
| + TCD α=1.0 | 1.00 | 0.112 | 0.227 |
| + TCD α=1.5 | 1.00 | 0.124 | 0.249 |
| + TCD α=2.0 | 1.00 | 0.128 | 0.253 |
| + TCD shuffle | 1.00 | 0.114 | 0.233 |
| + SEASON full | 1.00 | 0.119 | 0.240 |

**Talking points:**
- Omission: **0.462 → 0.107** (huge, significant, better on **140/150** videos).
- Hallucination: **0.227 → 0.223**, \(p=0.81\) — **not fixed**.
- Strong TCD makes both costs **worse**.

---

### Table C — Temporal grounding + prior baselines

| System | tIoU ↑ | THR@0.25 ↑ | THR@0.50 ↑ |
|--------|--------|------------|------------|
| ZS Qwen2.5-VL-7B | 0.012 | 0.000 | 0.000 |
| ZS Qwen2.5-VL-3B | 0.010 | 0.000 | 0.000 |
| ZS Qwen2-VL-2B | 0.000 | 0.000 | 0.000 |
| ZS LLaVA-NeXT-Video-7B | 0.024 | 0.000 | 0.000 |
| CrashLogic greedy | 0.373 | 0.573 | 0.440 |
| CrashLogic + TCD α=0.5 | 0.394 | 0.600 | 0.473 |
| Constant `[2,3]`s | 0.366 | 0.567 | 0.433 |
| **Constant `[3,4]`s** | **0.408** | 0.593 | 0.473 |
| **Constant `[2,4]`s** | **0.518** | **0.873** | **0.767** |

**Extra facts to say:**
- CrashLogic emits only **two** windows: `[2,3]` (88) and `[3,4]` (53).
- Predicted vs true start Spearman **ρ = 0.009** (\(p=0.91\)) — essentially uncorrelated.
- A constant window **beats** the model → “good tIoU” was misleading without prior baselines.

---

### Table D — Paired statistics (significance)

**ZS-7B → CrashLogic greedy**

| Metric | Δ | p | Verdict |
|--------|---|---|---------|
| BLEU-4 (per-video) | +0.108 | <1e-4 | Significant win |
| ROUGE-L | +0.176 | <1e-4 | Significant win |
| tIoU | +0.360 | <1e-4 | Significant (but see prior!) |
| C_O | −0.356 | <1e-4 | Significant win |
| C_H | −0.005 | 0.81 | **No change** |
| NLI Pe | +0.084 | 0.007 | More entailment |
| NLI Pc | +0.137 | <1e-4 | **More contradiction** |
| NLI-Score | −0.052 | 0.41 | Flat |
| Full-NLI | +0.170 | 0.0005 | Significant win |

**Greedy → TCD α=0.5:** all key metrics **non-significant**.  
**Greedy → TCD α=2.0:** significant **degradations** on ROUGE-L, C_O, C_H, Full-NLI.

---

### Table E — Where hallucinations concentrate (conflict %)

| System | Window | Severity | Colour-disjoint | Weather |
|--------|--------|----------|-----------------|---------|
| ZS Qwen2.5-VL-7B | 26.0 | 9.3 | 14.2 (33) | 14.0 |
| ZS LLaVA-NeXT-Video-7B | 45.3 | 57.3 | 7.1 (20) | 58.7 |
| CrashLogic greedy | 42.7 | 54.0 | 21.3 (22) | 12.0 |
| + TCD α=0.5 | 40.0 | 51.3 | 20.6 (21) | 12.7 |
| + TCD α=2.0 | 44.0 | 60.0 | 29.8 (31) | 12.0 |

**Talking point:** After adaptation, wrong **window** and **severity** dominate C_H. Colour-disjoint ~21% — and mild TCD does **not** fix colour (as predicted).

---

### Table F — Decoding ablation (full)

| Variant | BLEU-4 | ROUGE-L | BERTScore | NLI-Score | Full-NLI | C_H ↓ | C_O ↓ |
|---------|--------|---------|-----------|-----------|----------|-------|-------|
| Greedy | 0.142 | 0.336 | 0.686 | −0.005 | **0.247** | 0.223 | 0.107 |
| TCD α=0.5 rev | 0.142 | 0.339 | 0.686 | 0.057 | 0.234 | **0.212** | **0.102** |
| TCD α=1.0 rev | 0.140 | 0.338 | 0.684 | **0.090** | 0.237 | 0.227 | 0.112 |
| TCD α=1.5 rev | 0.133 | 0.327 | 0.681 | −0.022 | 0.223 | 0.249 | 0.124† |
| TCD α=2.0 rev | 0.133 | 0.319† | 0.676 | 0.016 | 0.148† | 0.253† | 0.128† |
| TCD α=0.5 shuffle | 0.143 | **0.340** | **0.688** | −0.025 | 0.205 | 0.233 | 0.114 |
| SEASON full | **0.147** | 0.332 | 0.685 | 0.064 | 0.204 | 0.240 | 0.119 |

† = significant degradation vs greedy (\(p<0.05\)).  
**No variant is significantly better than greedy on faithfulness.**

---

### Table G — No-crash fabrication (6 unambiguous videos)

| System | Fabricated | Hedged |
|--------|------------|--------|
| ZS Qwen2.5-VL-7B | 3 | 1 |
| ZS Qwen2.5-VL-3B | 5 | 0 |
| ZS Qwen2-VL-2B | 4 | 0 |
| ZS LLaVA-NeXT-Video-7B | 5 | 1 |
| CrashLogic greedy | 3 | 3 |
| TCD α=0.5 reverse | **6** | 0 |
| TCD α=1.0 reverse | **6** | 0 |
| TCD α=2.0 reverse | 1 | 5 |

**Talking point:** Mild TCD makes the collision prior **worse** (fabricates all 6). No system predicts severity `n/a`.

---

### Table H — Error taxonomy (% of 141 crash videos)

| System | No-Narr. | Under-Sp. | Wrong-Ag. | Wrong-Time | Wrong-Cause | Fluent-Fab. | Clean ↑ |
|--------|----------|-----------|-----------|------------|-------------|-------------|---------|
| ZS Qwen2.5-VL-7B | 10.6 | 40.4 | 9.9 | 19.9 | 5.7 | 9.9 | 23.4 |
| CrashLogic greedy | 0.0 | 0.0 | 21.3 | 39.0 | 17.7 | 1.4 | 38.3 |
| TCD α=0.5 | 0.0 | 0.0 | 20.6 | 36.2 | **10.6** | 0.7 | **45.4** |

**Talking point:** Adaptation removes omission-type codes, but increases Wrong-Agent / Wrong-Time because it asserts fields on every video.

---

## 11. Three headline findings (memorise these)

### Finding 1 — Adaptation removes omissions, not hallucinations
- C_O: 0.462 → 0.107 (\(p<10^{-4}\))
- C_H: unchanged (\(p=0.81\))
- Lexical metrics jump strongly
- NLI contradiction mass **rises** (model says more checkable wrong things)

### Finding 2 — Temporal localisation is a dataset prior
- tIoU looks improved (0.012 → 0.373)
- But only 2 predicted windows; ρ≈0 with true start
- Constant `[3,4]` beats the model; `[2,4]` is even higher
- Cause: **no time stamps in input** + supervised Start/End tokens

### Finding 3 — TCD does not fix residual errors
- Mild α=0.5: safe, **no significant gain**
- Strong α≥1.5: **significant harm**
- Mechanism: reverse preserves colour/weather/identity; cannot break a prior that ignores order

---

## 12. “Are we better than other models / papers?” — honest answer

### Where we clearly outperform zero-shot baselines
On **completeness / wording / structured coverage**:
- Best BLEU/ROUGE/BERTScore among compared systems
- Lowest omission cost
- Always emits the expected field format

### Where we are **not** claiming a free win
- Hallucination cost ≈ same as unadapted Qwen2.5-VL-7B
- Timestamps beaten by a **dummy constant window**
- TCD/SEASON **not** a significant faithfulness upgrade here
- We fine-tuned **one** backbone only (limitation)

### How we differ from / improve on related papers (story for guide)
| Compared to | Their focus | Our edge |
|-------------|-------------|----------|
| VRU-Accident / MM-AU style | QA / captions / cause text | Dual omission–hallucination costs + paired stats |
| SEASON / VCD papers | Show decode contrast helps on QA-style hall. benches | Stress-test on accident **generation**; find null/harm + why |
| TimeChat / VTimeLLM | Add time tokens | Diagnose what happens **without** time tokens (prior) |
| Generic caption metrics only | BLEU/ROUGE | Show lexical gains can hide wrong agents |

**Best framing:** We provide a **rigorous evaluation protocol and diagnostic findings** that future methods must beat — not “SOTA decoder that solves hallucination.”

---

## 13. Qualitative examples (for slides)

1. **Video 001227 — adaptation recovers narrative**  
   Zero-shot: markdown / no crash story.  
   CrashLogic: names white oncoming car + impact (better ROUGE), but wrong timestamp window.

2. **Video 000104 — adaptation hallucinates agent**  
   GT: camera car hits **black truck**.  
   CrashLogic: coherent rear-end story but with a **red car** (Wrong-Agent).  
   Lexical metrics still look OK → motivates separate C_H.

3. **Video 000661 — collision prior**  
   GT: “no accidents took place.”  
   Models invent a crash + severity + window.

---

## 14. Conclusions (say at the end)

1. Zero-shot Video-LLMs are fluent but incomplete / unreliable for accident explanation.
2. Domain adaptation (CrashLogic) **solves the format/omission problem**.
3. It does **not** solve faithfulness: wrong colour, wrong severity, fabricated crashes remain.
4. Reported temporal IoU without prior baselines can **mislead**.
5. Reverse-frame contrastive decoding is **not** a general fix for these residual errors.
6. Next steps that make sense: time-aware inputs, no-crash-aware prompts/targets, training that penalises unsupported claims, fine-tune more backbones, human eval.

---

## 15. Limitations (be ready if guide asks)

- Test set n=150 (power limited for small TCD effects)
- Automatic judges (regex field match + NLI); human study unfinished
- 8 frames, no timestamps (causes temporal prior)
- Prompt presupposes a crash
- Single backbone, single seed for adaptation
- No proprietary models (GPT-4o / Gemini)
- SEASON implementation is recipe-faithful but not a perfect reproduction of the original system internals

---

## 16. How to run / reproduce (if asked)

```bash
# Install
pip install -r crashx/requirements.txt
pip install -e .

# Splits
python -m crashx.data.process_ccd \
  --excel Car_Crash_Text_Dataset_ground_truth.xlsx \
  --video-dir video1500 \
  --out-dir crashx/data/splits

# Train CrashLogic
python -m crashx.models.train_qlora \
  --train-jsonl crashx/data/splits/train.jsonl \
  --val-jsonl crashx/data/splits/val.jsonl \
  --output-dir outputs/crashlogic_7b_lora

# Inference / tables
python -m crashx.run_experiments --lora-path outputs/crashlogic_7b_lora --results-dir results
```

Paper compile: upload `paper/CrashX_IEEE_Overleaf.zip` to Overleaf → `main.tex` → pdfLaTeX → recompile twice.

---

## 17. Suggested 10–12 minute talk outline

1. Problem + example (1 min)  
2. Dataset + split (1 min)  
3. Systems: ZS vs CrashLogic vs TCD (1 min)  
4. Metrics: why C_O / C_H matter (2 min)  
5. Table A + B: adaptation wins completeness, not faithfulness (2 min)  
6. Table C: temporal prior (1.5 min)  
7. Table D/F + mechanism: TCD null/harm (1.5 min)  
8. No-crash + qualitative (1 min)  
9. Conclusions + future work (1 min)

---

## 18. File map for the IEEE paper

| Section file | Content |
|--------------|---------|
| `sections/00_abstract.tex` | Abstract |
| `01_introduction.tex` | Problem + contributions |
| `02_related_work.tex` | Prior work |
| `03_dataset.tex` | Task, CCD, metrics definitions |
| `04_method.tex` | CrashLogic + TCD + predictions P1–P3 |
| `05_experiments.tex` | Setup |
| `06_results.tex` | Main quantitative results |
| `07_analysis.tex` | Fields, prior, no-crash, taxonomy |
| `08_ablation.tex` | TCD ablation |
| `09_qualitative.tex` | Examples |
| `10_limitations.tex` | Limits |
| `11_conclusion.tex` | Conclusions |
| `12_appendix.tex` | Extra tables/figures |

---

*This briefing matches the latest CrashX IEEE draft in `paper/ieee_crashx/` (trimmed main paper + appendix). Update numbers here if you recompute metrics.*
