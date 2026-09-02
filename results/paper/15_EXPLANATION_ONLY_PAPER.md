# CrashX — Explanation-Only Paper Narrative (Advisor-Aligned)

**Claim:** Domain-adapted CrashLogic-7B produces far better crash *explanations* than zero-shot Video-LLMs; temporal contrastive decoding (TCD, $\alpha$=0.5) further improves explanation faithfulness and temporal consistency.

**Out of scope (per guide):** severity classification, severity-stratified tables, human Likert studies, commercial API baselines.

---

## Main results location

→ [`results/tables/explanation/all_explanation_tables.md`](../tables/explanation/all_explanation_tables.md)

| Table | Content |
|-------|---------|
| E1 | Explanation quality (BLEU / ROUGE / METEOR / CIDEr / BERTScore) |
| E2 | Explanation faithfulness (NLI + Expl.-BERTScore) |
| E3 | Temporal consistency support (tIoU only) |
| E4 | TCD / SEASON decoding ablation |
| E5 | Relative gains vs zero-shot Qwen2.5-VL-7B |

---

## Suggested abstract numbers

- Zero-shot VLMs: BLEU-4 ≈ **0.02**, explanation BERTScore ≈ **0.42–0.55**, tIoU ≈ **0.00–0.02**
- CrashLogic-7B: BLEU-4 ≈ **0.14**, BERTScore ≈ **0.69**, tIoU ≈ **0.37**
- TCD ($\alpha$=0.5): best temporal support (tIoU ≈ **0.39**) with matched caption quality; SEASON $\alpha$=1.0 best NLI-Score among defaults

---

## Paper structure (short)

1. Introduction — accident explanation is safety-critical; VLMs hallucinate events/timing  
2. Method — QLoRA CrashLogic-7B + TCD at inference  
3. Experiments — Tables E1–E5 only  
4. Qualitative — 3 failure/success explanation examples  
5. Conclusion — domain adaptation essential; TCD helps faithfulness  

Do **not** expand into severity forensics or large human studies unless a reviewer asks.
