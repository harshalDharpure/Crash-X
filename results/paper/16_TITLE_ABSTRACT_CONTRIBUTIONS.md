# CrashX — Title, Abstract & Contributions (Explanation Paper)

Advisor-aligned draft. Numbers from Tables E1–E5 (`results/tables/explanation/`).

---

## Title options (pick one)

1. **CrashX: Domain-Adapted Video-LLMs for Faithful Traffic Accident Explanations** *(recommended)*  
2. CrashLogic: Mitigating Temporal Hallucinations in Crash Video Explanations  
3. When Video-LLMs Fail at Crash Explanations: CrashX Benchmarking and Temporal Contrastive Decoding  

---

## Abstract (≈180 words)

Traffic accident understanding requires not only recognizing that a crash occurred, but producing a **faithful causal explanation** of what happened and when. We show that strong open Video-LLMs (Qwen2.5-VL, Qwen2-VL, LLaVA-NeXT-Video) fail at this task on the Car Crash Dataset: explanation BLEU-4 stays near **0.02** and temporal IoU near **0.01**, despite fluent free-form text. We introduce **CrashX**, an explanation-centric evaluation protocol, and **CrashLogic-7B**, a QLoRA adaptation of Qwen2.5-VL-7B on dense crash explanations. Fine-tuning yields large gains in explanation quality (**BLEU-4 0.016→0.142**, BERTScore **0.486→0.686**) and crash-window localization (**tIoU 0.012→0.373**). At inference, **Temporal Contrastive Decoding (TCD)** with a reversed-video negative (α=0.5) preserves caption quality while improving explanation faithfulness (**NLI-Score −0.005→0.057**) and temporal consistency (**tIoU 0.394**). Ablations show that overly strong contrast (α≥1.5) hurts explanation metrics. Qualitative analysis confirms that zero-shot models often invent agents or causes, whereas CrashLogic aligns with ground-truth narratives. CrashX demonstrates that **domain adaptation is necessary** for trustworthy accident explanations, and that lightweight temporal contrast at decode time further reduces unfaithful explanations.

---

## Contribution bullets (use 3–4)

1. **Empirical finding:** Open Video-LLMs produce fluent but largely unfaithful crash explanations on CCD (low BLEU/BERTScore, near-zero tIoU).  
2. **CrashLogic-7B:** QLoRA adaptation that substantially improves explanation quality and temporal alignment on a held-out 150-video test split.  
3. **TCD for explanations:** Reverse-video contrastive decoding (α=0.5) improves NLI faithfulness and tIoU without sacrificing BLEU/BERTScore; we ablate α and negative construction.  
4. **Explanation-centric protocol:** Public comparisons on Explanation-field metrics (lexical + NLI + temporal support), with qualitative failure cases. *(Optional 4th)*

---

## One-paragraph intro hook

> A dashboard camera captures a multi-vehicle collision in under two seconds. A Video-LLM may output a polished paragraph—*“the driver lost control on a snowy road…”*—yet invent the weather, the other vehicle, or the crash time. For accident analysis, **fluency is not faithfulness**. CrashX studies this gap and shows how domain adaptation plus temporal contrastive decoding closes it.

---

## How to talk about TCD (reviewer-safe)

| Say | Don’t say |
|-----|-----------|
| Matched explanation quality, better faithfulness / timing | Novel SOTA decoder that beats everything |
| α=0.5 is primary; α=1.0 peaks NLI-Score (0.090) | SEASON always wins |
| Supporting inference-time alignment | Core theoretical contribution |

---

## Suggested figure plan

| Fig | Content | Source |
|-----|---------|--------|
| Fig 1 | Pipeline: video → CrashLogic → optional TCD → explanation | Draw |
| Fig 2 | Qualitative side-by-side (GT / ZS / Greedy / TCD) | `qualitative/EXAMPLES.md` §A–B |
| Fig 3 | Ablation bar chart (α sweep on NLI + tIoU) | Table E4 |
| Fig 4 | Failure modes pie / examples | `qualitative/EXAMPLES.md` §D + taxonomy below |

---

## Error taxonomy (for Discussion / qualitative)

Code labels for annotating failures (manual or semi-auto):

| Code | Meaning |
|------|---------|
| `WRONG_AGENT` | Invents / swaps vehicles involved |
| `WRONG_CAUSE` | Incorrect causal story (brake, turn, weather) |
| `WRONG_TIME` | Crash timing far from GT window |
| `FLUENT_FABRICATION` | Long fluent text with little GT overlap |
| `UNDER_SPECIFIC` | Vague explanation missing key events |

See mined examples: [`qualitative/EXAMPLES.md`](qualitative/EXAMPLES.md).

---

## Recommended reporting table in main paper

Use **E1 + E2 + E3** in the main body; **E4** in ablation; **E5** in text (“~9× BLEU”).  
Do **not** put severity tables in the main paper.
