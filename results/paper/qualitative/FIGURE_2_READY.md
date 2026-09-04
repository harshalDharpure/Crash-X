# Figure 2 — Ready qualitative panel (use in paper)

Pick **two** panels for the main figure. Full set: [`EXAMPLES.md`](EXAMPLES.md).

---

## Panel A — Fine-tuning recovers a real crash narrative  
**Video `001227` (moderate)** · ZS ROUGE-L=0.00 → Greedy 0.45 → TCD 0.50

| Source | Explanation (shortened for figure) |
|--------|-------------------------------------|
| **GT** | Camera car on a curved two-lane road; white car from opposite direction; front-left to front-left collision after evasive left turn. |
| **Zero-shot Qwen** | Collapses / non-explanation (structured prompt failure). |
| **CrashLogic Greedy** | Clear opposite-direction white car, loss of control, front/left impact — matches GT agents & cause family. |
| **CrashLogic TCD** | Same agents; adds intersection crossing framing; front-left / front-right impact — closer spatial wording. |

**Caption idea:** *Zero-shot fails to produce a usable crash explanation; CrashLogic recovers the opposite-direction collision story.*

---

## Panel B — Fluent zero-shot still fabricates  
**Video `000104` (moderate)** · ZS ROUGE-L=0.09, Greedy 0.41

| Source | Issue |
|--------|--------|
| **GT** | Camera car hits **black truck** ahead (slow lead vehicle); red bus passes. |
| **Zero-shot** | Talks weather/road “conditions” — **no crash narrative**. |
| **CrashLogic** | Rear-end story present (wrong partner color sometimes) — still a crash explanation. |

**Caption idea:** *Fluency ≠ faithfulness: zero-shot describes context; CrashLogic attempts causal crash explanation.*

---

## Panel C — TCD temporal win (appendix or Fig 3)  
**Video `000461`** · Greedy tIoU=0.0 → TCD tIoU=1.0 (ROUGE remains high)

Use when discussing TCD: *same explanation quality band, better crash timing.*

---

## Limitations panel (Discussion)

From §D hard cases: even CrashLogic can swap vehicle colors/roles. State clearly: *domain adaptation is necessary but not sufficient for perfect forensic explanations.*
