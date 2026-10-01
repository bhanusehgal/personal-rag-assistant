# LinkedIn post — draft v2 (real-world example + model metrics)

Current preferred draft as of 2026-09-23. See `linkedin_post_draft_v1.md` for the earlier RAM-constraint-framed version it replaced.

---

**"ChatGPT cited six court cases. All six were fake. A lawyer submitted them to a federal judge and got sanctioned for it."**

That case (*Mata v. Avianca*, 2023) is why most RAG development obsesses over one failure mode: hallucination. It's the visible, embarrassing one — the kind that ends up in a legal filing or a headline.

This week I put a retrieval-augmented system through an actual benchmark instead of eyeballing outputs — a labeled question set, automated citation/grounding scoring, a hard pass/fail gate. I ran it against five different local, open models: Llama 3.2 3B, two variants of Phi-4-mini, Qwen 2.5 7B, and Mistral 7B. Two of the five didn't even survive to finish a full run — too slow or unreliable under real load, itself a useful data point. The two that did complete told a far more interesting story than I expected.

Across every model, every run: **100% citation accuracy, 100% grounding. Zero hallucinations. Not once.**

So the system was safe. It just wasn't useful. Llama 3.2 3B and Phi-4-mini passed my overall quality gate at only **42.9% and 28.6%** respectively — not because they made anything up, but because they refused to answer. On 8 out of 10 genuinely answerable questions, the model said "I don't have enough information," despite having retrieved exactly the right source material.

I assumed it was a prompting problem and rewrote the instructions to explicitly say "don't be overly cautious." It got *worse*. I swapped model families entirely. Same failure, same shape.

So I went after retrieval itself — added a confidence-scoring layer that tells me exactly how strong a match the system found before the model even sees it. On two separate questions this week, that score came back above **98%**. Unambiguous, "the answer is right there" territory. The model abstained anyway.

That's the part that's stuck with me: a system that says "I don't know" *looks* trustworthy — no fake citations, no confident nonsense, nothing you'd catch on a casual read. It took labeled ground truth and a real confidence score to prove "cautious" had quietly become "useless."

The lesson: hallucination and refusal are different failure modes that trade off against each other. Optimize only against "did it make something up" — the *Avianca*-lawyer failure mode — and a model that never answers scores perfectly on that metric while adding zero value.

Still chasing the actual cause — next hypothesis is that it's not *what* the model retrieves, it's *how* that retrieved context gets handed to it. More soon.

---

## Open items for review tomorrow

- Confirm the *Mata v. Avianca* framing/citation reads accurately and isn't a stretch of the actual case facts.
- Decide whether to trim the five-model list or expand with one more line on why two models timed out.
- The "98%" confidence stat is from the 2026-09-22 probe run (q002/q003, real numbers: 0.988 and 0.994) — solid to publish. The same run also had a server crash and a gate-scoring anomaly on a different question (q004) that are deliberately left out here since they're unresolved bugs, not a finding.
