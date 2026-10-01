# LinkedIn post — draft v1 (RAM-constraint framing)

Superseded by `linkedin_post_draft_v2.md`, kept here for comparison/review.

---

**I spent weeks trying to fix my RAG system's hallucination problem. It didn't have one.**

I'm building a personal RAG assistant on a laptop with 7.7GB of RAM — no cloud GPU, no API budget. That constraint forced me to stop eyeballing outputs and build an actual measurement harness: a golden set of questions, a faithfulness scorer, a pass/fail gate. I expected the harness to catch the model making things up.

Across every single run — different models, different prompts, dozens of questions — citation accuracy and grounding stayed at 100%. Zero hallucinations. Not once.

The model was failing in the exact opposite direction: it kept saying "I don't have enough information" on questions it clearly had the context to answer. 8 out of 10 answerable questions, wrongly refused. I rewrote the prompt to explicitly say "don't be so cautious." It got *worse*. I swapped in a completely different model family. Same failure, same shape.

So I went after retrieval itself — added a reranking step that scores exactly how confident the system should be in what it found. On two questions this week, that confidence score came back above 98%. Textbook, unambiguous, "the answer is right there" territory. The model abstained anyway.

That's the part that stuck with me: a system that refuses to answer *looks* safe. No embarrassing wrong answers, no fabricated citations — on a casual read it just looks appropriately careful. It took a labeled test set (and now a confidence score to rule out "maybe retrieval just didn't find it") to prove that "careful" had tipped into "useless."

The lesson I keep coming back to: hallucination and refusal are two different failure modes, and optimizing against one can quietly make the other worse. If you only ever measure "did it make something up," a model that never answers gets a perfect score.

Next hypothesis: it's not *what* the model retrieves, it's *how* that retrieved information is handed to it. More soon.

---

## Notes on this version

- Timeframe said "weeks" — corrected to "this week" in v2.
- Led with the 7.7GB RAM hardware constraint as the hook — replaced in v2 with a real-world hallucination example (Mata v. Avianca).
- Mentioned models only vaguely ("different models," "a completely different model family") — v2 names all five and gives real gate-pass metrics.
- "Did it make something up" was stated abstractly — v2 grounds it in a concrete, citable real-world incident.
