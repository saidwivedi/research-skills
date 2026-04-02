# Common LLM Agent Mistakes in Research

Known failure modes of LLM agents in ML/AI research collaboration.
The agent reads this before acting and appends new entries when corrected
(see SKILL.md "Learning From Mistakes"). Users can add entries directly too.

---

## Experiment Design

- **Proxy metrics as go/no-go criteria.** Don't declare a hypothesis dead based on proxy metrics (Pearson r, cosine similarity, etc.). Proxies miss non-linear relationships. Kill tests must use the actual task metric on held-out data.
- **Optimizing downstream before validating upstream.** Don't tune parameters for a component before verifying the component itself produces signal. If X depends on Y, validate Y first.
- **Reformulating instead of extending.** Don't propose ambitious new formulations when validated results exist. Extend what works before pivoting.

## Result Interpretation

- **Reconstruction metrics on generative tasks.** Don't use point-wise error (MSE, L1) against a single ground-truth to judge generative models. Use distributional metrics (FID, KID), semantic correctness, and diversity measures.
- **Contradicting past decisions blindly.** Don't recommend reverting an architectural change without checking whether it helped. Check metric history before proposing reversals.

## Code Generation

_(No entries yet)_

## Methodology

_(No entries yet)_

## Communication

_(No entries yet)_

---

**Entry format:** `- **[Pattern name.]** [What not to do and why. What to do instead.]`
