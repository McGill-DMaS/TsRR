# TsRR: Tie-Sensitive Reciprocal Rank

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TSE-00629B.svg)](https://doi.org/10.1109/TSE.2026.3705321)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTSE.2026.3705321-007EC6.svg)](https://doi.org/10.1109/TSE.2026.3705321)
[![BibTeX](https://img.shields.io/badge/Cite-BibTeX-success.svg)](#citation)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Dependency-NumPy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)

> **TsRR is introduced in:**  
> **LENA: Llama-based Embeddings of Neutralized Assembly for Cross-compiler/optimization Binary Code Similarity Detection**  
> Mohammadhossein Amouei, Benjamin C. M. Fung, Philippe Charland, and Jun Meng  
> *IEEE Transactions on Software Engineering*, pp. 1–24, 2026  
> **[Read the paper](https://doi.org/10.1109/TSE.2026.3705321)** · **[DOI](https://doi.org/10.1109/TSE.2026.3705321)** · **[BibTeX](#citation)**

## Overview

**Tie-Sensitive Reciprocal Rank (TsRR)** is a ranking metric for information-retrieval systems that explicitly accounts for tied similarity or relevance scores.

Ties occur when multiple candidates receive identical scores. In such cases, conventional Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR) depend on an arbitrary ordering among tied candidates and can therefore overestimate the quality of a ranking system. TsRR addresses this limitation by penalizing large ties involving irrelevant candidates while continuing to reward systems that rank relevant candidates early.

## Motivation

Traditional reciprocal-rank metrics assume a strict ordering of retrieved candidates. This assumption becomes problematic when a model assigns the same score to multiple candidates, because their displayed order may be arbitrary.

A particularly important failure case is a fully tied ranking. A system that makes no distinction among candidates should not receive a favorable score merely because the relevant candidate happens to appear early after arbitrary tie breaking.

TsRR is designed to:

- **Penalize excessive ties:** Large tie groups containing irrelevant candidates receive a stronger penalty.
- **Reward genuine ranking effort:** A system is rewarded for moving relevant candidates upward or placing them in smaller tie groups.
- **Remain compatible with standard metrics:** Without ties, TsRR reduces to classical RR; for a fully tied ranking with one relevant candidate, it reduces to pessimistic RR.

## Definition

For a query, consider the highest-ranked tie group containing at least one relevant candidate.

Let:

- $r_{\mathrm{pre}}$ be the number of candidates ranked strictly before that tie group.
- $G$ be the tie group containing the first relevant candidate or candidates.
- $|G|$ be the size of the tie group.
- $k$ be the number of relevant candidates in $G$.
- $|G_{\mathrm{irr}}| = |G| - k$ be the number of irrelevant candidates in $G$.
- $N_{\mathrm{irr}}$ be the total number of retrieved irrelevant candidates.
- $L$ be the position of the first relevant candidate within $G$ under random tie breaking.

The expected position of the first relevant candidate within the tie group is:

$$
\mathbb{E}[L] = \frac{|G|+1}{k+1}.
$$

The worst-case position of the first relevant candidate within the tie group is:

$$
L_{\max} = |G|-k+1.
$$

### Tie-Penalty Factor

The tie-penalty factor is the fraction of all retrieved irrelevant candidates that are tied with the first relevant candidate:

$$
\tau =
\begin{cases}
\dfrac{|G_{\mathrm{irr}}|}{N_{\mathrm{irr}}}, & N_{\mathrm{irr}} > 0, \\[6pt]
0, & N_{\mathrm{irr}} = 0.
\end{cases}
$$

A larger $\tau$ indicates that the model has failed to distinguish the first relevant candidate from a larger fraction of the irrelevant retrieval pool.

### Tie-Adjusted Expected Position

TsRR interpolates between the expected and worst-case positions within the tie group:

$$
\mathbb{E}_{\tau}[L]
=
(1-\tau)\mathbb{E}[L]
+
\tau L_{\max}.
$$

The query-level TsRR score is then:

$$
\mathrm{TsRR}
=
\frac{1}
{r_{\mathrm{pre}}+\mathbb{E}_{\tau}[L]}.
$$

For a set of queries, the final score is the arithmetic mean of their query-level TsRR values.

## Interpretation

TsRR responds to tied rankings in three ways:

1. **Tie size:** Tying the first relevant candidate with more irrelevant candidates increases the penalty.
2. **Global context:** The penalty depends on how much of the irrelevant retrieval pool participates in the tie.
3. **Relevant multiplicity:** A tie group containing multiple relevant candidates receives a less pessimistic expected first-relevant position than a group containing only one.

Consequently, TsRR distinguishes between systems that produce similar nominal ranks but substantially different amounts of score discrimination.

## Special Cases

### Strict Ranking

When the first relevant candidate is not tied:

$$
|G|=k=1,\qquad
\tau=0,\qquad
\mathbb{E}_{\tau}[L]=1.
$$

Therefore:

$$
\mathrm{TsRR}
=
\frac{1}{r_{\mathrm{pre}}+1},
$$

which is exactly classical Reciprocal Rank.

### Fully Tied Ranking

Suppose all $R$ candidates are tied and exactly one is relevant:

$$
r_{\mathrm{pre}}=0,\qquad
|G|=R,\qquad
k=1,\qquad
\tau=1.
$$

Then:

$$
\mathbb{E}_{\tau}[L]=L_{\max}=R
$$

and:

$$
\mathrm{TsRR}=\frac{1}{R},
$$

which is identical to pessimistic Reciprocal Rank.

### Partial Tie

For a partial tie, $0 < \tau < 1$. TsRR lies between tie-aware RR based on random tie breaking and pessimistic RR, with larger irrelevant tie groups shifting the score toward the pessimistic case.

## Comparison with Related Metrics

| Metric | Treatment of ties |
|---|---|
| **RR** | Requires an arbitrary strict order and may over-reward tied rankings. |
| **Tie-aware RR** | Uses the expected first-relevant position under random tie breaking. |
| **Pessimistic RR** | Always places relevant candidates at the worst possible position within a tie. |
| **TsRR** | Adaptively interpolates between expected and pessimistic treatment according to tie severity. |

## Requirements

- Python 3
- [NumPy](https://numpy.org/)

## Installation

Clone the repository:

```bash
git clone https://github.com/McGill-DMaS/TsRR.git
cd TsRR
```

Install the required dependency:

```bash
python -m pip install numpy
```

## Citation

If you use TsRR in your research, please cite the paper in which the metric was introduced:

> M. Amouei, B. C. M. Fung, P. Charland, and J. Meng, “LENA: Llama-based Embeddings of Neutralized Assembly for Cross-compiler/optimization Binary Code Similarity Detection,” *IEEE Transactions on Software Engineering*, pp. 1–24, 2026, doi: 10.1109/TSE.2026.3705321.

<details>
<summary><strong>Show BibTeX</strong></summary>

```bibtex
@article{amouei2026lena,
  author   = {Amouei, Mohammadhossein and
              Fung, Benjamin C. M. and
              Charland, Philippe and
              Meng, Jun},
  title    = {{LENA: Llama-based Embeddings of Neutralized Assembly for
               Cross-compiler/optimization Binary Code Similarity Detection}},
  journal  = {IEEE Transactions on Software Engineering},
  year     = {2026},
  pages    = {1--24},
  doi      = {10.1109/TSE.2026.3705321},
  keywords = {Binary code similarity detection, self-supervised learning,
              assembly code embeddings, compiler-induced variations,
              tied similarity scores, vulnerability detection}
}
```

</details>

## Disclaimer

This software is provided as-is, without warranty or support. The authors assume no responsibility for damages, loss of income, or other problems arising from its use.

For further information, please consult the paper and the source code. Questions and issue reports may be submitted through the repository's issue tracker.
