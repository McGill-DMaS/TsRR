# TsRR: Tie-sensitive Reciprocal Rank

TsRR is a ranking metric designed for information retrieval systems that properly account for tied relevance scores. Ties occur when multiple documents receive identical scores, which can lead to misleading evaluations with traditional Mean Reciprocal Rank (MRR). TsRR addresses this challenge by penalizing excessive ties while still rewarding systems that distinguish truly relevant documents.

## Motivation

In many learning-to-rank scenarios, multiple documents may receive the same relevance score for a given query. Standard metrics assume a strict ranking order and may overestimate performance when ties are present. For example, in a fully tied scenario (i.e., when a system makes no distinctions between documents), the expected rank of a relevant document can be deceptively favorable. TsRR was proposed to counter this paradox by:

- **Mitigating over-rewarding of fully tied rankings:** Systems that group too many items together receive a substantial penalty.
- **Maintaining computational efficiency:** TsRR uses simple logarithmic and multiplicative operations.
- **Adapting flexibly to tie scenarios:** The tunable parameter \(\alpha\) adjusts the sensitivity of the tie penalty.

## The TsRR Formula

For a given target document (or query), let:
- \(r_{\mathrm{pre}}\) be the number of documents ranked before the tie group that contains at least one relevant document.
- \(G\) denote the tie group that contains the relevant document(s).
- \(k\) be the number of relevant documents within \(G\).
- \(F_G = |G| - k\) be the number of irrelevant (false positive) documents in the tie group.
- \(F_{\mathrm{total}}\) be the total number of irrelevant documents retrieved.
- \(\alpha > 0\) be a parameter governing the sensitivity of the tie penalty.

Then, TsRR is defined as:

\[
\mathrm{TsRR} =
\begin{cases} 
\left( 1 - \left( \dfrac{\ln(1 + F_G)}{\ln(1 + F_{\mathrm{total}})} \right)^{\frac{1}{\alpha}} \right)
\cdot \dfrac{1}{\,r_{\mathrm{pre}}+1}, & \text{if } F_{\mathrm{total}} > 0, \\
1, & \text{if } F_{\mathrm{total}} = 0.
\end{cases}
\]

### How It Works

1. **Tie Handling:**  
   The logarithmic term \( \ln(1 + F_G) \) grows sublinearly, which means that even if many irrelevant documents are tied with the relevant ones, the penalty is moderated. Normalizing by \( \ln(1 + F_{\mathrm{total}}) \) scales the penalty in context of the overall retrieval performance.

2. **Rank Rewarding:**  
   Multiplying by \( \frac{1}{r_{\mathrm{pre}} + 1} \) rewards systems that rank the tie group (containing at least one relevant document) higher in the list.

3. **Parameter \(\alpha\):**  
   The sensitivity parameter \(\alpha\) adjusts how harshly the metric penalizes ties:
   - Lower \(\alpha\) values (e.g., 0.5 or 1.0) are more forgiving.
   - Higher \(\alpha\) values (e.g., 2.0 or 4.0) impose a stricter penalty on ties.

### An Illustrative Comparison

Consider two scenarios:
- **Fully Tied System:** All retrieved documents have the same score. If there is one relevant document among \(R\) documents, then:
  \[
  E[L] = \frac{R+1}{2} \quad\text{and}\quad \text{ta-RR} = \frac{2}{R+1}.
  \]
- **Distinct Ranking System:** The relevant document is placed at rank \(r\) (with no ties around it), then:
  \[
  \text{ta-RR} = \frac{1}{r}.
  \]
If \( r > \frac{R+1}{2} \), a tied system might yield a higher score than a system that makes an effort to rank documents—even if the latter correctly identifies a relevant document. TsRR avoids this counterintuitive result by applying the tie penalty.

## Requirements

- **[NumPy](https://numpy.org/)** (the only dependency)

## Installation

Clone the repository and install NumPy:

```bash
git clone https://github.com/yourusername/tsrr.git
cd tsrr
pip install numpy
