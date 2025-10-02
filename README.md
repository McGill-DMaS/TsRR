# TsRR: Tie-sensitive Reciprocal Rank

TsRR is a ranking metric for information retrieval systems that properly accounts for *tied relevance scores*. Ties occur when multiple documents receive identical scores, which can lead to misleading evaluations with traditional Mean Reciprocal Rank (MRR). TsRR addresses this challenge by **penalizing excessive ties** while still rewarding systems that distinguish truly relevant documents.

## Motivation

In many learning-to-rank scenarios, multiple documents may receive the same relevance score. Standard metrics assume a strict ranking order and may overestimate performance when ties are present. For example, in a fully tied scenario (i.e., when a system makes no distinctions between documents), the expected rank of a relevant document can appear deceptively favorable.

TsRR was proposed to counter this paradox by:

- **Mitigating over-rewarding of fully tied rankings:** Large tie groups of irrelevant items receive a proportional penalty.  
- **Preserving sensitivity to genuine ranking effort:** Systems that push relevant items upward, or into smaller tie groups, are rewarded.  
- **Reducing to known cases:** In the absence of ties, TsRR reduces to standard RR; in the extreme case of a fully tied list, TsRR reduces to the pessimistic RR (PRR).  

## The TsRR Formula

For a given query, let:

- ![r_pre](https://latex.codecogs.com/svg.image?r_{\mathrm{pre}}) = the number of documents strictly ranked before the tie group containing the first relevant document.  
- ![G](https://latex.codecogs.com/svg.image?G) = the tie group containing the first relevant document(s).  
- ![k](https://latex.codecogs.com/svg.image?k) = the number of relevant documents within ![G](https://latex.codecogs.com/svg.image?G).  
- ![G_irr](https://latex.codecogs.com/svg.image?|G_{\mathrm{irr}}|=|G|-k) = the number of irrelevants in that tie group.  
- ![N_irr](https://latex.codecogs.com/svg.image?N_{\mathrm{irr}}) = the total number of irrelevants retrieved.  
- ![E(L)](https://latex.codecogs.com/svg.image?E[L]) = the expected position of the first relevant document *within its tie group* (computed combinatorially).  
- ![L_max](https://latex.codecogs.com/svg.image?L_{\max}=|G|-k+1) = the worst-case position of the first relevant in the tie.  

We define the tie penalty factor:

![tau](https://latex.codecogs.com/svg.image?\tau=\frac{|G_{\mathrm{irr}}|}{N_{\mathrm{irr}}})

which represents the fraction of all irrelevants that are tied with the first relevant document.

The tie-adjusted expected rank is then:

![E_tau](https://latex.codecogs.com/svg.image?E_{\tau}[L]=(1-\tau)E[L]+\tau\,L_{\max})

Finally, TsRR is:

![TsRR](https://latex.codecogs.com/svg.image?\mathrm{TsRR}=\frac{1}{r_{\mathrm{pre}}+E_{\tau}[L]})

## How It Works

1. **Tie Handling:**  
   - If many irrelevants are tied with the first relevant, ![tau](https://latex.codecogs.com/svg.image?\tau) grows, shifting the expectation toward the *worst case* position.  
   - If the tie is small or contains mostly relevants, ![tau](https://latex.codecogs.com/svg.image?\tau) is small, and TsRR behaves more like tie-aware RR (ta-RR).  

2. **Rank Rewarding:**  
   - The reciprocal form ensures that systems placing the first relevant earlier (smaller denominator) get higher scores.  

3. **Special Cases:**  
   - **No ties:** TsRR = ![1/r](https://latex.codecogs.com/svg.image?\frac{1}{r}), reducing exactly to classical Reciprocal Rank.  
   - **Full tie (all docs tied, one relevant):** TsRR = ![1/R](https://latex.codecogs.com/svg.image?\frac{1}{R}), identical to pessimistic RR (PRR).  

## An Illustrative Comparison

- **Fully Tied System (R documents, one relevant):**  
  ![fully_tied](https://latex.codecogs.com/svg.image?r_{\mathrm{pre}}=0,\;|G|=R,\;k=1,\;\tau=1)  
  ![fully_tied_result](https://latex.codecogs.com/svg.image?E_{\tau}[L]=R,\;\;\mathrm{TsRR}=\frac{1}{R})  
  matching PRR.  

- **Strict Ranking (first relevant at rank r, no ties):**  
  ![strict_case](https://latex.codecogs.com/svg.image?E[L]=1,\;\tau=0,\;\mathrm{TsRR}=\frac{1}{r})  
  matching classical RR.  

Thus TsRR interpolates between ta-RR and PRR while penalizing larger tie groups appropriately.

## Requirements

- **[NumPy](https://numpy.org/)**  

## Installation

Clone the repository and install NumPy:

```bash
git clone https://github.com/McGill-DMaS/TsRR.git
