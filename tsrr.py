import numpy as np
import warnings
from math import comb

def expected_rank(grank, ns: int, nt: int) -> float:
    """
    Expected rank of the first target in a randomly ranked set.

    Parameters:
        grank (int): Rank offset before this set (we pass 0 to get within-tie expectation).
        ns (int): Number of non-target (irrelevant) items in the set.
        nt (int): Number of target (relevant) items in the set.

    Returns:
        float: grank + E[min position of a target among ns+nt items].
    """
    # Interpret ns as non-targets and nt as targets; total size is ns + nt.
    ns = ns + nt  # follow the provided reference implementation semantics
    if ns <= 0 or nt <= 0 or nt > ns:
        raise ValueError("Invalid input: ns must be >= nt > 0")

    expected_r = 0.0
    for rank in range(1, ns + 1):
        prob = comb(ns - rank, nt - 1) / comb(ns, nt)
        expected_r += rank * prob
    return grank + expected_r


def tsrr(target, results, similarities, alpha=None, reduction='mean'):
    """
    Tie-sensitive Reciprocal Rank (TsRR) with tie-size penalty.

    TsRR = 1 / ( r_pre + E_tau[L] ), where
        E_tau[L] = (1 - tau) * E[L] + tau * L_max
        tau      = |G_irr| / N_irr
        E[L]     = expected rank of the first relevant *within the tie group* G
                   (computed via combinatorics; see expected_rank with grank=0)
        L_max    = |G| - k + 1  (worst position of first relevant in G)

    Inputs accept single or batched targets, as before.
    'alpha' is deprecated and ignored (kept for backward compatibility).
    """
    if alpha is not None:
        warnings.warn(
            "tsrr(alpha=...) is deprecated and ignored; the updated TsRR no longer uses alpha.",
            DeprecationWarning
        )

    # ---------- normalize inputs to batched form ----------
    if not isinstance(target, (list, np.ndarray)):
        if not (isinstance(results, list) and isinstance(similarities, list)):
            raise ValueError("For a single target, 'results' and 'similarities' must be lists.")
        if len(results) != len(similarities):
            raise ValueError("Lengths of 'results' and 'similarities' must match (single target).")
        target = [target]
        results = [results]
        similarities = [similarities]
    elif isinstance(target, np.ndarray) and target.ndim == 0:
        if isinstance(results, np.ndarray):
            results = results.tolist()
        if isinstance(similarities, np.ndarray):
            similarities = similarities.tolist()
        if len(results) != len(similarities):
            raise ValueError("Lengths of 'results' and 'similarities' must match (single target).")
        target = [target.item()]
        results = [results]
        similarities = [similarities]
    else:
        if isinstance(target, np.ndarray):
            if target.ndim != 1:
                raise ValueError("For multiple targets, 'target' must be 1D.")
            target = target.tolist()

        if isinstance(results, np.ndarray):
            if results.ndim != 2:
                raise ValueError("For multiple targets, 'results' must be 2D.")
            results = results.tolist()
        else:
            if not (isinstance(results, list) and all(isinstance(r, list) for r in results)):
                raise ValueError("For multiple targets, 'results' must be a 2D list/array.")

        if isinstance(similarities, np.ndarray):
            if similarities.ndim != 2:
                raise ValueError("For multiple targets, 'similarities' must be 2D.")
            similarities = similarities.tolist()
        else:
            if not (isinstance(similarities, list) and all(isinstance(s, list) for s in similarities)):
                raise ValueError("For multiple targets, 'similarities' must be a 2D list/array.")

        if len(results) != len(target) or len(similarities) != len(target):
            raise ValueError("Number of rows in 'results'/'similarities' must match #targets.")

        for i, (r_row, s_row) in enumerate(zip(results, similarities)):
            if len(r_row) != len(s_row):
                raise ValueError(f"Row {i}: 'results' and 'similarities' lengths differ.")

    # numpy arrays for convenience
    results = np.array(results, dtype=object)
    similarities = np.array(similarities, dtype=float)

    scores = []
    for i, label in enumerate(target):
        # sort by similarity desc
        order = np.argsort(similarities[i])[::-1]
        sims = similarities[i, order]
        labs = results[i, order]

        # indices of all relevant items (== target)
        rel_idx = np.where(labs == label)[0]
        if rel_idx.size == 0:
            scores.append(0.0)
            continue

        # first relevant's similarity and position
        j = rel_idx[0]
        s = sims[j]

        # r_pre: items strictly above this tie
        r_pre = int(np.sum(sims > s))

        # tie group G (all items with the same score s)
        tie_idx = np.where(sims == s)[0]
        G = int(tie_idx.size)
        k = int(np.sum(labs[tie_idx] == label))  # #relevants in G
        ns = G - k                               # #irrelevants in G

        # totals (for tau)
        N = int(labs.size)
        R_total = int(np.sum(labs == label))
        N_irr = N - R_total

        # tau = |G_irr| / N_irr (0 if no irrelevants overall)
        tau = 0.0 if N_irr == 0 else ns / float(N_irr)

        # E[L]: expected rank of FIRST relevant *within tie G*
        # use the provided combinatorial routine with grank=0, ns=|G_irr|, nt=k
        E_L = expected_rank(0, ns, k)

        # L_max within the tie
        L_max = ns + 1.0  # == |G| - k + 1

        # blend expected and worst-case ranks
        E_tau = (1.0 - tau) * E_L + tau * L_max

        # TsRR
        denom = r_pre + E_tau
        scores.append(1.0 / denom)

    return float(np.mean(scores)) if reduction == 'mean' else scores
