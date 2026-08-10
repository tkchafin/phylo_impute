#!/usr/bin/env python3
"""Generate deterministic test fixtures for the 10-state diploid CTMC model.

Simulates genotype evolution on a known tree using a GTGTR4 Q matrix,
then saves everything to tests/data/ for reproducible testing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.linalg
import toytree as tt

SEED = 42
RNG = np.random.default_rng(SEED)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---- Tree ----
NEWICK = "((s1:0.05,s2:0.08):0.12,((s3:0.03,s4:0.06):0.10,(s5:0.04,(s6:0.07,s7:0.09):0.05):0.08):0.03);"

# ---- GTR rates (AC, AG, AT, CG, CT, GT) ----
GTR_RATES = np.array([1.0, 4.0, 1.0, 1.0, 4.0, 1.0])  # HKY-like (ts/tv = 4)

# ---- Genotype frequencies (slight HW departure) ----
# Base: f_A=0.3, f_C=0.2, f_G=0.2, f_T=0.3
# Under HW: AA=0.09, CC=0.04, GG=0.04, TT=0.09, AC=0.12, AG=0.12, AT=0.18, CG=0.08, CT=0.12, GT=0.12
FREQS_10 = np.array([0.09, 0.04, 0.04, 0.09, 0.12, 0.12, 0.18, 0.08, 0.12, 0.12])
FREQS_10 /= FREQS_10.sum()  # normalize

# States and IUPAC
GENOTYPE_STATES = ("AA", "CC", "GG", "TT", "AC", "AG", "AT", "CG", "CT", "GT")
GT_IUPAC = {"AA": "A", "CC": "C", "GG": "G", "TT": "T",
            "AC": "M", "AG": "R", "AT": "W", "CG": "S", "CT": "Y", "GT": "K"}

# Rate symmetry (GTGTR4)
_GT_RATE_SYM_FLAT = [
    -1, -1, -1,  0,  1,  2, -1, -1, -1,
        -1, -1,  0, -1, -1,  3,  4, -1,
            -1, -1,  1, -1,  3, -1,  5,
                -1, -1,  2, -1,  4,  5,
                     3,  4,  1,  2, -1,
                         5,  0, -1,  2,
                            -1,  0,  1,
                                 5,  4,
                                     3,
]


def build_gt_rate_sym():
    n = 10
    mat = np.full((n, n), -1, dtype=int)
    k = 0
    for i in range(n):
        mat[i, i] = -2
        for j in range(i + 1, n):
            mat[i, j] = _GT_RATE_SYM_FLAT[k]
            mat[j, i] = _GT_RATE_SYM_FLAT[k]
            k += 1
    return mat


def build_q10(gtr_rates, freqs_10):
    n = 10
    sym = build_gt_rate_sym()
    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            ridx = sym[i, j]
            if ridx < 0:
                continue
            s = gtr_rates[ridx]
            Q[i, j] = s * freqs_10[j]
            Q[j, i] = s * freqs_10[i]
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    mean_rate = float(-np.dot(freqs_10, np.diag(Q)))
    if mean_rate > 0:
        Q /= mean_rate
    return Q


def simulate_on_tree(tree, Q, freqs, n_sites, rng):
    """Simulate genotype evolution on a tree. Returns dict: tip_name -> list of IUPAC codes."""
    # Precompute P(t) for each branch
    node_states = {}

    for node in tree.treenode.traverse("preorder"):
        if node.is_root():
            # Draw root states from stationary distribution
            states = rng.choice(10, size=n_sites, p=freqs)
            node_states[node.idx] = states
        else:
            parent_states = node_states[node.up.idx]
            t = max(node.dist, 1e-10)
            P = scipy.linalg.expm(Q * t)
            # Clamp and normalize
            P = np.maximum(P, 0.0)
            P /= P.sum(axis=1, keepdims=True)

            child_states = np.zeros(n_sites, dtype=int)
            for site_i in range(n_sites):
                ps = parent_states[site_i]
                child_states[site_i] = rng.choice(10, p=P[ps])
            node_states[node.idx] = child_states

    # Collect tip states
    tip_genotypes = {}
    for node in tree.treenode.traverse("postorder"):
        if node.is_leaf():
            states = node_states[node.idx]
            iupac_seq = [GT_IUPAC[GENOTYPE_STATES[s]] for s in states]
            tip_genotypes[node.name] = iupac_seq

    return tip_genotypes


def main():
    tree = tt.tree(NEWICK)
    Q = build_q10(GTR_RATES, FREQS_10)

    n_sites = 500
    genotypes = simulate_on_tree(tree, Q, FREQS_10, n_sites, RNG)

    # Choose sites to mask (10% of non-gap entries)
    mask_rng = np.random.default_rng(SEED + 1)
    tip_names = sorted(genotypes.keys())
    all_entries = [(tip, site) for tip in tip_names for site in range(n_sites)]
    n_mask = int(len(all_entries) * 0.10)
    mask_indices = mask_rng.choice(len(all_entries), size=n_mask, replace=False)
    masked_positions = sorted([all_entries[i] for i in mask_indices])

    # Store original for accuracy checking
    original_genotypes = {tip: list(seq) for tip, seq in genotypes.items()}

    # Apply mask
    masked_genotypes = {tip: list(seq) for tip, seq in genotypes.items()}
    for tip, site in masked_positions:
        masked_genotypes[tip][site] = "N"

    # Save fixtures
    # 1. Tree
    tree_path = DATA_DIR / "diploid_tree.nwk"
    tree_path.write_text(NEWICK + "\n")

    # 2. GTR rates
    rates_path = DATA_DIR / "diploid_gtr_rates.json"
    rates_path.write_text(json.dumps({
        "rates": GTR_RATES.tolist(),
        "rate_labels": ["AC", "AG", "AT", "CG", "CT", "GT"],
    }, indent=2) + "\n")

    # 3. Genotype frequencies
    freqs_path = DATA_DIR / "diploid_freqs.json"
    freqs_path.write_text(json.dumps({
        "freqs": FREQS_10.tolist(),
        "states": list(GENOTYPE_STATES),
    }, indent=2) + "\n")

    # 4. Original genotypes (truth)
    orig_path = DATA_DIR / "diploid_original.json"
    orig_path.write_text(json.dumps(original_genotypes, indent=2) + "\n")

    # 5. Masked genotypes (input to imputation)
    masked_path = DATA_DIR / "diploid_masked.json"
    masked_path.write_text(json.dumps(masked_genotypes, indent=2) + "\n")

    # 6. Masked positions
    positions_path = DATA_DIR / "diploid_masked_positions.json"
    positions_path.write_text(json.dumps(masked_positions, indent=2) + "\n")

    # 7. Q matrix (for reference)
    q_path = DATA_DIR / "diploid_q10.json"
    q_path.write_text(json.dumps({
        "Q": Q.tolist(),
        "states": list(GENOTYPE_STATES),
    }, indent=2) + "\n")

    print(f"Generated {n_sites} sites for {len(tip_names)} tips")
    print(f"Masked {len(masked_positions)} entries ({100*len(masked_positions)/len(all_entries):.1f}%)")

    # Quick sanity: count variable sites
    n_variable = 0
    for site_i in range(n_sites):
        states = set()
        for tip in tip_names:
            states.add(original_genotypes[tip][site_i])
        if len(states) > 1:
            n_variable += 1
    print(f"Variable sites: {n_variable}/{n_sites}")


if __name__ == "__main__":
    main()
