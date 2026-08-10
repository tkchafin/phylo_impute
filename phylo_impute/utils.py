from __future__ import annotations

import numpy as np

CHAR_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
NUC_MAP = {v: k for k, v in CHAR_MAP.items()}
MISSING_TOKENS = {"N", "n", "-9", ".", "?", "./.", ""}
MIN_BRANCH_LENGTH = 1e-10
STATE_ORDER = ("A", "C", "G", "T")

IUPAC_MAP = {
    "A": ["A"],
    "G": ["G"],
    "C": ["C"],
    "T": ["T"],
    "N": ["A", "C", "T", "G"],
    "-": ["A", "C", "T", "G"],
    "R": ["A", "G"],
    "Y": ["C", "T"],
    "S": ["G", "C"],
    "W": ["A", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
}


def is_missing(x):
    return str(x).strip() in MISSING_TOKENS


def get_iupac_full(char):
    return IUPAC_MAP.get(str(char).upper(), ["A", "C", "T", "G"])


def normalize_rows(mat):
    sums = mat.sum(axis=1, keepdims=True)
    bad = (sums <= 0.0) | ~np.isfinite(sums)
    if np.any(bad):
        mat = mat.copy()
        mat[bad[:, 0], :] = 1.0 / mat.shape[1]
        sums = mat.sum(axis=1, keepdims=True)
    return mat / sums


def likelihood_matrix_for_chars(chars):
    lik = np.zeros((len(chars), 4), dtype=float)
    for row_i, char in enumerate(chars):
        if is_missing(char):
            lik[row_i, :] = 1.0
        else:
            for state in get_iupac_full(char):
                lik[row_i, CHAR_MAP[state]] = 1.0
    return normalize_rows(lik)


# ---------------------------------------------------------------------------
# 10-state diploid genotype model (GTGTR4 / GT10)
# ---------------------------------------------------------------------------

# States in the canonical order used by coraxlib/RAxML-NG
GENOTYPE_STATES = ("AA", "CC", "GG", "TT", "AC", "AG", "AT", "CG", "CT", "GT")

# Map each genotype to its IUPAC code (for encoding observed data)
GENOTYPE_IUPAC = {
    "AA": "A", "CC": "C", "GG": "G", "TT": "T",
    "AC": "M", "AG": "R", "AT": "W",
    "CG": "S", "CT": "Y", "GT": "K",
}

# Reverse: IUPAC code -> genotype state index(es)
# Homozygous codes map to a single state; ambiguity codes map to multiple.
IUPAC_TO_GT_INDICES = {
    "A": [0],           # AA
    "C": [1],           # CC
    "G": [2],           # GG
    "T": [3],           # TT
    "M": [4],           # AC
    "R": [5],           # AG
    "W": [6],           # AT
    "S": [7],           # CG
    "Y": [8],           # CT
    "K": [9],           # GT
}

# Index map for quick lookup
GT_STATE_INDEX = {s: i for i, s in enumerate(GENOTYPE_STATES)}

# Rate symmetry matrix (upper triangle, flat) for the GTGTR4 model.
# Encodes which of the 6 GTR DNA rates (AC=0, AG=1, AT=2, CG=3, CT=4, GT=5)
# governs each pair of genotype states, or -1 if the transition is forbidden.
#
# This is the biologically correct version derived from single-mutation rules:
#   Hom<->Hom: -1 (requires 2 mutations)
#   Hom<->Het: rate of the nucleotide that changes
#   Het<->Het: rate of the allele that changes; -1 if both change
#
# Order: AA CC GG TT AC AG AT CG CT GT
_GT_RATE_SYM_FLAT = [
    # AA row (vs CC,GG,TT,AC,AG,AT,CG,CT,GT)
    -1, -1, -1,  0,  1,  2, -1, -1, -1,
    # CC row (vs GG,TT,AC,AG,AT,CG,CT,GT)
        -1, -1,  0, -1, -1,  3,  4, -1,
    # GG row (vs TT,AC,AG,AT,CG,CT,GT)
            -1, -1,  1, -1,  3, -1,  5,
    # TT row (vs AC,AG,AT,CG,CT,GT)
                -1, -1,  2, -1,  4,  5,
    # AC row (vs AG,AT,CG,CT,GT)
                     3,  4,  1,  2, -1,
    # AG row (vs AT,CG,CT,GT)
                         5,  0, -1,  2,
    # AT row (vs CG,CT,GT)
                            -1,  0,  1,
    # CG row (vs CT,GT)
                                 5,  4,
    # CT row (vs GT)
                                     3,
]


def _build_gt_rate_sym_matrix():
    """Expand the flat upper triangle into a full 10x10 symmetric matrix."""
    n = 10
    mat = np.full((n, n), -1, dtype=int)
    k = 0
    for i in range(n):
        mat[i, i] = -2  # diagonal marker
        for j in range(i + 1, n):
            mat[i, j] = _GT_RATE_SYM_FLAT[k]
            mat[j, i] = _GT_RATE_SYM_FLAT[k]
            k += 1
    return mat


GT_RATE_SYMMETRY = _build_gt_rate_sym_matrix()


def likelihood_matrix_for_genotypes(chars):
    """Build (L x 10) likelihood matrix for observed IUPAC genotype codes.

    Each character is mapped to the genotype state(s) it represents.
    Missing data (N, ?, etc.) maps to uniform over all 10 states.
    """
    lik = np.zeros((len(chars), 10), dtype=float)
    for row_i, char in enumerate(chars):
        if is_missing(char):
            lik[row_i, :] = 1.0
        else:
            c = str(char).upper()
            if c in IUPAC_TO_GT_INDICES:
                for idx in IUPAC_TO_GT_INDICES[c]:
                    lik[row_i, idx] = 1.0
            else:
                # Unknown character -> treat as missing
                lik[row_i, :] = 1.0
    return normalize_rows(lik)


# Reverse map: genotype state index -> IUPAC code
GT_INDEX_TO_IUPAC = {i: GENOTYPE_IUPAC[s] for i, s in enumerate(GENOTYPE_STATES)}
