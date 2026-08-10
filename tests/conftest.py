from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import toytree as tt

from phylo_impute.utils import STATE_ORDER


@pytest.fixture
def jc_q_matrix() -> pd.DataFrame:
    """Jukes-Cantor Q matrix (mean rate = 1 after scaling)."""
    Q = np.array([
        [-1.0, 1/3, 1/3, 1/3],
        [1/3, -1.0, 1/3, 1/3],
        [1/3, 1/3, -1.0, 1/3],
        [1/3, 1/3, 1/3, -1.0],
    ])
    return pd.DataFrame(Q, index=STATE_ORDER, columns=STATE_ORDER)


@pytest.fixture
def simple_tree() -> tt.ToyTree:
    """4-tip tree: ((A:0.1,B:0.2):0.3,(C:0.15,D:0.25):0.1);"""
    return tt.tree("((A:0.1,B:0.2):0.3,(C:0.15,D:0.25):0.1);")


@pytest.fixture
def genotypes_with_missing() -> dict:
    """Genotype dict with missing data at known positions."""
    return {
        "A": ["A", "C", "G", "T", "N"],
        "B": ["A", "C", "G", "T", "A"],
        "C": ["A", "N", "G", "T", "A"],
        "D": ["A", "C", "G", "N", "A"],
    }


@pytest.fixture
def star_tree_unanimous() -> tuple:
    """Star tree where all observed tips have 'C' -> imputed should be 'C'."""
    tree = tt.tree("((tip1:0.1,tip2:0.1):0.0,(tip3:0.1,tip4:0.1):0.0);")
    genotypes = {
        "tip1": ["N"],
        "tip2": ["C"],
        "tip3": ["C"],
        "tip4": ["C"],
    }
    return tree, genotypes
