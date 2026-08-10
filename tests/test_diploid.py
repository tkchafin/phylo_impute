"""Tests for the 10-state diploid genotype CTMC model (GTGTR4)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import toytree as tt

from phylo_impute.core import (
    QCache,
    build_q10_from_gtr,
    ctmc_pruning_loglik_diploid,
    estimate_genotype_freqs,
    phylogenetic_imputation_diploid,
    stationary_pi_from_q,
)
from phylo_impute.io import extract_gtr_rates_from_q4
from phylo_impute.utils import (
    GENOTYPE_STATES,
    GT_INDEX_TO_IUPAC,
    GT_RATE_SYMMETRY,
    IUPAC_TO_GT_INDICES,
    likelihood_matrix_for_genotypes,
    normalize_rows,
)

DATA_DIR = Path(__file__).parent / "data"


# --- Fixtures ---

@pytest.fixture
def gtr_rates():
    with open(DATA_DIR / "diploid_gtr_rates.json") as f:
        return np.array(json.load(f)["rates"])


@pytest.fixture
def freqs_10():
    with open(DATA_DIR / "diploid_freqs.json") as f:
        return np.array(json.load(f)["freqs"])


@pytest.fixture
def q10_matrix(gtr_rates, freqs_10):
    return build_q10_from_gtr(gtr_rates, freqs_10)


@pytest.fixture
def diploid_tree():
    return tt.tree(str(DATA_DIR / "diploid_tree.nwk"))


@pytest.fixture
def original_genotypes():
    with open(DATA_DIR / "diploid_original.json") as f:
        return json.load(f)


@pytest.fixture
def masked_genotypes():
    with open(DATA_DIR / "diploid_masked.json") as f:
        return json.load(f)


@pytest.fixture
def masked_positions():
    with open(DATA_DIR / "diploid_masked_positions.json") as f:
        return [tuple(x) for x in json.load(f)]


@pytest.fixture
def reference_q10():
    with open(DATA_DIR / "diploid_q10.json") as f:
        data = json.load(f)
        return np.array(data["Q"])


# --- Test: Q matrix construction ---

class TestBuildQ10:
    def test_shape(self, q10_matrix):
        assert q10_matrix.shape == (10, 10)

    def test_rows_sum_to_zero(self, q10_matrix):
        row_sums = q10_matrix.to_numpy().sum(axis=1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-12)

    def test_off_diagonal_non_negative(self, q10_matrix):
        Q = q10_matrix.to_numpy()
        off_diag = Q[~np.eye(10, dtype=bool)]
        assert np.all(off_diag >= 0.0)

    def test_diagonal_negative(self, q10_matrix):
        Q = q10_matrix.to_numpy()
        diag = np.diag(Q)
        assert np.all(diag < 0.0)

    def test_forbidden_transitions_zero(self, q10_matrix):
        """Hom<->Hom transitions and double-mutation Het<->Het should be 0."""
        Q = q10_matrix.to_numpy()
        for i in range(10):
            for j in range(10):
                if i == j:
                    continue
                if GT_RATE_SYMMETRY[i, j] == -1:
                    assert Q[i, j] == 0.0, f"Q[{i},{j}] should be 0 (forbidden)"

    def test_mean_rate_is_one(self, q10_matrix, freqs_10):
        Q = q10_matrix.to_numpy()
        pi = stationary_pi_from_q(Q)
        mean_rate = float(-np.dot(pi, np.diag(Q)))
        assert math.isclose(mean_rate, 1.0, rel_tol=0.05)

    def test_matches_reference(self, q10_matrix, reference_q10):
        """Check our Q matches the one saved during fixture generation."""
        np.testing.assert_allclose(
            q10_matrix.to_numpy(), reference_q10, atol=1e-10
        )

    def test_index_labels(self, q10_matrix):
        assert list(q10_matrix.index) == list(GENOTYPE_STATES)
        assert list(q10_matrix.columns) == list(GENOTYPE_STATES)

    def test_jc_like_rates(self):
        """With equal rates and equal freqs, should get JC-like Q."""
        rates = np.ones(6)
        freqs = np.ones(10) / 10.0
        Q = build_q10_from_gtr(rates, freqs)
        q = Q.to_numpy()
        # Check rows sum to 0
        np.testing.assert_allclose(q.sum(axis=1), 0.0, atol=1e-12)
        # All allowed transitions should have same rate
        allowed_rates = set()
        for i in range(10):
            for j in range(10):
                if i != j and GT_RATE_SYMMETRY[i, j] >= 0:
                    allowed_rates.add(round(q[i, j], 10))
        assert len(allowed_rates) == 1


# --- Test: QCache with 10x10 ---

class TestQCache10:
    def test_p_shape(self, q10_matrix):
        qc = QCache(q10_matrix)
        P = qc.P(0.1)
        assert P.shape == (10, 10)

    def test_p_rows_sum_to_one(self, q10_matrix):
        qc = QCache(q10_matrix)
        P = qc.P(0.1)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-10)

    def test_p_non_negative(self, q10_matrix):
        qc = QCache(q10_matrix)
        P = qc.P(0.1)
        assert np.all(P >= 0.0)

    def test_p_identity_at_zero(self, q10_matrix):
        qc = QCache(q10_matrix)
        P = qc.P(0.0)
        np.testing.assert_allclose(P, np.eye(10), atol=1e-10)

    def test_p_converges_to_stationary(self, q10_matrix):
        """At very large t, all rows of P should approach the stationary dist."""
        qc = QCache(q10_matrix)
        P = qc.P(100.0)
        pi = stationary_pi_from_q(q10_matrix.to_numpy())
        for i in range(10):
            np.testing.assert_allclose(P[i], pi, atol=0.01)

    def test_caching(self, q10_matrix):
        qc = QCache(q10_matrix)
        P1 = qc.P(0.1)
        P2 = qc.P(0.1)
        assert P1 is P2  # Same object from cache


# --- Test: Genotype frequency estimation ---

class TestEstimateGenotypeFreqs:
    def test_uniform_data(self):
        genotypes = {
            "s1": ["A", "C", "G", "T", "M", "R", "W", "S", "Y", "K"],
            "s2": ["A", "C", "G", "T", "M", "R", "W", "S", "Y", "K"],
        }
        freqs = estimate_genotype_freqs(genotypes, pseudocount=0.0)
        assert len(freqs) == 10
        assert math.isclose(sum(freqs), 1.0, abs_tol=1e-10)
        # Each state observed exactly 2 times
        np.testing.assert_allclose(freqs, np.ones(10) / 10, atol=1e-10)

    def test_all_missing(self):
        genotypes = {"s1": ["N", "N", "N"]}
        freqs = estimate_genotype_freqs(genotypes, pseudocount=1.0)
        assert len(freqs) == 10
        assert math.isclose(sum(freqs), 1.0, abs_tol=1e-10)

    def test_sums_to_one(self, masked_genotypes):
        freqs = estimate_genotype_freqs(masked_genotypes)
        assert math.isclose(sum(freqs), 1.0, abs_tol=1e-10)
        assert np.all(freqs > 0)


# --- Test: Likelihood matrix for genotypes ---

class TestLikelihoodMatrixForGenotypes:
    def test_homozygous(self):
        lik = likelihood_matrix_for_genotypes(["A"])
        expected = np.zeros(10)
        expected[0] = 1.0
        np.testing.assert_allclose(lik[0], expected)

    def test_heterozygous(self):
        lik = likelihood_matrix_for_genotypes(["M"])  # AC
        expected = np.zeros(10)
        expected[4] = 1.0  # AC index
        np.testing.assert_allclose(lik[0], expected)

    def test_missing(self):
        lik = likelihood_matrix_for_genotypes(["N"])
        # Should be uniform
        np.testing.assert_allclose(lik[0], np.ones(10) / 10.0)

    def test_batch(self):
        chars = ["A", "C", "M", "R", "N"]
        lik = likelihood_matrix_for_genotypes(chars)
        assert lik.shape == (5, 10)
        np.testing.assert_allclose(lik.sum(axis=1), 1.0, atol=1e-10)


# --- Test: GTR rate extraction ---

class TestExtractGTRRates:
    def test_jc_rates(self):
        """JC model should give equal rates."""
        Q = np.array([
            [-1.0, 1/3, 1/3, 1/3],
            [1/3, -1.0, 1/3, 1/3],
            [1/3, 1/3, -1.0, 1/3],
            [1/3, 1/3, 1/3, -1.0],
        ])
        q_df = pd.DataFrame(Q, index=["A", "C", "G", "T"], columns=["A", "C", "G", "T"])
        rates = extract_gtr_rates_from_q4(q_df)
        # All rates should be equal (normalized to GT=1)
        np.testing.assert_allclose(rates, np.ones(6), atol=1e-10)

    def test_hky_rates(self):
        """HKY model: transitions (AG, CT) should differ from transversions."""
        kappa = 4.0
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        # Build HKY Q
        Q = np.zeros((4, 4))
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        rate_vals = [1.0, kappa, 1.0, 1.0, kappa, 1.0]  # AC,AG,AT,CG,CT,GT
        for (i, j), r in zip(pairs, rate_vals):
            Q[i, j] = r * pi[j]
            Q[j, i] = r * pi[i]
        np.fill_diagonal(Q, -Q.sum(axis=1))
        mean_rate = -np.dot(pi, np.diag(Q))
        Q /= mean_rate

        q_df = pd.DataFrame(Q, index=["A", "C", "G", "T"], columns=["A", "C", "G", "T"])
        rates = extract_gtr_rates_from_q4(q_df)

        # AG and CT should be kappa, rest 1 (normalized to GT=1)
        assert math.isclose(rates[1], kappa, rel_tol=0.01)  # AG
        assert math.isclose(rates[4], kappa, rel_tol=0.01)  # CT
        assert math.isclose(rates[0], 1.0, rel_tol=0.01)    # AC
        assert math.isclose(rates[5], 1.0, rel_tol=0.01)    # GT

    def test_recovers_known_rates(self, gtr_rates, freqs_10):
        """Build Q4 from known GTR rates and verify extraction recovers them."""
        # Build a simple Q4 from the GTR rates
        pi4 = np.array([0.3, 0.2, 0.2, 0.3])
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        Q = np.zeros((4, 4))
        for (i, j), r in zip(pairs, gtr_rates):
            Q[i, j] = r * pi4[j]
            Q[j, i] = r * pi4[i]
        np.fill_diagonal(Q, -Q.sum(axis=1))
        mean_rate = -np.dot(pi4, np.diag(Q))
        Q /= mean_rate

        q_df = pd.DataFrame(Q, index=["A", "C", "G", "T"], columns=["A", "C", "G", "T"])
        recovered = extract_gtr_rates_from_q4(q_df)

        # Normalized rates should match (both normalized to GT=1)
        expected = gtr_rates / gtr_rates[5]
        np.testing.assert_allclose(recovered, expected, rtol=0.01)


# --- Test: Rate symmetry matrix ---

class TestRateSymmetry:
    def test_symmetric(self):
        for i in range(10):
            for j in range(10):
                assert GT_RATE_SYMMETRY[i, j] == GT_RATE_SYMMETRY[j, i]

    def test_diagonal(self):
        for i in range(10):
            assert GT_RATE_SYMMETRY[i, i] == -2

    def test_hom_hom_forbidden(self):
        """All homozygote-homozygote transitions should be forbidden."""
        hom = [0, 1, 2, 3]  # AA, CC, GG, TT
        for i in hom:
            for j in hom:
                if i != j:
                    assert GT_RATE_SYMMETRY[i, j] == -1

    def test_allowed_count(self):
        """Count allowed transitions: should be 24 pairs (upper triangle)."""
        count = 0
        for i in range(10):
            for j in range(i + 1, 10):
                if GT_RATE_SYMMETRY[i, j] >= 0:
                    count += 1
        assert count == 24


# --- Test: Diploid imputation ---

class TestPhylogeneticImputationDiploid:
    def test_fills_all_missing(self, diploid_tree, masked_genotypes, q10_matrix):
        imputed, likelihoods = phylogenetic_imputation_diploid(
            tree=diploid_tree,
            genotypes=masked_genotypes,
            q_matrix=q10_matrix,
        )
        # No missing values remain
        for sample, seq in imputed.items():
            for site_i, val in enumerate(seq):
                assert val != "N", f"Missing value at {sample}[{site_i}]"

    def test_posteriors_sum_to_one(self, diploid_tree, masked_genotypes, q10_matrix):
        _, likelihoods = phylogenetic_imputation_diploid(
            tree=diploid_tree,
            genotypes=masked_genotypes,
            q_matrix=q10_matrix,
        )
        for key, posterior in likelihoods.items():
            assert len(posterior) == 10
            assert math.isclose(sum(posterior), 1.0, abs_tol=0.01)

    def test_imputation_accuracy(
        self, diploid_tree, masked_genotypes, original_genotypes,
        masked_positions, q10_matrix
    ):
        imputed, _ = phylogenetic_imputation_diploid(
            tree=diploid_tree,
            genotypes=masked_genotypes,
            q_matrix=q10_matrix,
        )
        n_correct = 0
        n_total = 0
        for tip, site in masked_positions:
            if tip not in imputed:
                continue
            truth = original_genotypes[tip][site]
            pred = imputed[tip][site]
            n_total += 1
            if pred == truth:
                n_correct += 1

        accuracy = n_correct / n_total if n_total > 0 else 0
        # Should be at least 30% on simulated data
        assert accuracy >= 0.30, f"Accuracy {accuracy:.3f} < 0.30"

    def test_unanimous_imputation(self, q10_matrix):
        """When all observed tips have same genotype, imputed should match."""
        tree = tt.tree("((t1:0.01,t2:0.01):0.01,(t3:0.01,t4:0.01):0.01);")
        genotypes = {
            "t1": ["N"],
            "t2": ["R"],  # AG
            "t3": ["R"],  # AG
            "t4": ["R"],  # AG
        }
        imputed, likelihoods = phylogenetic_imputation_diploid(
            tree=tree, genotypes=genotypes, q_matrix=q10_matrix
        )
        assert imputed["t1"][0] == "R"

    def test_valid_iupac_codes(self, diploid_tree, masked_genotypes, q10_matrix):
        """All imputed values should be valid IUPAC genotype codes."""
        valid_codes = set(GT_INDEX_TO_IUPAC.values())
        imputed, _ = phylogenetic_imputation_diploid(
            tree=diploid_tree,
            genotypes=masked_genotypes,
            q_matrix=q10_matrix,
        )
        for sample, seq in imputed.items():
            for site_i, val in enumerate(seq):
                assert val in valid_codes, f"Invalid code '{val}' at {sample}[{site_i}]"


# --- Test: Diploid log-likelihood ---

class TestCTMCPruningLoglikDiploid:
    def test_finite(self, diploid_tree, masked_genotypes, q10_matrix):
        ll = ctmc_pruning_loglik_diploid(
            tree=diploid_tree,
            genotypes=masked_genotypes,
            q_matrix=q10_matrix,
        )
        assert math.isfinite(ll)

    def test_negative(self, diploid_tree, masked_genotypes, q10_matrix):
        ll = ctmc_pruning_loglik_diploid(
            tree=diploid_tree,
            genotypes=masked_genotypes,
            q_matrix=q10_matrix,
        )
        assert ll < 0

    def test_original_higher_than_random(
        self, diploid_tree, original_genotypes, q10_matrix
    ):
        """Log-likelihood of original data should be higher than shuffled data."""
        ll_orig = ctmc_pruning_loglik_diploid(
            tree=diploid_tree,
            genotypes=original_genotypes,
            q_matrix=q10_matrix,
        )
        # Shuffle genotypes
        rng = np.random.default_rng(99)
        shuffled = {}
        for tip, seq in original_genotypes.items():
            arr = list(seq)
            rng.shuffle(arr)
            shuffled[tip] = arr

        ll_shuffled = ctmc_pruning_loglik_diploid(
            tree=diploid_tree,
            genotypes=shuffled,
            q_matrix=q10_matrix,
        )
        assert ll_orig > ll_shuffled
