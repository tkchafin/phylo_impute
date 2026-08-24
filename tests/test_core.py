"""Unit tests for core CTMC algorithm."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phylo_impute.core import (
    QCache,
    ctmc_pruning_loglik,
    phylogenetic_imputation,
    repair_and_scale_q,
    stationary_pi_from_q,
)
from phylo_impute.utils import STATE_ORDER, normalize_rows


class TestStationary:
    def test_jc_uniform(self, jc_q_matrix):
        pi = stationary_pi_from_q(jc_q_matrix.to_numpy())
        np.testing.assert_allclose(pi, 0.25, atol=1e-10)

    def test_sums_to_one(self, jc_q_matrix):
        pi = stationary_pi_from_q(jc_q_matrix.to_numpy())
        assert abs(pi.sum() - 1.0) < 1e-12

    def test_nonnegative(self, jc_q_matrix):
        pi = stationary_pi_from_q(jc_q_matrix.to_numpy())
        assert np.all(pi >= 0)


class TestRepairAndScaleQ:
    def test_rows_sum_zero(self, jc_q_matrix):
        Q = repair_and_scale_q(jc_q_matrix)
        np.testing.assert_allclose(
            Q.to_numpy().sum(axis=1), 0.0, atol=1e-12
        )

    def test_mean_rate_one(self, jc_q_matrix):
        Q = repair_and_scale_q(jc_q_matrix)
        pi = stationary_pi_from_q(Q.to_numpy())
        mean_rate = float(-np.dot(pi, np.diag(Q.to_numpy())))
        assert abs(mean_rate - 1.0) < 1e-10

    def test_negative_offdiag_clamped(self):
        Q = np.array([
            [-1.0, 0.5, -0.1, 0.6],
            [0.3, -1.0, 0.4, 0.3],
            [0.3, 0.3, -1.0, 0.4],
            [0.3, 0.3, 0.4, -1.0],
        ])
        q_df = pd.DataFrame(Q, index=STATE_ORDER, columns=STATE_ORDER)
        result = repair_and_scale_q(q_df)
        off_mask = ~np.eye(4, dtype=bool)
        assert np.all(result.to_numpy()[off_mask] >= 0)

    def test_missing_states_raises(self):
        Q = pd.DataFrame(
            np.eye(3), index=["A", "C", "G"], columns=["A", "C", "G"]
        )
        with pytest.raises(ValueError, match="missing"):
            repair_and_scale_q(Q)


class TestQCache:
    def test_stochastic_matrix(self, jc_q_matrix):
        qc = QCache(jc_q_matrix)
        P = qc.P(0.5)
        # Rows sum to 1
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-10)
        # All non-negative
        assert np.all(P >= 0)

    def test_identity_at_t_zero(self, jc_q_matrix):
        qc = QCache(jc_q_matrix)
        P = qc.P(0.0)
        np.testing.assert_allclose(P, np.eye(4), atol=1e-10)

    def test_caching(self, jc_q_matrix):
        qc = QCache(jc_q_matrix)
        P1 = qc.P(0.3)
        P2 = qc.P(0.3)
        assert P1 is P2

    def test_rate_scaling(self, jc_q_matrix):
        qc = QCache(jc_q_matrix)
        P_fast = qc.P(0.5, rate=2.0)
        P_long = qc.P(1.0, rate=1.0)
        np.testing.assert_allclose(P_fast, P_long, atol=1e-10)


class TestNormalizeRows:
    def test_normal(self):
        mat = np.array([[1.0, 2.0, 3.0]])
        result = normalize_rows(mat)
        np.testing.assert_allclose(result.sum(axis=1), 1.0)

    def test_zero_row_fallback(self):
        mat = np.array([[0.0, 0.0, 0.0, 0.0]])
        result = normalize_rows(mat)
        np.testing.assert_allclose(result, 0.25)


class TestPhylogeneticImputation:
    def test_fills_all_missing(self, simple_tree, genotypes_with_missing, jc_q_matrix):
        q = repair_and_scale_q(jc_q_matrix)
        imputed, likelihoods = phylogenetic_imputation(
            simple_tree, genotypes_with_missing, q
        )
        for sample, seq in imputed.items():
            for i, val in enumerate(seq):
                assert val in "ACGT", f"Missing value at {sample}:{i}"

    def test_posteriors_sum_to_one(self, simple_tree, genotypes_with_missing, jc_q_matrix):
        q = repair_and_scale_q(jc_q_matrix)
        _, likelihoods = phylogenetic_imputation(
            simple_tree, genotypes_with_missing, q
        )
        for key, posterior in likelihoods.items():
            assert abs(posterior.sum() - 1.0) < 1e-10, f"Posterior doesn't sum to 1 for {key}"

    def test_unanimous_tips(self, star_tree_unanimous, jc_q_matrix):
        tree, genotypes = star_tree_unanimous
        q = repair_and_scale_q(jc_q_matrix)
        imputed, likelihoods = phylogenetic_imputation(tree, genotypes, q)
        assert imputed["tip1"][0] == "C"

    def test_no_missing_noop(self, simple_tree, jc_q_matrix):
        genotypes = {
            "A": ["A", "C"],
            "B": ["A", "C"],
            "C": ["A", "C"],
            "D": ["A", "C"],
        }
        q = repair_and_scale_q(jc_q_matrix)
        imputed, likelihoods = phylogenetic_imputation(
            simple_tree, genotypes, q
        )
        assert len(likelihoods) == 0

    def test_no_common_samples_raises(self, simple_tree, jc_q_matrix):
        genotypes = {"X": ["A"], "Y": ["C"]}
        q = repair_and_scale_q(jc_q_matrix)
        with pytest.raises(ValueError, match="No samples in common"):
            phylogenetic_imputation(simple_tree, genotypes, q)

    def test_min_posterior_retains_uncertain(self, simple_tree, jc_q_matrix):
        """With a very high threshold, uncertain sites should stay missing."""
        genotypes = {
            "A": ["A", "N"],
            "B": ["C", "N"],
            "C": ["G", "N"],
            "D": ["T", "N"],
        }
        q = repair_and_scale_q(jc_q_matrix)
        # All four tips disagree at site 1 -> posterior ~0.25 per state
        imputed, likelihoods = phylogenetic_imputation(
            simple_tree, genotypes, q, min_posterior=0.99
        )
        from phylo_impute.utils import is_missing
        # Site 1 should remain missing for all samples (no state > 0.99)
        for sample in ["A", "B", "C", "D"]:
            assert is_missing(imputed[sample][1])
        # Posteriors should still be recorded
        assert len(likelihoods) > 0

    def test_min_posterior_none_imputes_all(self, simple_tree, genotypes_with_missing, jc_q_matrix):
        """Default (min_posterior=None) should impute everything."""
        q = repair_and_scale_q(jc_q_matrix)
        imputed, _ = phylogenetic_imputation(
            simple_tree, genotypes_with_missing, q, min_posterior=None
        )
        from phylo_impute.utils import is_missing
        for sample, seq in imputed.items():
            for val in seq:
                assert not is_missing(val)

    def test_min_posterior_zero_imputes_all(self, star_tree_unanimous, jc_q_matrix):
        """min_posterior near zero should impute everything."""
        tree, genotypes = star_tree_unanimous
        q = repair_and_scale_q(jc_q_matrix)
        imputed, _ = phylogenetic_imputation(tree, genotypes, q, min_posterior=0.01)
        assert imputed["tip1"][0] == "C"

    def test_min_posterior_posteriors_always_recorded(self, simple_tree, jc_q_matrix):
        """Posteriors are recorded even for filtered sites."""
        genotypes = {
            "A": ["N"], "B": ["A"], "C": ["C"], "D": ["G"],
        }
        q = repair_and_scale_q(jc_q_matrix)
        imputed, likelihoods = phylogenetic_imputation(
            simple_tree, genotypes, q, min_posterior=0.99
        )
        assert ("A", 0) in likelihoods
        assert abs(likelihoods[("A", 0)].sum() - 1.0) < 1e-10


class TestCTMCPruningLoglik:
    def test_finite(self, simple_tree, jc_q_matrix):
        genotypes = {
            "A": ["A", "C", "G"],
            "B": ["A", "C", "G"],
            "C": ["A", "C", "G"],
            "D": ["A", "C", "G"],
        }
        q = repair_and_scale_q(jc_q_matrix)
        ll = ctmc_pruning_loglik(simple_tree, genotypes, q)
        assert np.isfinite(ll)
        assert ll < 0  # Log-likelihood should be negative

    def test_identical_seqs_higher_ll(self, simple_tree, jc_q_matrix):
        """Identical sequences should have higher likelihood than divergent."""
        q = repair_and_scale_q(jc_q_matrix)
        identical = {
            "A": ["A", "A", "A"],
            "B": ["A", "A", "A"],
            "C": ["A", "A", "A"],
            "D": ["A", "A", "A"],
        }
        divergent = {
            "A": ["A", "C", "G"],
            "B": ["C", "G", "T"],
            "C": ["G", "T", "A"],
            "D": ["T", "A", "C"],
        }
        ll_ident = ctmc_pruning_loglik(simple_tree, identical, q)
        ll_div = ctmc_pruning_loglik(simple_tree, divergent, q)
        assert ll_ident > ll_div
