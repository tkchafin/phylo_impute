from __future__ import annotations

import operator
from collections.abc import ItemsView, Mapping, ValuesView
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.linalg
import toytree as tt

import warnings

from phylo_impute.utils import (
    CHAR_MAP,
    GENOTYPE_STATES,
    GT_INDEX_TO_IUPAC,
    GT_RATE_SYMMETRY,
    MIN_BRANCH_LENGTH,
    MISSING_TOKENS,
    NUC_MAP,
    STATE_ORDER,
    is_missing,
    likelihood_matrix_for_chars,
    likelihood_matrix_for_genotypes,
    normalize_rows,
)


def _reconcile_tree_genotypes(tree, genotypes):
    """Check sample overlap between tree and genotypes, warn about mismatches.

    Returns set of common sample names. Warns if some genotype samples
    are missing from the tree (those samples will be skipped during imputation
    and returned unmodified). Raises ValueError if no overlap at all.
    """
    tree_tips = set(tree.get_tip_labels())
    geno_samples = set(genotypes.keys())
    common = tree_tips & geno_samples

    data_only = geno_samples - tree_tips
    if data_only:
        warnings.warn(
            f"{len(data_only)} sample(s) in genotype data not found in tree "
            f"and will be returned unmodified (not imputed): "
            f"{', '.join(sorted(data_only)[:10])}"
            f"{'...' if len(data_only) > 10 else ''}",
            stacklevel=3,
        )

    if not common:
        tree_example = ", ".join(sorted(tree_tips)[:5])
        geno_example = ", ".join(sorted(geno_samples)[:5])
        raise ValueError(
            f"No samples in common between tree and genotypes. "
            f"Tree tips: [{tree_example}{'...' if len(tree_tips) > 5 else ''}], "
            f"Genotype samples: [{geno_example}{'...' if len(geno_samples) > 5 else ''}]"
        )

    return common


class QCache:
    """Eigendecompose Q once, cache P(t) = exp(Qt) by branch length."""

    def __init__(self, q_df: pd.DataFrame) -> None:
        m = np.asarray(q_df, dtype=float)
        evals, V = scipy.linalg.eig(m)
        Vinv = scipy.linalg.inv(V)
        self.evals = evals
        self.V = V
        self.Vinv = Vinv
        self._cache: dict[float, np.ndarray] = {}

    def P(self, t: float, rate: float = 1.0) -> np.ndarray:
        key = round(float(rate) * float(t), 12)
        if key in self._cache:
            return self._cache[key]
        expo = np.exp(self.evals * key)
        P = (self.V * expo) @ self.Vinv
        P = np.real_if_close(P, tol=1e5)
        P[P < 0.0] = 0.0
        P /= P.sum(axis=1, keepdims=True).clip(min=1.0)
        self._cache[key] = P
        return P


def stationary_pi_from_q(Q: np.ndarray) -> np.ndarray:
    """Left eigenvector of Q^T at eigenvalue 0."""
    w, v = scipy.linalg.eig(Q.T)
    k = int(np.argmin(np.abs(w)))
    pi = np.real(v[:, k])
    # Eigenvector sign is arbitrary; ensure it's positive
    if pi.sum() < 0:
        pi = -pi
    pi = np.maximum(pi, 0.0)
    s = float(pi.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones(Q.shape[0]) / Q.shape[0]
    return pi / s


def repair_and_scale_q(
    q_df: pd.DataFrame, target_mean_rate: float = 1.0
) -> pd.DataFrame:
    """Clamp negative off-diags, reset diags, rescale to target mean rate."""
    missing = set(STATE_ORDER) - set(q_df.index) | set(STATE_ORDER) - set(
        q_df.columns
    )
    if missing:
        raise ValueError(
            f"Q must have states {STATE_ORDER}, missing: {sorted(missing)}"
        )

    Q = q_df.loc[STATE_ORDER, STATE_ORDER].to_numpy(dtype=float, copy=True)

    off_mask = ~np.eye(Q.shape[0], dtype=bool)
    Q[off_mask] = np.maximum(Q[off_mask], 0.0)

    row_off_sum = Q.sum(axis=1) - np.diag(Q)
    np.fill_diagonal(Q, -row_off_sum)

    zero_rows = row_off_sum <= 0
    if np.any(zero_rows):
        eps = 1e-8
        Q[zero_rows, :] = 0.0
        for i in np.where(zero_rows)[0]:
            Q[i, :] = eps / (Q.shape[0] - 1)
            Q[i, i] = -eps

    if not np.allclose(Q.sum(axis=1), 0.0, atol=1e-12):
        raise RuntimeError("Q rows do not sum to zero after repair.")

    pi = stationary_pi_from_q(Q)
    mean_rate = float(-np.dot(pi, np.diag(Q)))

    if np.isfinite(mean_rate) and mean_rate > 0:
        Q /= mean_rate / float(target_mean_rate)

    return pd.DataFrame(Q, index=STATE_ORDER, columns=STATE_ORDER)


def bin_site_rates(
    rates: List[float], n_bins: int,
) -> List[float]:
    """Bin site rates into n_bins quantile-based categories.

    Each rate is replaced by the mean of its bin, so only n_bins distinct
    rate values remain.  This dramatically reduces the number of unique
    P(t) matrices needed during variable-rate imputation.
    """
    arr = np.asarray(rates, dtype=float)
    edges = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.digitize(arr, edges, right=True)
    bin_idx = np.clip(bin_idx, 1, n_bins)
    bin_means = np.array([
        arr[bin_idx == b].mean() for b in range(1, n_bins + 1)
    ])
    return bin_means[bin_idx - 1].tolist()


_NUC_CALLS = np.array([NUC_MAP[i] for i in range(4)], dtype=object)
_GT_CALLS = np.array([GT_INDEX_TO_IUPAC[i] for i in range(10)], dtype=object)
_ITER_CHUNK = 1 << 16


def _index_dtype(n):
    return np.int32 if n < np.iinfo(np.int32).max else np.int64


class PosteriorMap(Mapping):
    """Read-only mapping ``(sample, site) -> posterior array``, backed by arrays.

    Returned by :func:`phylogenetic_imputation` and
    :func:`phylogenetic_imputation_diploid` in place of the plain dict used
    before 0.2.0. Reading works as before: ``post[(sample, site)]``, ``in``,
    ``len``, ``.get``, and iteration, ``.keys()``, ``.items()`` and
    ``.values()`` in the same order. Assignment is not supported. Posteriors
    are stored as one array row per imputed cell rather than one Python
    object per cell.

    Vectorised callers can use ``sample_names``, ``sample_index``, ``sites``
    and ``posteriors`` directly. ``to_dict()`` returns a plain dict.
    """

    def __init__(self, sample_names, sample_index, sites, posteriors, n_sites):
        self._names = list(sample_names)
        self._row_of = {s: i for i, s in enumerate(self._names)}
        self._rows = np.asarray(sample_index)
        self._sites = np.asarray(sites)
        self._post = np.asarray(posteriors, dtype=float)
        self._n_sites = int(n_sites)
        self._index = None  # (n_samples, n_sites) cell -> row; built on first lookup

    @property
    def sample_names(self) -> List[str]:
        """Sample names; ``sample_index`` indexes into this list."""
        return list(self._names)

    @property
    def sample_index(self) -> np.ndarray:
        """For each imputed cell, the index of its sample in ``sample_names``."""
        return self._rows

    @property
    def sites(self) -> np.ndarray:
        """For each imputed cell, its 0-based site (column) index."""
        return self._sites

    @property
    def posteriors(self) -> np.ndarray:
        """``(n_cells, n_states)`` posterior probabilities, one row per imputed cell."""
        return self._post

    def to_dict(self) -> Dict[Tuple[str, int], np.ndarray]:
        """Plain dict holding a copy of each posterior."""
        return {key: post.copy() for key, post in self.items()}

    def _find(self, key) -> int:
        if not isinstance(key, tuple) or len(key) != 2:
            return -1
        sample, site = key
        try:
            row = self._row_of[sample]
        except (KeyError, TypeError):
            return -1
        try:
            site = operator.index(site)
        except TypeError:
            try:
                as_int = int(site)
            except (TypeError, ValueError, OverflowError):
                return -1
            if as_int != site:
                return -1
            site = as_int
        if not 0 <= site < self._n_sites:
            return -1
        if self._index is None:
            dtype = _index_dtype(len(self._rows))
            index = np.full((len(self._names), self._n_sites), -1, dtype=dtype)
            index[self._rows, self._sites] = np.arange(len(self._rows), dtype=dtype)
            self._index = index
        return int(self._index[row, site])

    def __getitem__(self, key):
        k = self._find(key)
        if k < 0:
            raise KeyError(key)
        return self._post[k]

    def __contains__(self, key):
        return self._find(key) >= 0

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        names = self._names
        for start in range(0, len(self._rows), _ITER_CHUNK):
            stop = start + _ITER_CHUNK
            rows = self._rows[start:stop].tolist()
            sites = self._sites[start:stop].tolist()
            for r, s in zip(rows, sites):
                yield (names[r], s)

    def items(self):
        return _PosteriorItemsView(self)

    def values(self):
        return _PosteriorValuesView(self)

    def __repr__(self):
        return (
            f"{type(self).__name__}({len(self)} imputed cells, "
            f"{self._post.shape[1]} states)"
        )


class _PosteriorItemsView(ItemsView):
    def __iter__(self):
        post = self._mapping._post
        for k, key in enumerate(self._mapping):
            yield key, post[k]


class _PosteriorValuesView(ValuesView):
    def __iter__(self):
        post = self._mapping._post
        for k in range(len(post)):
            yield post[k]


def _encode_tokens(seq_array, lik_fn):
    """Map every cell of ``seq_array`` to a code over its distinct tokens.

    Per-token work is then done once per distinct token instead of once per
    cell. Returns:
        codes: integer array with ``seq_array``'s shape.
        lik_table: ``lik_fn(tokens)``, the leaf likelihood row of each token.
        impute_flag: token exactly equals one of ``MISSING_TOKENS``; these
            cells are imputed.
        still_missing: ``is_missing(token)``, used by the completeness check.
    """
    flat = seq_array.ravel()
    codes, tokens = pd.factorize(flat, use_na_sentinel=False)
    tokens = np.asarray(tokens, dtype=object)
    if not all(isinstance(t, str) for t in tokens):
        # Non-string tokens can compare equal yet print differently (e.g. -9
        # and -9.0); key on type too so each token keeps its own handling.
        seen: Dict[Tuple[type, object], int] = {}
        codes = np.fromiter(
            (seen.setdefault((type(x), x), len(seen)) for x in flat),
            dtype=np.intp,
            count=flat.size,
        )
        tokens = np.empty(len(seen), dtype=object)
        for i, (_, x) in enumerate(seen):
            tokens[i] = x
    codes = codes.astype(np.min_scalar_type(max(len(tokens) - 1, 0)), copy=False)
    impute_flag = np.zeros(len(tokens), dtype=bool)
    for token in MISSING_TOKENS:
        impute_flag |= tokens == token
    still_missing = np.array([is_missing(t) for t in tokens], dtype=bool)
    return codes.reshape(seq_array.shape), lik_fn(tokens), impute_flag, still_missing


class _PosteriorSink:
    """Preallocated per-cell posterior store, filled in imputation order."""

    def __init__(self, n_cells, n_samples, n_sites, n_states):
        self.rows = np.empty(n_cells, dtype=_index_dtype(n_samples))
        self.sites = np.empty(n_cells, dtype=_index_dtype(n_sites))
        self.post = np.empty((n_cells, n_states), dtype=float)
        self.n = 0

    def add(self, row_i, sites, post):
        k, m = self.n, len(sites)
        self.rows[k:k + m] = row_i
        self.sites[k:k + m] = sites
        self.post[k:k + m] = post
        self.n = k + m


def _two_pass(postorder_nodes, preorder_nodes, P_by_node, leaf_lik, n_cols,
              n_states, root_prior, root_idx):
    """Down (postorder) and up (preorder) messages for one block of columns.

    ``leaf_lik(node)`` returns a leaf's ``(n_cols, n_states)`` likelihoods.
    """
    down_liks = {}
    for node in postorder_nodes:
        if node.is_leaf():
            down_liks[node.idx] = leaf_lik(node)
        else:
            msg = np.ones((n_cols, n_states), dtype=float)
            for child in node.children:
                P = P_by_node[child.idx]
                msg *= down_liks[child.idx] @ P.T
            down_liks[node.idx] = normalize_rows(msg)

    up_liks = {
        root_idx: np.broadcast_to(root_prior, (n_cols, n_states)).copy()
    }
    for node in preorder_nodes:
        if node.is_root():
            continue
        parent = node.up
        sib_prod = np.ones((n_cols, n_states), dtype=float)
        for sib in parent.children:
            if sib.idx == node.idx:
                continue
            P_sib = P_by_node[sib.idx]
            sib_prod *= down_liks[sib.idx] @ P_sib.T
        parent_msg = up_liks[parent.idx] * sib_prod
        P = P_by_node[node.idx]
        up_liks[node.idx] = normalize_rows(parent_msg @ P)
    return down_liks, up_liks


def _impute_alignment(
    tree,
    filt_genotypes,
    qcache,
    root_prior,
    site_rates,
    batch_size,
    min_posterior,
    n_states,
    lik_fn,
    call_symbols,
    fail_msg,
):
    """Shared core of the haploid and diploid imputation functions.

    Message passing runs on blocks of columns (``batch_size`` missing sites at
    a time, or all at once when it is None), and within a block on groups of
    sites sharing a rate. Encoding the input, recording posteriors and
    writing calls are done on arrays, never one cell at a time.
    """
    sample_names = list(filt_genotypes.keys())
    sample_to_row = {sample: i for i, sample in enumerate(sample_names)}
    num_sites = len(next(iter(filt_genotypes.values())))
    seq_array = np.array(
        [filt_genotypes[sample] for sample in sample_names], dtype=object
    )
    if seq_array.ndim != 2:
        raise ValueError("All genotype sequences must have the same length.")

    codes, lik_table, impute_flag, still_missing = _encode_tokens(seq_array, lik_fn)
    missing_mask = impute_flag[codes]
    missing_sites = np.where(missing_mask.any(axis=0))[0]

    if len(missing_sites) == 0:
        empty = PosteriorMap(
            sample_names, [], [], np.empty((0, n_states)), num_sites
        )
        return filt_genotypes, empty

    missing_rates = None
    if site_rates is not None:
        missing_rates = np.array([site_rates[i] for i in missing_sites])

    postorder_nodes = list(tree.treenode.traverse("postorder"))
    preorder_nodes = list(tree.treenode.traverse("preorder"))
    root_idx = tree.treenode.idx

    leaf_by_sample = {
        node.name: node
        for node in postorder_nodes
        if node.is_leaf() and node.name in filt_genotypes
    }
    tips = [
        (row_i, leaf_by_sample[sample])
        for sample, row_i in sample_to_row.items()
        if sample in leaf_by_sample
    ]
    tip_rows = np.array([row_i for row_i, _ in tips], dtype=np.intp)
    sink = _PosteriorSink(
        int(missing_mask[tip_rows].sum()), len(sample_names), num_sites, n_states
    )
    fill = 1.0 / n_states

    step = len(missing_sites) if batch_size is None else batch_size
    for batch_start in range(0, len(missing_sites), step):
        batch_end = min(batch_start + step, len(missing_sites))
        batch_sites = missing_sites[batch_start:batch_end]

        batch_rates = None
        if missing_rates is not None:
            batch_rates = missing_rates[batch_start:batch_end]

        if batch_rates is None or np.all(batch_rates == batch_rates[0]):
            rate = 1.0 if batch_rates is None else float(batch_rates[0])
            groups = [(rate, batch_sites)]
        else:
            groups = [
                (float(rate_val), batch_sites[batch_rates == rate_val])
                for rate_val in np.unique(batch_rates)
            ]

        for rate, cols in groups:
            if len(cols) == 0:
                continue
            P_by_node = {
                node.idx: qcache.P(max(node.dist, MIN_BRANCH_LENGTH), rate=rate)
                for node in postorder_nodes
                if not node.is_root()
            }

            def leaf_lik(node, cols=cols):
                row_i = sample_to_row.get(node.name)
                if row_i is None:
                    return np.full((len(cols), n_states), fill, dtype=float)
                return lik_table[codes[row_i, cols]]

            down_liks, up_liks = _two_pass(
                postorder_nodes, preorder_nodes, P_by_node, leaf_lik,
                len(cols), n_states, root_prior, root_idx,
            )

            # Tip reconstruction: one vectorised step per sample.
            for row_i, node in tips:
                pos = np.flatnonzero(missing_mask[row_i, cols])
                if len(pos) == 0:
                    continue
                tip_post = normalize_rows(
                    up_liks[node.idx][pos] * down_liks[node.idx][pos]
                )
                sites = cols[pos]
                sink.add(row_i, sites, tip_post)
                calls = np.argmax(tip_post, axis=1)
                if min_posterior is not None:
                    keep = ~(tip_post.max(axis=1) < min_posterior)
                    sites, calls = sites[keep], calls[keep]
                seq_array[row_i, sites] = call_symbols[calls]

    if sink.n != len(sink.rows):
        raise AssertionError("Internal error: not every missing cell was visited.")

    if min_posterior is None:
        remaining = still_missing[codes]
        remaining[sink.rows, sink.sites] = False  # these now hold a call
        if remaining.any():
            raise AssertionError(fail_msg)

    imputed = {sample: seq_array[i].tolist() for i, sample in enumerate(sample_names)}
    posteriors = PosteriorMap(sample_names, sink.rows, sink.sites, sink.post, num_sites)
    return imputed, posteriors


def phylogenetic_imputation(
    tree: tt.ToyTree,
    genotypes: Dict[str, List[str]],
    q_matrix: pd.DataFrame,
    site_rates: Optional[List[float]] = None,
    batch_size: int = 5000,
    min_posterior: Optional[float] = None,
) -> Tuple[Dict[str, List[str]], PosteriorMap]:
    """Two-pass message passing for phylogenetic imputation.

    Args:
        tree: Phylogenetic tree (toytree).
        genotypes: Dict mapping sample name -> list of nucleotide chars.
        q_matrix: 4x4 rate matrix (A,C,G,T indexed DataFrame).
        site_rates: Optional per-site evolutionary rates.
        batch_size: Max missing sites to process at once (controls memory).
        min_posterior: If set, only impute sites where the maximum posterior
            probability meets or exceeds this threshold. Sites below the
            threshold retain their original missing value.

    Returns:
        (imputed_genotypes, imputed_likelihoods) where imputed_likelihoods
        maps (sample_name, site_index) -> posterior probability array (4,).
        It is a read-only :class:`PosteriorMap` (a ``Mapping``); its
        ``posteriors``, ``sample_index`` and ``sites`` arrays give the same
        data for vectorised use.
    """
    qcache = QCache(q_matrix)
    root_prior = stationary_pi_from_q(q_matrix.to_numpy())

    common_samples = _reconcile_tree_genotypes(tree, genotypes)
    non_tree_samples = set(genotypes.keys()) - common_samples

    def _merge_non_tree(result):
        """Add back samples that weren't in the tree (unmodified)."""
        for s in non_tree_samples:
            result[s] = list(genotypes[s])
        return result

    filt_genotypes = {s: list(genotypes[s]) for s in common_samples}
    num_sites = len(next(iter(filt_genotypes.values())))

    if site_rates is not None and len(site_rates) != num_sites:
        raise ValueError(
            f"len(site_rates)={len(site_rates)} != num_sites={num_sites}"
        )

    imputed, imputed_likelihoods = _impute_alignment(
        tree,
        filt_genotypes,
        qcache,
        root_prior,
        site_rates,
        batch_size,
        min_posterior,
        n_states=4,
        lik_fn=likelihood_matrix_for_chars,
        call_symbols=_NUC_CALLS,
        fail_msg="Imputation failed. Missing values remain.",
    )
    return _merge_non_tree(imputed), imputed_likelihoods


def ctmc_pruning_loglik(
    tree: tt.ToyTree,
    genotypes: Dict[str, List[str]],
    q_matrix: pd.DataFrame,
) -> float:
    """Standard pruning log-likelihood with per-site log-scaling."""
    qcache = QCache(q_matrix)
    root_prior = stationary_pi_from_q(q_matrix.to_numpy())

    common_samples = _reconcile_tree_genotypes(tree, genotypes)

    filt_genotypes = {s: list(genotypes[s]) for s in common_samples}
    num_sites = len(next(iter(filt_genotypes.values())))

    postorder_nodes = list(tree.treenode.traverse("postorder"))
    P_by_node = {
        node.idx: qcache.P(max(node.dist, MIN_BRANCH_LENGTH), rate=1.0)
        for node in postorder_nodes
        if not node.is_root()
    }

    down_liks = {}
    log_scale = np.zeros(num_sites, dtype=float)

    for node in postorder_nodes:
        if node.is_leaf():
            if node.name not in filt_genotypes:
                lik = np.ones((num_sites, 4), dtype=float)
            else:
                lik = np.zeros((num_sites, 4), dtype=float)
                seq = filt_genotypes[node.name]
                for site_idx, obs in enumerate(seq):
                    if is_missing(obs):
                        lik[site_idx, :] = 1.0
                    else:
                        for state in get_iupac_full(obs):
                            lik[site_idx, CHAR_MAP[state]] = 1.0
        else:
            lik = np.ones((num_sites, 4), dtype=float)
            for child in node.children:
                P = P_by_node[child.idx]
                lik *= down_liks[child.idx] @ P.T

        scale = lik.sum(axis=1)
        bad = (scale <= 0.0) | ~np.isfinite(scale)
        if np.any(bad):
            lik[bad, :] = 0.25
            scale[bad] = 1.0

        lik = lik / scale[:, None]
        log_scale += np.log(scale)
        down_liks[node.idx] = lik

    root_lik = down_liks[tree.treenode.idx] @ root_prior
    root_lik = np.where(
        (root_lik <= 0.0) | ~np.isfinite(root_lik),
        np.finfo(float).tiny,
        root_lik,
    )

    return float(np.sum(np.log(root_lik) + log_scale))


# ---------------------------------------------------------------------------
# 10-state diploid genotype model (GTGTR4 / GT10)
# ---------------------------------------------------------------------------

# DNA rate pair labels in canonical order
_DNA_RATE_PAIRS = ("AC", "AG", "AT", "CG", "CT", "GT")


def build_q10_from_gtr(
    gtr_rates: np.ndarray,
    freqs_10: np.ndarray,
) -> pd.DataFrame:
    """Build a 10x10 Q matrix from 6 GTR DNA rates + 10 genotype frequencies.

    Uses the GTGTR4 rate symmetry structure: each allowed transition between
    genotype states uses the DNA exchangeability rate for the single nucleotide
    that changes. Forbidden transitions (requiring 2+ mutations) have rate 0.

    Args:
        gtr_rates: Array of 6 GTR exchangeability rates [AC, AG, AT, CG, CT, GT].
        freqs_10: Array of 10 genotype state frequencies (must sum to 1).

    Returns:
        10x10 Q matrix as DataFrame indexed by GENOTYPE_STATES, scaled so
        mean substitution rate = 1.
    """
    gtr_rates = np.asarray(gtr_rates, dtype=float)
    freqs_10 = np.asarray(freqs_10, dtype=float)

    if len(gtr_rates) != 6:
        raise ValueError(f"Expected 6 GTR rates, got {len(gtr_rates)}")
    if len(freqs_10) != 10:
        raise ValueError(f"Expected 10 genotype frequencies, got {len(freqs_10)}")

    n = 10
    Q = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            rate_idx = GT_RATE_SYMMETRY[i, j]
            if rate_idx < 0:
                continue  # forbidden transition
            s_ij = gtr_rates[rate_idx]
            Q[i, j] = s_ij * freqs_10[j]
            Q[j, i] = s_ij * freqs_10[i]

    # Set diagonal so rows sum to 0
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))

    # Scale so mean substitution rate = 1
    mean_rate = float(-np.dot(freqs_10, np.diag(Q)))
    if np.isfinite(mean_rate) and mean_rate > 0:
        Q /= mean_rate

    return pd.DataFrame(Q, index=list(GENOTYPE_STATES), columns=list(GENOTYPE_STATES))


def estimate_genotype_freqs(
    genotypes: Dict[str, List[str]],
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Estimate 10-state genotype frequencies from observed IUPAC data.

    Args:
        genotypes: Dict mapping sample name -> list of IUPAC chars.
        pseudocount: Added to each state count to avoid zeros.

    Returns:
        Array of 10 frequencies summing to 1.
    """
    from phylo_impute.utils import IUPAC_TO_GT_INDICES

    counts = np.full(10, pseudocount, dtype=float)
    for seq in genotypes.values():
        for char in seq:
            if is_missing(char):
                continue
            c = str(char).upper()
            if c in IUPAC_TO_GT_INDICES:
                indices = IUPAC_TO_GT_INDICES[c]
                for idx in indices:
                    counts[idx] += 1.0 / len(indices)

    total = counts.sum()
    if total <= 0:
        return np.ones(10) / 10.0
    return counts / total


def phylogenetic_imputation_diploid(
    tree: tt.ToyTree,
    genotypes: Dict[str, List[str]],
    q_matrix: pd.DataFrame,
    site_rates: Optional[List[float]] = None,
    min_posterior: Optional[float] = None,
) -> Tuple[Dict[str, List[str]], PosteriorMap]:
    """Two-pass message passing for diploid genotype imputation.

    Same algorithm as phylogenetic_imputation but with 10 states
    (AA, CC, GG, TT, AC, AG, AT, CG, CT, GT) instead of 4.

    Args:
        tree: Phylogenetic tree (toytree).
        genotypes: Dict mapping sample name -> list of IUPAC genotype chars.
        q_matrix: 10x10 rate matrix (GENOTYPE_STATES indexed DataFrame).
        site_rates: Optional per-site evolutionary rates.
        min_posterior: If set, only impute sites where the maximum posterior
            probability meets or exceeds this threshold. Sites below the
            threshold retain their original missing value.

    Returns:
        (imputed_genotypes, imputed_likelihoods) where imputed_likelihoods
        maps (sample_name, site_index) -> posterior probability array (10,),
        as a read-only :class:`PosteriorMap`.
    """
    qcache = QCache(q_matrix)
    root_prior = stationary_pi_from_q(q_matrix.to_numpy())

    common_samples = _reconcile_tree_genotypes(tree, genotypes)
    non_tree_samples_dip = set(genotypes.keys()) - common_samples

    def _merge_non_tree_dip(result):
        for s in non_tree_samples_dip:
            result[s] = list(genotypes[s])
        return result

    filt_genotypes = {s: list(genotypes[s]) for s in common_samples}
    num_sites = len(next(iter(filt_genotypes.values())))

    if site_rates is not None and len(site_rates) != num_sites:
        raise ValueError(
            f"len(site_rates)={len(site_rates)} != num_sites={num_sites}"
        )

    imputed, imputed_likelihoods = _impute_alignment(
        tree,
        filt_genotypes,
        qcache,
        root_prior,
        site_rates,
        None,  # all missing sites in one block
        min_posterior,
        n_states=10,
        lik_fn=likelihood_matrix_for_genotypes,
        call_symbols=_GT_CALLS,
        fail_msg="Diploid imputation failed. Missing values remain.",
    )
    return _merge_non_tree_dip(imputed), imputed_likelihoods


def ctmc_pruning_loglik_diploid(
    tree: tt.ToyTree,
    genotypes: Dict[str, List[str]],
    q_matrix: pd.DataFrame,
) -> float:
    """Standard pruning log-likelihood for the 10-state diploid model."""
    qcache = QCache(q_matrix)
    root_prior = stationary_pi_from_q(q_matrix.to_numpy())

    common_samples = _reconcile_tree_genotypes(tree, genotypes)

    filt_genotypes = {s: list(genotypes[s]) for s in common_samples}
    num_sites = len(next(iter(filt_genotypes.values())))

    n_states = 10
    postorder_nodes = list(tree.treenode.traverse("postorder"))
    P_by_node = {
        node.idx: qcache.P(max(node.dist, MIN_BRANCH_LENGTH), rate=1.0)
        for node in postorder_nodes
        if not node.is_root()
    }

    down_liks = {}
    log_scale = np.zeros(num_sites, dtype=float)

    for node in postorder_nodes:
        if node.is_leaf():
            if node.name not in filt_genotypes:
                lik = np.ones((num_sites, n_states), dtype=float)
            else:
                seq = filt_genotypes[node.name]
                lik = likelihood_matrix_for_genotypes(seq)
        else:
            lik = np.ones((num_sites, n_states), dtype=float)
            for child in node.children:
                P = P_by_node[child.idx]
                lik *= down_liks[child.idx] @ P.T

        scale = lik.sum(axis=1)
        bad = (scale <= 0.0) | ~np.isfinite(scale)
        if np.any(bad):
            lik[bad, :] = 1.0 / n_states
            scale[bad] = 1.0

        lik = lik / scale[:, None]
        log_scale += np.log(scale)
        down_liks[node.idx] = lik

    root_lik = down_liks[tree.treenode.idx] @ root_prior
    root_lik = np.where(
        (root_lik <= 0.0) | ~np.isfinite(root_lik),
        np.finfo(float).tiny,
        root_lik,
    )

    return float(np.sum(np.log(root_lik) + log_scale))


# ---------------------------------------------------------------------------
# Baseline imputation methods
# ---------------------------------------------------------------------------


def majority_rule_imputation(
    genotypes: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Impute missing values with the most common allele at each site.

    Args:
        genotypes: Dict mapping sample name -> list of chars.

    Returns:
        New genotype dict with missing values filled.
    """
    import copy

    result = copy.deepcopy(genotypes)
    sample_names = list(result.keys())
    n_sites = len(next(iter(result.values())))

    for site_i in range(n_sites):
        # Count alleles at this site
        counts: Dict[str, int] = {}
        for sample in sample_names:
            val = result[sample][site_i]
            if not is_missing(val):
                counts[val] = counts.get(val, 0) + 1
        if not counts:
            continue
        majority = max(counts, key=counts.get)
        # Fill missing
        for sample in sample_names:
            if is_missing(result[sample][site_i]):
                result[sample][site_i] = majority

    return result


def reference_imputation(
    genotypes: Dict[str, List[str]],
    reference_seq: List[str],
) -> Dict[str, List[str]]:
    """Impute missing values with the reference allele at each site.

    Args:
        genotypes: Dict mapping sample name -> list of chars.
        reference_seq: List of chars (same length as genotype sequences).

    Returns:
        New genotype dict with missing values filled from the reference.
    """
    import copy

    result = copy.deepcopy(genotypes)
    n_sites = len(next(iter(result.values())))

    if len(reference_seq) != n_sites:
        raise ValueError(
            f"Reference length ({len(reference_seq)}) != alignment length ({n_sites})"
        )

    for sample in result:
        for site_i in range(n_sites):
            if is_missing(result[sample][site_i]):
                ref_val = reference_seq[site_i]
                result[sample][site_i] = ref_val if not is_missing(ref_val) else "N"

    return result


from phylo_impute.utils import get_iupac_full  # noqa: E402, F401
