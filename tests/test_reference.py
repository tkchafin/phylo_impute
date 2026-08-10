"""Full pipeline test mirroring solve.py from phylogenetic-imputation-genomics.

Requires:
- IQ-TREE (iqtree3) installed and on PATH
- Reference FASTA data (packaged in tests/data/genomes.fasta, or set REFERENCE_FASTA env var)
"""
from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pytest
import toytree as tt

from phylo_impute.core import (
    ctmc_pruning_loglik,
    phylogenetic_imputation,
    repair_and_scale_q,
)
from phylo_impute.io import q_from_iqtree, read_fasta, records_to_genotype_dict, write_fasta
from phylo_impute.utils import CHAR_MAP, is_missing

SEED = 123
MASK_PROP = 0.10
PROBE_SAMPLE = "EPI_ISL_535892"
PROBE_SITE = 240  # 0-based alignment column

REFERENCE_FASTA_PATH = Path(
    os.environ.get(
        "REFERENCE_FASTA",
        str(Path(__file__).parent / "data" / "genomes.fasta"),
    )
)

# Expected values from test_outputs.py in the reference repo
EXPECTED = {
    "diff_accuracy_variable_all": 0.0083,
    "diff_accuracy_variable_10": 0.0426,
    "diff_accuracy_variable_25": 0.0174,
    "diff_accuracy_variable_50": 0.0125,
    "diff_accuracy_variable_75": 0.0097,
    "slope": 0.724,
    "loglik_all": -46279.3707,
    "loglik_10": -41334.1274,
    "loglik_25": -42164.6076,
    "loglik_50": -43400.7927,
    "loglik_75": -44287.8701,
    "probe_map_vector_all": [0.0, 0.9998, 0.0, 0.0001],
    "probe_map_vector_10": [0.0, 0.0003, 0.0001, 0.9995],
    "probe_map_vector_25": [0.0, 0.9996, 0.0, 0.0003],
    "probe_map_vector_50": [0.0, 0.9997, 0.0, 0.0003],
    "probe_map_vector_75": [0.0, 0.9997, 0.0, 0.0003],
}

DIFF_ACC_TOL = 0.005
LOGLIK_ABS_TOL = 1000.0  # IQ-TREE -fast is stochastic; tree topology varies across runs
POSTERIOR_ABS_TOL = 0.02
SLOPE_MIN = 0.5
SLOPE_MAX = 1.5


# --- Helper functions matching solve.py ---


def sample_index(records, sample_name):
    for i, (name, _) in enumerate(records):
        if name == sample_name:
            return i
    raise ValueError(f"Sample not found: {sample_name}")


def mask_random_sites(records, prop=0.10, force_mask=None):
    random.seed(SEED)
    names = [name for name, _ in records]
    seqs = [list(seq) for _, seq in records]
    name_to_index = {name: i for i, name in enumerate(names)}

    eligible = [
        (i, j)
        for i, seq in enumerate(seqs)
        for j, base in enumerate(seq)
        if base != "-"
    ]

    n_mask = int(len(eligible) * prop)
    masked_positions = set(random.sample(eligible, n_mask))

    if force_mask:
        for sample_name, site_i in force_mask:
            sample_i = name_to_index[sample_name]
            masked_positions.add((sample_i, site_i))

    masked_positions = sorted(masked_positions)
    for i, j in masked_positions:
        seqs[i][j] = "N"

    masked_records = [(name, "".join(seq)) for name, seq in zip(names, seqs)]
    return masked_records, masked_positions


def variable_columns(records):
    seqs = [seq for _, seq in records]
    n_sites = len(seqs[0])
    variable_sites = set()
    for site_i in range(n_sites):
        states = {seq[site_i] for seq in seqs if seq[site_i] in CHAR_MAP}
        if len(states) > 1:
            variable_sites.add(site_i)
    return variable_sites


def imputation_accuracy(original_records, imputed_genotypes, masked_positions, allowed_sites=None):
    n_correct = 0
    n_total = 0
    for sample_i, site_i in masked_positions:
        if allowed_sites is not None and site_i not in allowed_sites:
            continue
        sample = original_records[sample_i][0]
        truth = original_records[sample_i][1][site_i]
        pred = imputed_genotypes[sample][site_i]
        if truth == "-":
            continue
        n_total += 1
        if pred == truth:
            n_correct += 1
    return {"n_correct": n_correct, "n_total": n_total, "accuracy": n_correct / n_total if n_total else None}


def most_common_allele_baseline_accuracy(original_records, masked_records, masked_positions, allowed_sites=None):
    masked_by_name = {name: seq for name, seq in masked_records}
    n_correct = 0
    n_total = 0
    for sample_i, site_i in masked_positions:
        if allowed_sites is not None and site_i not in allowed_sites:
            continue
        sample = original_records[sample_i][0]
        truth = original_records[sample_i][1][site_i]
        if truth == "-":
            continue
        pred = most_common_allele_for_site(masked_by_name, site_i, exclude_sample=sample)
        if pred is None:
            continue
        n_total += 1
        if pred == truth:
            n_correct += 1
    return {"n_correct": n_correct, "n_total": n_total, "accuracy": n_correct / n_total if n_total else None}


def most_common_allele_for_site(masked_by_name, site_i, exclude_sample=None):
    counts = {base: 0 for base in CHAR_MAP}
    for sample, seq in masked_by_name.items():
        if sample == exclude_sample:
            continue
        base = seq[site_i]
        if base in counts:
            counts[base] += 1
    max_count = max(counts.values())
    if max_count == 0:
        return None
    for base in ("A", "C", "G", "T"):
        if counts[base] == max_count:
            return base
    return None


def run_iqtree(fasta_path, out_prefix):
    cmd = [
        "iqtree3",
        "-s", str(fasta_path),
        "-st", "DNA",
        "-m", "GTR+I+R3",
        "-safe",
        "-fast",
        "-seed", str(SEED),
        "-T", "1",
        "-redo",
        "-pre", str(out_prefix),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    iqtree_file = Path(str(out_prefix) + ".iqtree")
    tree_file = Path(str(out_prefix) + ".treefile")
    if not iqtree_file.exists():
        raise FileNotFoundError(f"Expected IQ-TREE report not found: {iqtree_file}")
    if not tree_file.exists():
        raise FileNotFoundError(f"Expected IQ-TREE treefile not found: {tree_file}")
    return iqtree_file, tree_file


def round_float(x):
    if x is None:
        return None
    return round(float(x), 4)


def round_vector(v):
    if v is None:
        return None
    return [round_float(x) for x in v]


# --- Check prerequisites ---

def iqtree_available():
    try:
        subprocess.run(["iqtree3", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


requires_iqtree = pytest.mark.skipif(
    not iqtree_available(), reason="iqtree3 not found on PATH"
)
requires_reference_data = pytest.mark.skipif(
    not REFERENCE_FASTA_PATH.exists(),
    reason=f"Reference FASTA not found at {REFERENCE_FASTA_PATH}",
)


# --- Test fixtures ---

@pytest.fixture(scope="module")
def reference_results():
    """Run the full pipeline once and return results for all subsets."""
    if not REFERENCE_FASTA_PATH.exists():
        pytest.skip(f"Reference FASTA not found: {REFERENCE_FASTA_PATH}")
    if not iqtree_available():
        pytest.skip("iqtree3 not available")

    tmp_dir = Path("/tmp/_phylo_impute_test")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        records = read_fasta(str(REFERENCE_FASTA_PATH))
        masked_records, masked_sites = mask_random_sites(
            records, prop=MASK_PROP, force_mask=[(PROBE_SAMPLE, PROBE_SITE)]
        )

        subset_specs = [
            ("10", 10),
            ("25", 25),
            ("50", 50),
            ("75", 75),
            ("all", len(records)),
        ]

        run_results = {}

        for label, nseq in subset_specs:
            subset_records = records[:nseq]
            subset_masked_records = masked_records[:nseq]
            subset_masked_sites = [
                (si, sj) for si, sj in masked_sites if si < nseq
            ]

            masked_fasta = tmp_dir / f"genomes.masked.{label}.fasta"
            out_prefix = tmp_dir / f"genomes.masked.{label}"

            write_fasta(subset_masked_records, str(masked_fasta))
            iqtree_file, tree_file = run_iqtree(masked_fasta, out_prefix)

            tree = tt.tree(str(tree_file))
            q_matrix = repair_and_scale_q(q_from_iqtree(str(iqtree_file)))

            masked_genotypes = records_to_genotype_dict(subset_masked_records)

            t0 = time.perf_counter()
            imputed_genotypes, imputed_likelihoods = phylogenetic_imputation(
                tree=tree, genotypes=masked_genotypes, q_matrix=q_matrix
            )
            runtime = time.perf_counter() - t0

            # Probe vector
            probe_idx = sample_index(subset_records, PROBE_SAMPLE)
            probe_key = (PROBE_SAMPLE, PROBE_SITE)
            probe_vector = imputed_likelihoods[probe_key].tolist() if probe_key in imputed_likelihoods else None

            # Accuracy
            var_sites = variable_columns(subset_records)
            var_acc = imputation_accuracy(
                subset_records, imputed_genotypes, subset_masked_sites, allowed_sites=var_sites
            )
            baseline_var_acc = most_common_allele_baseline_accuracy(
                subset_records, subset_masked_records, subset_masked_sites, allowed_sites=var_sites
            )

            # Log-likelihood
            loglik = ctmc_pruning_loglik(tree=tree, genotypes=masked_genotypes, q_matrix=q_matrix)

            run_results[label] = {
                "accuracy_variable": var_acc["accuracy"],
                "baseline_accuracy_variable": baseline_var_acc["accuracy"],
                "runtime": runtime,
                "loglik": loglik,
                "probe_map_vector": probe_vector,
            }

        # Compute final results
        results = {}
        for label in ("10", "25", "50", "75", "all"):
            diff = float(run_results[label]["accuracy_variable"]) - float(
                run_results[label]["baseline_accuracy_variable"]
            )
            results[f"diff_accuracy_variable_{label}"] = round_float(diff)
            results[f"loglik_{label}"] = round_float(run_results[label]["loglik"])
            results[f"probe_map_vector_{label}"] = round_vector(
                run_results[label]["probe_map_vector"]
            )

        # Scaling slope
        runtimes = np.array([
            run_results["10"]["runtime"],
            run_results["25"]["runtime"],
            run_results["50"]["runtime"],
            run_results["75"]["runtime"],
            run_results["all"]["runtime"],
        ])
        logN = np.log(np.array([10, 25, 50, 75, len(records)], dtype=float))
        logT = np.log(runtimes)
        results["slope"] = round_float(float(np.polyfit(logN, logT, 1)[0]))

        yield results

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Tests ---

@requires_iqtree
@requires_reference_data
class TestReferenceAccuracy:
    @pytest.mark.parametrize("key", [
        "diff_accuracy_variable_all",
        "diff_accuracy_variable_10",
        "diff_accuracy_variable_25",
        "diff_accuracy_variable_50",
        "diff_accuracy_variable_75",
    ])
    def test_diff_accuracy(self, reference_results, key):
        assert isinstance(reference_results[key], (int, float))
        assert float(reference_results[key]) >= EXPECTED[key] - DIFF_ACC_TOL


@requires_iqtree
@requires_reference_data
class TestReferenceLoglik:
    @pytest.mark.parametrize("key", [
        "loglik_all",
        "loglik_10",
        "loglik_25",
        "loglik_50",
        "loglik_75",
    ])
    def test_loglik(self, reference_results, key):
        assert isinstance(reference_results[key], (int, float))
        assert math.isclose(
            float(reference_results[key]),
            EXPECTED[key],
            rel_tol=1e-4,
            abs_tol=LOGLIK_ABS_TOL,
        )


@requires_iqtree
@requires_reference_data
class TestReferenceProbeVectors:
    @pytest.mark.parametrize("key", [
        "probe_map_vector_all",
        "probe_map_vector_10",
        "probe_map_vector_25",
        "probe_map_vector_50",
        "probe_map_vector_75",
    ])
    def test_probe_vector(self, reference_results, key):
        observed = reference_results[key]
        expected = EXPECTED[key]
        assert isinstance(observed, list)
        assert len(observed) == 4
        assert math.isclose(sum(float(x) for x in observed), 1.0, abs_tol=0.02)
        for obs, exp in zip(observed, expected):
            assert math.isclose(float(obs), float(exp), rel_tol=0.05, abs_tol=POSTERIOR_ABS_TOL)


@requires_iqtree
@requires_reference_data
class TestReferenceScaling:
    def test_slope(self, reference_results):
        assert isinstance(reference_results["slope"], (int, float))
        assert SLOPE_MIN <= float(reference_results["slope"]) <= SLOPE_MAX
