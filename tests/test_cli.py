"""CLI smoke tests."""
from __future__ import annotations

import subprocess
import sys
from io import StringIO

import pytest

from phylo_impute.cli import _mask_known_sites, _print_accuracy_table


def test_help():
    result = subprocess.run(
        [sys.executable, "-m", "phylo_impute.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "phylo-impute" in result.stdout


def test_version():
    result = subprocess.run(
        [sys.executable, "-m", "phylo_impute.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "0.1.0" in result.stdout


def test_missing_required_args():
    result = subprocess.run(
        [sys.executable, "-m", "phylo_impute.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_reference_requires_evaluate():
    result = subprocess.run(
        [
            sys.executable, "-m", "phylo_impute.cli",
            "--input", "dummy.phy", "--treefile", "dummy.nwk",
            "--qmatrix", "dummy.iqtree", "--reference", "sample1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "require --evaluate" in result.stderr


def test_reference_fasta_requires_evaluate():
    result = subprocess.run(
        [
            sys.executable, "-m", "phylo_impute.cli",
            "--input", "dummy.phy", "--treefile", "dummy.nwk",
            "--qmatrix", "dummy.iqtree", "--reference-fasta", "ref.fasta",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "require --evaluate" in result.stderr


def test_reference_and_fasta_mutually_exclusive():
    result = subprocess.run(
        [
            sys.executable, "-m", "phylo_impute.cli",
            "--input", "dummy.phy", "--treefile", "dummy.nwk",
            "--qmatrix", "dummy.iqtree", "--evaluate", "0.1",
            "--reference", "sample1", "--reference-fasta", "ref.fasta",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_mask_excludes_reference_sample():
    genotypes = {
        "ref": ["A", "C", "G", "T"],
        "s1": ["A", "C", "G", "T"],
        "s2": ["A", "C", "G", "T"],
    }
    masked, positions, original = _mask_known_sites(
        genotypes, frac=1.0, seed=42, exclude_samples={"ref"},
    )
    # ref should be untouched
    assert masked["ref"] == ["A", "C", "G", "T"]
    # all masked positions should be from s1 or s2
    for sample, site_i in positions:
        assert sample != "ref"


def test_min_posterior_validation():
    result = subprocess.run(
        [
            sys.executable, "-m", "phylo_impute.cli",
            "--input", "dummy.phy", "--treefile", "dummy.nwk",
            "--qmatrix", "dummy.iqtree", "--min-posterior", "1.5",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "min-posterior" in result.stderr


def test_accuracy_table_with_reference(capsys):
    original = {"s1": ["A", "C", "G"], "s2": ["T", "C", "A"]}
    imputed = {"s1": ["A", "C", "G"], "s2": ["T", "C", "A"]}
    masked = {"s1": ["N", "N", "N"], "s2": ["N", "N", "N"]}
    positions = [("s1", 0), ("s1", 1), ("s1", 2), ("s2", 0), ("s2", 1), ("s2", 2)]
    variable = {0, 2}
    ref_seq = ["A", "C", "A"]  # matches s1 exactly, partially matches s2

    _print_accuracy_table(original, imputed, masked, positions, variable, reference_seq=ref_seq)
    out = capsys.readouterr().out
    assert "Reference fill" in out
    assert "Phylo - Reference" in out
