#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from phylo_impute import __version__
from phylo_impute.utils import is_missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phylo-impute",
        description=(
            "Phylogenetic imputation of missing genotype data "
            "using CTMC models (4-state haploid or 10-state diploid)."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    # Required
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input genotype file (FASTA, VCF, PHYLIP, STRUCTURE, GENEPOP)",
    )
    parser.add_argument(
        "--treefile", required=True, help="Path to Newick tree file"
    )
    parser.add_argument(
        "--qmatrix", required=True, help="Path to Q-matrix file (IQ-TREE .iqtree or text)"
    )

    # Optional tree-related
    parser.add_argument(
        "--siterates", default=None,
        help=(
            "Path to per-site rates file. Must have one rate per site in the "
            "same order as the input alignment. For SNP-based formats (VCF, "
            "STRUCTURE, GENEPOP), rates correspond to SNP columns, not genome "
            "positions."
        )
    )

    # I/O options
    parser.add_argument(
        "--format",
        default=None,
        choices=["fasta", "vcf", "phylip", "structure", "genepop"],
        help="Input format (default: infer from extension)",
    )
    parser.add_argument(
        "--popmap", default=None, help="Population map file"
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output prefix (default: input file stem)",
    )
    parser.add_argument(
        "--output-format",
        default=None,
        choices=["vcf", "phylip", "structure", "genepop", "fasta"],
        help="Output format (default: same as input)",
    )

    # Diploid model
    parser.add_argument(
        "--diploid",
        action="store_true",
        help="Use 10-state diploid genotype model (GTGTR4) instead of 4-state haploid",
    )

    # Evaluation
    parser.add_argument(
        "--evaluate",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "Mask FRAC (0-1) of known genotypes and report imputation accuracy. "
            "Prints a table comparing phylogenetic vs. majority-rule accuracy "
            "on all masked sites and variable sites only."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --evaluate masking (default: 42)",
    )
    parser.add_argument(
        "--reference",
        default=None,
        metavar="SAMPLE",
        help=(
            "Use this sample from the dataset as a reference-fill baseline "
            "in --evaluate mode. The sample is excluded from masking and its "
            "alleles are used for naive reference-based imputation comparison."
        ),
    )
    parser.add_argument(
        "--reference-fasta",
        default=None,
        metavar="PATH",
        help=(
            "Path to an external FASTA file with a reference sequence for "
            "reference-fill baseline in --evaluate mode. The sequence must "
            "have the same number of sites as the input, matched by index "
            "(position i in the reference = site i in the alignment). For "
            "SNP-based formats, this means the reference must contain only "
            "the SNP columns in the same order, not the full genome."
        ),
    )

    # Posterior filter
    parser.add_argument(
        "--min-posterior",
        type=float,
        default=None,
        metavar="PROB",
        help=(
            "Minimum posterior probability threshold for imputation. Sites "
            "where the maximum posterior is below this value retain their "
            "missing value instead of being imputed. (default: off)"
        ),
    )

    # General
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    return parser


def _mask_known_sites(
    genotypes: Dict[str, List[str]],
    frac: float,
    seed: int,
    exclude_samples: Set[str] | None = None,
) -> Tuple[Dict[str, List[str]], List[Tuple[str, int]], Dict[str, List[str]]]:
    """Mask a fraction of known genotypes for evaluation.

    Args:
        exclude_samples: Sample names to exclude from masking (e.g. reference).

    Returns:
        (masked_genotypes, masked_positions, original_genotypes)
        where masked_positions only includes artificially masked entries.
    """
    original = copy.deepcopy(genotypes)
    masked = copy.deepcopy(genotypes)
    exclude = exclude_samples or set()

    known = [
        (sample, site_i)
        for sample, seq in masked.items()
        for site_i, val in enumerate(seq)
        if not is_missing(val) and sample not in exclude
    ]

    n_mask = int(len(known) * frac)
    rng = random.Random(seed)
    to_mask = rng.sample(known, min(n_mask, len(known)))

    for sample, site_i in to_mask:
        masked[sample][site_i] = "N"

    return masked, to_mask, original


def _most_common_at_site(
    genotypes: Dict[str, List[str]],
    site_i: int,
    exclude: str,
) -> str | None:
    """Most common non-missing genotype at a site, excluding one sample."""
    counts: Dict[str, int] = {}
    for sample, seq in genotypes.items():
        if sample == exclude:
            continue
        val = seq[site_i]
        if not is_missing(val):
            counts[val] = counts.get(val, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _find_variable_sites(
    genotypes: Dict[str, List[str]],
) -> Set[int]:
    """Find sites with more than one observed state."""
    n_sites = len(next(iter(genotypes.values())))
    variable = set()
    for site_i in range(n_sites):
        states = set()
        for seq in genotypes.values():
            val = seq[site_i]
            if not is_missing(val):
                states.add(val)
        if len(states) > 1:
            variable.add(site_i)
    return variable


def _print_accuracy_table(
    original: Dict[str, List[str]],
    imputed: Dict[str, List[str]],
    masked_genotypes: Dict[str, List[str]],
    masked_positions: List[Tuple[str, int]],
    variable_sites: Set[int],
    reference_seq: List[str] | None = None,
) -> None:
    """Print accuracy table: phylogenetic vs majority rule vs reference fill."""
    phylo_all = [0, 0]
    major_all = [0, 0]
    phylo_var = [0, 0]
    major_var = [0, 0]
    ref_all = [0, 0]
    ref_var = [0, 0]

    for sample, site_i in masked_positions:
        truth = original[sample][site_i]
        pred_phylo = imputed[sample][site_i]
        pred_major = _most_common_at_site(masked_genotypes, site_i, exclude=sample)

        phylo_all[1] += 1
        major_all[1] += 1
        if pred_phylo == truth:
            phylo_all[0] += 1
        if pred_major is not None and pred_major == truth:
            major_all[0] += 1

        if reference_seq is not None:
            ref_all[1] += 1
            if not is_missing(reference_seq[site_i]) and reference_seq[site_i] == truth:
                ref_all[0] += 1

        if site_i in variable_sites:
            phylo_var[1] += 1
            major_var[1] += 1
            if pred_phylo == truth:
                phylo_var[0] += 1
            if pred_major is not None and pred_major == truth:
                major_var[0] += 1
            if reference_seq is not None:
                ref_var[1] += 1
                if not is_missing(reference_seq[site_i]) and reference_seq[site_i] == truth:
                    ref_var[0] += 1

    def _pct(c, t):
        return f"{c/t*100:5.1f}% ({c}/{t})" if t > 0 else "  N/A"

    print()
    print("Imputation accuracy (artificially masked sites):")
    print(f"  Variable sites: {len(variable_sites)}")
    print()
    print(f"  {'Method':<20} {'All sites':>18} {'Variable only':>18}")
    print(f"  {'-'*20} {'-'*18} {'-'*18}")
    print(f"  {'Phylogenetic':<20} {_pct(*phylo_all):>18} {_pct(*phylo_var):>18}")
    print(f"  {'Majority rule':<20} {_pct(*major_all):>18} {_pct(*major_var):>18}")
    if reference_seq is not None:
        print(f"  {'Reference fill':<20} {_pct(*ref_all):>18} {_pct(*ref_var):>18}")

    if phylo_all[1] > 0 and major_all[1] > 0:
        base_all = major_all[0] / major_all[1]
        base_var = major_var[0] / major_var[1] if major_var[1] > 0 else 0
        diff_all = phylo_all[0] / phylo_all[1] - base_all
        diff_var = (
            (phylo_var[0] / phylo_var[1] - base_var)
            if phylo_var[1] > 0
            else 0
        )
        print(f"  {'Phylo - Majority':<20} {diff_all:>+17.1%} {diff_var:>+17.1%}")
        if reference_seq is not None and ref_all[1] > 0:
            ref_base_all = ref_all[0] / ref_all[1]
            ref_base_var = ref_var[0] / ref_var[1] if ref_var[1] > 0 else 0
            diff_ref_all = phylo_all[0] / phylo_all[1] - ref_base_all
            diff_ref_var = (
                (phylo_var[0] / phylo_var[1] - ref_base_var)
                if phylo_var[1] > 0
                else 0
            )
            print(f"  {'Phylo - Reference':<20} {diff_ref_all:>+17.1%} {diff_ref_var:>+17.1%}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.evaluate is not None and not (0 < args.evaluate < 1):
        parser.error("--evaluate must be between 0 and 1 (exclusive)")

    if args.min_posterior is not None and not (0 < args.min_posterior <= 1):
        parser.error("--min-posterior must be between 0 (exclusive) and 1 (inclusive)")

    if (args.reference or args.reference_fasta) and args.evaluate is None:
        parser.error("--reference and --reference-fasta require --evaluate")

    if args.reference and args.reference_fasta:
        parser.error("--reference and --reference-fasta are mutually exclusive")

    prefix = args.prefix or Path(args.input).stem
    input_fmt = args.format
    output_fmt = args.output_format

    # Infer formats
    from phylo_impute.io import infer_format

    if input_fmt is None:
        input_fmt = infer_format(args.input)
    if output_fmt is None:
        output_fmt = input_fmt

    if args.verbose:
        print(f"Input format: {input_fmt}")
        print(f"Output format: {output_fmt}")

    # Load data
    from phylo_impute.io import load_genotype_data, load_tree_data

    if args.verbose:
        print(f"Loading genotype data from {args.input}...")

    for path_arg, label in [(args.input, "input"), (args.treefile, "treefile"),
                             (args.qmatrix, "qmatrix")]:
        if not Path(path_arg).exists():
            parser.error(f"--{label} file not found: {path_arg}")
    if args.siterates and not Path(args.siterates).exists():
        parser.error(f"--siterates file not found: {args.siterates}")

    gd = load_genotype_data(
        input_path=args.input,
        fmt=input_fmt,
        popmap=args.popmap,
        prefix=prefix,
        verbose=args.verbose,
    )

    if args.verbose:
        print(f"Loading tree and model parameters...")

    tp = load_tree_data(
        genotype_data=gd,
        treefile=args.treefile,
        qmatrix=args.qmatrix,
        siterates=args.siterates,
        verbose=args.verbose,
    )

    # Extract data
    genotypes = gd.snpsdict
    tree = tp.tree
    try:
        q_matrix = tp.qmat
    except (ValueError, Exception):
        # SNPio may fail on plain-text Q matrices; fall back to built-in parser
        from phylo_impute.io import _read_qmatrix
        if args.verbose:
            print("  SNPio Q parser failed, using built-in parser")
        q_matrix = _read_qmatrix(args.qmatrix)
    try:
        site_rates = tp.site_rates
    except (TypeError, AttributeError):
        site_rates = None

    # Validate dimensions: genotypes, site_rates, and reference must all
    # be indexed by the same site positions (same length, same order).
    n_sites = len(next(iter(genotypes.values())))
    seq_lengths = {s: len(seq) for s, seq in genotypes.items() if len(seq) != n_sites}
    if seq_lengths:
        parser.error(
            f"Inconsistent sequence lengths across samples. Expected {n_sites} "
            f"sites but found: {seq_lengths}"
        )
    if site_rates is not None and len(site_rates) != n_sites:
        parser.error(
            f"Site rates length ({len(site_rates)}) does not match the number "
            f"of sites in the genotype data ({n_sites}). When using SNP-based "
            f"formats (VCF, STRUCTURE, GENEPOP), the site rates file must have "
            f"one rate per SNP in the same order as the input."
        )

    # Evaluation mode: mask known sites before imputation
    masked_positions = None
    original_genotypes = None
    masked_genotypes_for_eval = None
    reference_seq = None

    if args.evaluate is not None:
        # Resolve reference sequence for baseline comparison
        exclude_samples: Set[str] = set()

        if args.reference:
            if args.reference not in genotypes:
                parser.error(
                    f"--reference sample '{args.reference}' not found in dataset. "
                    f"Available: {', '.join(sorted(genotypes.keys())[:10])}..."
                )
            reference_seq = list(genotypes[args.reference])
            exclude_samples.add(args.reference)
            if args.verbose:
                print(f"Reference sample: {args.reference} (excluded from masking)")

        elif args.reference_fasta:
            from phylo_impute.io import read_fasta as _read_fasta

            ref_records = _read_fasta(args.reference_fasta)
            ref_name, ref_raw = ref_records[0]
            if len(ref_raw) != n_sites:
                parser.error(
                    f"Reference sequence length ({len(ref_raw)}) does not match "
                    f"alignment length ({n_sites}). The reference must have the "
                    f"same number of sites as the input data."
                )
            reference_seq = list(ref_raw)
            if args.verbose:
                print(f"External reference: {ref_name} from {args.reference_fasta}")

        genotypes, masked_positions, original_genotypes = _mask_known_sites(
            genotypes, args.evaluate, args.seed, exclude_samples=exclude_samples,
        )
        masked_genotypes_for_eval = copy.deepcopy(genotypes)
        print(
            f"Evaluation mode: masked {len(masked_positions)} known sites "
            f"({args.evaluate:.0%} of known, seed={args.seed})"
        )

    if args.verbose:
        n_missing = sum(
            1 for seq in genotypes.values() for x in seq if is_missing(x)
        )
        print(f"Missing values to impute: {n_missing}")

    # In evaluate mode, impute everything (no min_posterior) so accuracy
    # can be measured on all masked sites. The posterior filter is applied
    # afterwards to only the truly-missing sites.
    impute_min_posterior = None if masked_positions is not None else args.min_posterior

    from phylo_impute.core import phylogenetic_imputation, repair_and_scale_q

    if args.diploid:
        from phylo_impute.core import (
            build_q10_from_gtr,
            estimate_genotype_freqs,
            phylogenetic_imputation_diploid,
        )
        from phylo_impute.io import extract_gtr_rates_from_q4

        if args.verbose:
            print("Repairing and scaling Q matrix...")
        q_matrix = repair_and_scale_q(q_matrix)

        if args.verbose:
            print("Extracting GTR rates from 4-state Q matrix...")
        gtr_rates = extract_gtr_rates_from_q4(q_matrix)

        if args.verbose:
            print("Estimating genotype frequencies from data...")
        freqs_10 = estimate_genotype_freqs(genotypes)

        if args.verbose:
            print("Building 10-state diploid Q matrix (GTGTR4)...")
        q10 = build_q10_from_gtr(gtr_rates, freqs_10)

        if args.verbose:
            print("Running diploid phylogenetic imputation...")

        t0 = time.perf_counter()
        imputed_genotypes, imputed_likelihoods = phylogenetic_imputation_diploid(
            tree=tree,
            genotypes=genotypes,
            q_matrix=q10,
            site_rates=site_rates,
            min_posterior=impute_min_posterior,
        )
        elapsed = time.perf_counter() - t0
    else:
        if args.verbose:
            print("Repairing and scaling Q matrix...")
        q_matrix = repair_and_scale_q(q_matrix)

        if args.verbose:
            print("Running phylogenetic imputation...")

        t0 = time.perf_counter()
        imputed_genotypes, imputed_likelihoods = phylogenetic_imputation(
            tree=tree,
            genotypes=genotypes,
            q_matrix=q_matrix,
            site_rates=site_rates,
            min_posterior=impute_min_posterior,
        )
        elapsed = time.perf_counter() - t0

    n_imputed = sum(
        1 for key, post in imputed_likelihoods.items()
        if not is_missing(imputed_genotypes[key[0]][key[1]])
    )
    n_filtered = len(imputed_likelihoods) - n_imputed
    print(f"Imputation complete in {elapsed:.2f}s")
    print(f"  Imputed {n_imputed} missing values")
    if n_filtered > 0:
        print(f"  Retained {n_filtered} sites as missing (below --min-posterior {args.min_posterior})")

    # Evaluation report
    if masked_positions is not None:
        variable_sites = _find_variable_sites(original_genotypes)
        _print_accuracy_table(
            original_genotypes,
            imputed_genotypes,
            masked_genotypes_for_eval,
            masked_positions,
            variable_sites,
            reference_seq=reference_seq,
        )

        # Restore artificially masked sites to their original values
        # and apply min_posterior filter to truly-missing sites
        masked_set = set(masked_positions)
        for sample, site_i in masked_positions:
            imputed_genotypes[sample][site_i] = original_genotypes[sample][site_i]

        if args.min_posterior is not None:
            for (sample, site_i), posterior in imputed_likelihoods.items():
                if (sample, site_i) in masked_set:
                    continue
                if np.max(posterior) < args.min_posterior:
                    imputed_genotypes[sample][site_i] = "N"

    from phylo_impute.io import write_imputed

    output_path = f"{prefix}_imputed.{output_fmt}"
    if args.verbose:
        print(f"Writing output to {output_path}...")

    sample_names = list(imputed_genotypes.keys())
    write_imputed(
        genotypes=imputed_genotypes,
        sample_names=sample_names,
        output_path=output_path,
        fmt=output_fmt,
        genotype_data=gd,
    )

    print(f"Output written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
