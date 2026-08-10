from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import toytree as tt


def infer_format(path: str) -> str:
    """Infer input format from file extension."""
    p = Path(path)
    suffixes = [s.lower() for s in p.suffixes]
    if ".vcf" in suffixes:
        return "vcf"
    elif p.suffix.lower() in (".fa", ".fasta", ".fas", ".fna"):
        return "fasta"
    elif p.suffix.lower() in (".phy", ".phylip"):
        return "phylip"
    elif p.suffix.lower() in (".str", ".structure"):
        return "structure"
    elif p.suffix.lower() in (".gen", ".genepop"):
        return "genepop"
    else:
        raise ValueError(
            f"Cannot infer format from extension '{p.suffix}'. "
            f"Use --format to specify: fasta, vcf, phylip, structure, genepop"
        )


class FastaData:
    """Lightweight wrapper for FASTA input that provides the interface
    expected by the CLI (`.snpsdict`) without depending on SNPio."""

    def __init__(self, records: List[tuple]) -> None:
        self._records = records
        self._snpsdict = {name: list(seq) for name, seq in records}

    @property
    def snpsdict(self) -> Dict[str, List[str]]:
        return self._snpsdict

    @snpsdict.setter
    def snpsdict(self, value: Dict[str, List[str]]) -> None:
        self._snpsdict = value


def load_genotype_data(
    input_path: str,
    fmt: Optional[str] = None,
    popmap: Optional[str] = None,
    prefix: str = "phylo_impute",
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Load genotype data using the appropriate reader.

    For FASTA input, uses a lightweight built-in reader (no SNPio dependency).
    For all other formats, uses the corresponding SNPio reader.

    Returns:
        Object with a `.snpsdict` property (Dict[str, List[str]]).
    """
    if fmt is None:
        fmt = infer_format(input_path)

    fmt = fmt.lower()

    if fmt == "fasta":
        records = read_fasta(input_path)
        if verbose:
            print(f"  Read {len(records)} sequences, {len(records[0][1])} sites")
        return FastaData(records)
    elif fmt == "vcf":
        from snpio import VCFReader

        return VCFReader(
            filename=input_path,
            popmapfile=popmap,
            force_popmap=popmap is not None,
            prefix=prefix,
            verbose=verbose,
        )
    elif fmt == "phylip":
        from snpio import PhylipReader

        return PhylipReader(
            filename=input_path,
            popmapfile=popmap,
            force_popmap=popmap is not None,
            prefix=prefix,
            verbose=verbose,
        )
    elif fmt == "structure":
        from snpio import StructureReader

        return StructureReader(
            filename=input_path,
            popmapfile=popmap,
            force_popmap=popmap is not None,
            prefix=prefix,
            verbose=verbose,
        )
    elif fmt == "genepop":
        from snpio import GenePopReader

        return GenePopReader(
            filename=input_path,
            popmapfile=popmap,
            force_popmap=popmap is not None,
            prefix=prefix,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def load_tree_data(
    genotype_data: Any,
    treefile: str,
    qmatrix: Optional[str] = None,
    siterates: Optional[str] = None,
    verbose: bool = False,
) -> Any:
    """Load tree, Q-matrix, and site rates.

    For FASTA input (FastaData), reads tree/qmatrix/siterates directly.
    For SNPio objects, uses SNPio's TreeParser.

    Returns:
        Object with .tree, .qmat, and optionally .site_rates attributes.
    """
    if isinstance(genotype_data, FastaData):
        return _load_tree_data_direct(
            treefile=treefile,
            qmatrix=qmatrix,
            siterates=siterates,
            verbose=verbose,
        )

    from snpio import TreeParser

    try:
        return TreeParser(
            genotype_data,
            treefile=treefile,
            qmatrix=qmatrix,
            siterates=siterates,
            verbose=verbose,
        )
    except (ValueError, Exception) as e:
        # SNPio's TreeParser can fail on plain-text Q matrices.
        # Fall back to direct parsing.
        if qmatrix is not None and "mismatch" in str(e).lower() or "set_index" in str(e).lower():
            if verbose:
                print(f"  SNPio TreeParser failed on Q matrix ({e}), using built-in parser")
            return _load_tree_data_direct(
                treefile=treefile,
                qmatrix=qmatrix,
                siterates=siterates,
                verbose=verbose,
            )
        raise


class _DirectTreeData:
    """Simple container for tree, Q-matrix, and site rates loaded directly."""

    def __init__(self, tree, qmat, site_rates=None):
        self.tree = tree
        self.qmat = qmat
        self.site_rates = site_rates


def _load_tree_data_direct(
    treefile: str,
    qmatrix: Optional[str] = None,
    siterates: Optional[str] = None,
    verbose: bool = False,
) -> _DirectTreeData:
    """Load tree/qmatrix/siterates without SNPio."""
    tree = tt.tree(treefile)
    if verbose:
        print(f"  Tree: {tree.ntips} tips")

    qmat = None
    if qmatrix is not None:
        qmat = _read_qmatrix(qmatrix)

    rates = None
    if siterates is not None:
        rates = parse_site_rates(siterates)
        if verbose:
            print(f"  Site rates: {len(rates)} values")

    return _DirectTreeData(tree=tree, qmat=qmat, site_rates=rates)


def _read_qmatrix(path: str) -> pd.DataFrame:
    """Read Q-matrix from IQ-TREE .iqtree file or plain-text 4x4 matrix."""
    with open(path) as fh:
        content = fh.read()

    if "Rate matrix Q" in content:
        return q_from_iqtree(path)

    # Plain-text: 4 rows × 4 columns, optionally with row labels A C G T
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    states = ["A", "C", "G", "T"]
    rows = []
    for line in lines:
        parts = line.split()
        # Skip header/label if first token is a state letter
        vals = [p for p in parts if p.replace(".", "").replace("-", "").replace("e", "").replace("+", "").isdigit() or _is_float(p)]
        if len(vals) >= 4:
            rows.append([float(v) for v in vals[:4]])
    if len(rows) != 4:
        raise ValueError(
            f"Could not parse 4x4 Q matrix from {path}. "
            f"Expected 4 rows of 4 numbers."
        )
    return pd.DataFrame(rows, index=states, columns=states)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_site_rates(path: str) -> List[float]:
    """Parse per-site rates from a file.

    Supports two formats:
    1. One rate per line (simple)
    2. IQ-TREE tabular format with header (Site Rate Cat C_rate)
    """
    rates = []
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]

    if not lines:
        raise ValueError(f"No rates found in {path}")

    # Detect IQ-TREE tabular format
    header = lines[0].lower()
    if "site" in header and "rate" in header:
        # IQ-TREE format: skip header, take 'Rate' column (index 1)
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                rates.append(float(parts[1]))
    else:
        # Simple: one rate per line
        for line in lines:
            parts = line.split()
            rates.append(float(parts[0]))

    return rates


def q_from_iqtree(iqtree_file: str) -> pd.DataFrame:
    """Parse Rate matrix Q from an IQ-TREE .iqtree report file.

    Returns:
        4x4 Q matrix as DataFrame with A,C,G,T index/columns.
    """
    q: Dict[str, Dict[str, float]] = {
        n1: {n2: 0.0 for n2 in ["A", "C", "G", "T"]}
        for n1 in ["A", "C", "G", "T"]
    }
    qlines: List[List[str]] = []

    with open(iqtree_file) as fh:
        found = False
        matlinecount = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if "Rate matrix Q" in line:
                found = True
                continue
            if found:
                matlinecount += 1
                if matlinecount > 4:
                    break
                qlines.append(line.split())

    if len(qlines) != 4:
        raise ValueError(
            f"Could not find complete 4x4 Rate matrix Q in {iqtree_file}"
        )

    order = [row[0] for row in qlines]
    for row in qlines:
        for index in range(4):
            q[row[0]][order[index]] = float(row[index + 1])

    return pd.DataFrame(q).T


def write_imputed(
    genotypes: Dict[str, List[str]],
    sample_names: List[str],
    output_path: str,
    fmt: str = "phylip",
    genotype_data: Any = None,
) -> None:
    """Write imputed genotypes to file.

    Args:
        genotypes: Dict mapping sample name -> list of nucleotide chars.
        sample_names: Ordered list of sample names.
        output_path: Output file path.
        fmt: Output format (fasta, phylip, vcf, structure, genepop).
        genotype_data: Original SNPio GenotypeData object (required for
            vcf, structure, genepop output).
    """
    fmt = fmt.lower()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "fasta":
        with open(path, "w") as out:
            for name in sample_names:
                seq = "".join(genotypes[name])
                out.write(f">{name}\n")
                for i in range(0, len(seq), 80):
                    out.write(seq[i : i + 80] + "\n")
    elif fmt == "phylip":
        seqs = [genotypes[name] for name in sample_names]
        n_samples = len(sample_names)
        n_sites = len(seqs[0])
        with open(path, "w") as out:
            out.write(f"{n_samples} {n_sites}\n")
            for name, seq in zip(sample_names, seqs):
                out.write(f"{name}\t{''.join(seq)}\n")
    elif fmt in ("vcf", "structure", "genepop"):
        if genotype_data is None or isinstance(genotype_data, FastaData):
            raise ValueError(
                f"Output format '{fmt}' requires SNPio input data. "
                f"Use --output-format fasta or phylip for FASTA input, "
                f"or use a SNPio-supported input format (vcf, phylip, "
                f"structure, genepop)."
            )
        # Update the SNPio object's genotype data in place
        genotype_data.snpsdict = genotypes
        writer = {
            "vcf": genotype_data.write_vcf,
            "structure": genotype_data.write_structure,
            "genepop": genotype_data.write_genepop,
        }[fmt]
        writer(str(path))
    else:
        raise ValueError(f"Unsupported output format: {fmt}")


# --- FASTA helpers ---


def read_fasta(path: str) -> List[tuple]:
    """Read a FASTA file into a list of (name, sequence) tuples."""
    records = []
    name = None
    seq_parts: List[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(seq_parts)))
                name = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.upper())

    if name is not None:
        records.append((name, "".join(seq_parts)))

    if not records:
        raise ValueError("No FASTA records read")

    lengths = {len(seq) for _, seq in records}
    if len(lengths) != 1:
        raise ValueError(
            f"Sequences are not aligned; found lengths: {sorted(lengths)}"
        )

    return records


def write_fasta(records: List[tuple], path: str, width: int = 80) -> None:
    """Write records to FASTA file."""
    with open(path, "w") as out:
        for name, seq in records:
            out.write(f">{name}\n")
            for i in range(0, len(seq), width):
                out.write(seq[i : i + width] + "\n")


def records_to_genotype_dict(records: List[tuple]) -> Dict[str, List[str]]:
    """Convert FASTA records to genotype dict."""
    return {name: list(seq) for name, seq in records}


def extract_gtr_rates_from_q4(q4_df: pd.DataFrame) -> np.ndarray:
    """Extract 6 GTR exchangeability rates from a 4x4 Q matrix.

    The exchangeability rate s_ij = Q[i,j] / pi[j], where pi is the
    stationary distribution (proportional to the column sums of off-diag).

    Returns rates in canonical order: [AC, AG, AT, CG, CT, GT],
    normalized so GT = 1.0.
    """
    from phylo_impute.core import stationary_pi_from_q

    states = ["A", "C", "G", "T"]
    Q = q4_df.loc[states, states].to_numpy(dtype=float)
    pi = stationary_pi_from_q(Q)

    # Pairs in canonical order: AC, AG, AT, CG, CT, GT
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    rates = np.zeros(6, dtype=float)
    for k, (i, j) in enumerate(pairs):
        if pi[j] > 1e-12:
            rates[k] = Q[i, j] / pi[j]
        else:
            rates[k] = Q[j, i] / pi[i] if pi[i] > 1e-12 else 1.0

    # Normalize so last rate (GT) = 1.0
    if rates[5] > 0:
        rates /= rates[5]
    else:
        # Fallback: normalize by max
        mx = rates.max()
        if mx > 0:
            rates /= mx

    return rates
