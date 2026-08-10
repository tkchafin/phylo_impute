# phylo-impute

Phylogenetic imputation of missing genotypes using continuous-time Markov chain (CTMC) models on a guide tree. Supports both haploid (4-state: A, C, G, T) and diploid (10-state: AA, CC, ..., GT) models.

## Installation

```bash
pip install .
```

Requires Python >= 3.11.

### Dependencies

- numpy, pandas, scipy
- [toytree](https://github.com/eaton-lab/toytree)
- [SNPio](https://github.com/btmartin721/SNPio) (for VCF/PHYLIP/STRUCTURE/GENEPOP I/O)

### Optional

- [IQ-TREE](http://www.iqtree.org/) — to infer the tree and substitution model from your data

## Quick start

```bash
# 1. Run IQ-TREE on your alignment
iqtree2 -s alignment.phy -m GTR+I+R3 -T 4

# 2. Impute missing data (haploid)
phylo-impute \
  --input alignment.phy \
  --treefile alignment.phy.treefile \
  --qmatrix alignment.phy.iqtree

# 3. Impute with the diploid model
phylo-impute \
  --input genotypes.vcf \
  --treefile tree.nwk \
  --qmatrix model.iqtree \
  --diploid
```

## CLI reference

```
phylo-impute --input FILE --treefile FILE --qmatrix FILE [options]
```

### Required arguments

| Argument | Description |
|----------|-------------|
| `--input FILE` | Input genotype file (VCF, PHYLIP, STRUCTURE, or GENEPOP) |
| `--treefile FILE` | Newick tree file |
| `--qmatrix FILE` | Q-matrix file (IQ-TREE `.iqtree` report or plain-text 4×4 matrix) |

### Optional arguments

| Argument | Description |
|----------|-------------|
| `--diploid` | Use the 10-state diploid genotype model (GTGTR4) instead of the default 4-state haploid model |
| `--evaluate FRAC` | Mask FRAC (0–1) of known genotypes and report imputation accuracy vs. majority rule |
| `--seed INT` | Random seed for `--evaluate` masking (default: 42) |
| `--reference SAMPLE` | Use this dataset sample as reference-fill baseline (requires `--evaluate`) |
| `--reference-fasta PATH` | External FASTA reference for reference-fill baseline (requires `--evaluate`; must match alignment dimensions) |
| `--siterates FILE` | Per-site evolutionary rates file (one rate per site, same order as input) |
| `--format FMT` | Input format: `vcf`, `phylip`, `structure`, `genepop` (default: infer from extension) |
| `--output-format FMT` | Output format: `vcf`, `phylip`, `structure`, `genepop`, `fasta` (default: same as input) |
| `--popmap FILE` | Population map file (for SNPio) |
| `--prefix STR` | Output prefix (default: input file stem) |
| `--verbose` | Print progress messages |
| `--version` | Show version and exit |

### Input

The input genotype file can be in any format supported by SNPio. Missing data tokens recognized: `N`, `n`, `-9`, `.`, `?`, `./.`, empty string.

The Q-matrix is parsed from IQ-TREE's `.iqtree` output file (the "Rate matrix Q" section). The tree should have the same tip labels as the samples in the genotype file.

### Output

The output file is written to `{prefix}_imputed.{format}`. All missing genotypes are replaced with the maximum a posteriori (MAP) estimate from the phylogenetic model. The output retains the same sample order and format as the input.

## Models

### Haploid (default)

Standard 4-state nucleotide CTMC (A, C, G, T) with a GTR-family substitution model. Suitable for haploid data or consensus sequences.

### Diploid (`--diploid`) — *experimental, unvalidated*

> **Warning:** The diploid model is experimental and has not been validated on empirical data. Use with caution and verify results independently.

10-state unordered diploid genotype model (GTGTR4), with states AA, CC, GG, TT, AC, AG, AT, CG, CT, GT (IUPAC codes A, C, G, T, M, R, W, S, Y, K). The 10×10 rate matrix is constructed from 6 GTR exchangeability rates (extracted from the 4-state Q-matrix) and 10 genotype state frequencies (estimated from the data). This follows the GTGTR4 model implemented in [RAxML-NG](https://github.com/amkozlov/raxml-ng) and [coraxlib](https://codeberg.org/Exelixis-Lab/coraxlib) (Kozlov et al., 2019).

### Evaluation (`--evaluate`)

When `--evaluate 0.1` is passed, 10% of known genotypes are randomly masked before imputation. After imputation, a table is printed comparing phylogenetic imputation accuracy against a simple most-frequent-allele baseline, broken down by all sites and variable sites only.

A reference-based imputation baseline can be added with either:

- `--reference SAMPLE` — use a sample from the dataset as the reference (excluded from masking)
- `--reference-fasta PATH` — use an external FASTA file with a reference sequence (must match alignment length)

Both options add a "Reference fill" row to the accuracy table, showing naive reference-based imputation (replacing missing sites with the reference allele) as an additional comparison.

**Note on site indexing:** The reference sequence, site rates, and genotype data are all matched by position index (site *i* in one = site *i* in the others). For full-genome alignments (FASTA/PHYLIP), this is straightforward. For SNP-based formats (VCF, STRUCTURE, GENEPOP), the reference and site rates must contain only the SNP columns in the same order as the input — not the full genome.

```bash
# Compare against an internal sample as reference
phylo-impute --input aln.phy --treefile tree.nwk --qmatrix model.iqtree \
  --evaluate 0.1 --reference sample_name

# Compare against an external reference sequence (e.g. Wuhan-Hu-1)
phylo-impute --input aln.phy --treefile tree.nwk --qmatrix model.iqtree \
  --evaluate 0.1 --reference-fasta reference.fasta
```

## Python API

```python
from phylo_impute import (
    phylogenetic_imputation,         # 4-state haploid
    phylogenetic_imputation_diploid, # 10-state diploid
    build_q10_from_gtr,
    repair_and_scale_q,
    q_from_iqtree,
)
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

GPL-3.0-or-later
