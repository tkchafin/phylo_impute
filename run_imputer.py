#!/usr/bin/env python

# Standard library imports
import argparse
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats

import impute

from impute.read_input.read_input import GenotypeData
from impute.read_input.simgenodata import SimGenotypeData
from impute.simple_imputers.simple_imputers import ImputePhylo
from impute.simple_imputers.simple_imputers import ImputeNMF
from impute.simple_imputers.simple_imputers import ImputeAlleleFreq


def main():
    """Class instantiations and main package body"""

    args = get_arguments()

    if args.str and args.phylip:
        sys.exit("Error: Only one file type can be specified")

        # If VCF file is specified.
    if args.str:
        if not args.pop_ids and args.popmap is None:
            raise TypeError("Either --pop_ids or --popmap must be specified\n")

        if args.pop_ids:
            print("\n--pop_ids was specified as column 2\n")
        else:
            print(
                "\n--pop_ids was not specified; "
                "using popmap file to get population IDs\n"
            )

        if args.onerow_perind:
            print("\nUsing one row per individual...\n")
        else:
            print("\nUsing two rows per individual...\n")

        if args.onerow_perind:
            data = GenotypeData(
                filename=args.str,
                filetype="structure1row",
                popmapfile=args.popmap,
                guidetree=args.treefile,
                qmatrix_iqtree=args.iqtree,
            )
        else:
            data = GenotypeData(
                filename=args.str,
                filetype="structure2row",
                popmapfile=args.popmap,
                guidetree=args.treefile,
                qmatrix_iqtree=args.iqtree,
		siterates_iqtree=args.rates,
            )

    if args.phylip:
        if args.pop_ids or args.onerow_perind:

            print(
                "\nPhylip file was used with structure arguments; ignoring "
                "structure file arguments\n"
            )

        if args.popmap is None:
            raise TypeError("No popmap file supplied with PHYLIP file\n")

        data = GenotypeData(
            filename=args.phylip,
            filetype="phylip",
            popmapfile=args.popmap,
            guidetree=args.treefile,
            qmatrix_iqtree=args.iqtree,
            siterates_iqtree=args.rates,
        )


    if args.method=="phylo":
        data.q = None
        data.site_rates=None
        imputed=ImputePhylo(genotype_data=data, prefix=args.output)
    elif args.method=="phyloq":
        data.site_rates=None
        imputed=ImputePhylo(genotype_data=data, prefix=args.output)
    elif args.method=="phyloqr":
        imputed=ImputePhylo(genotype_data=data, prefix=args.output)
    elif args.method=="global":
        imputed=ImputeAlleleFreq(genotype_data=data, by_populations=False, prefix=args.output)
    elif "pop" in args.method:
        imputed=ImputeAlleleFreq(genotype_data=data, by_populations=True, prefix=args.output)
    elif args.method=="nmf":
        imputed=ImputeNMF(genotype_data=data, latent_features=2, max_iter=1000, n_fail=100, prefix=args.output)
    else:
        print("No imputation method selected")
        sys.exit()
    
    #imputed.decode_imputed(imputed.imputed012, write_output=True, prefix=args.output)    


def get_arguments():
    """[Parse command-line arguments. Imported with argparse]

    Returns:
        [argparse object]: [contains command-line arguments; accessed as method]
    """

    parser = argparse.ArgumentParser(
        description="Phylogenetic maximum likelihood genotype imputation",
        add_help=False,
    )

    required_args = parser.add_argument_group("Required arguments")
    filetype_args = parser.add_argument_group(
        "File type arguments (choose only one)"
    )
    structure_args = parser.add_argument_group("Structure file arguments")
    optional_args = parser.add_argument_group("Optional arguments")

    # File Type arguments
    filetype_args.add_argument(
        "-s", "--str", type=str, required=False, help="Input structure file"
    )
    filetype_args.add_argument(
        "-p", "--phylip", type=str, required=False, help="Input phylip file"
    )
    filetype_args.add_argument(
        "-o", "--output", type=str, required=False, help="prefix for output"
    )


    filetype_args.add_argument(
        "-t",
        "--treefile",
        type=str,
        required=False,
        default=None,
        help="Newick-formatted treefile",
    )

    filetype_args.add_argument(
        "-i",
        "--iqtree",
        type=str,
        required=False,
        help=".iqtree output file containing Rate Matrix Q",
    )
    filetype_args.add_argument(
        "-r",
        "--rates",
        type=str,
        required=False,
        help=".rate output file containing site rates table",
    )


    # Structure Arguments
    structure_args.add_argument(
        "--onerow_perind",
        default=False,
        action="store_true",
        help="Toggles on one row per individual option in structure file",
    )
    structure_args.add_argument(
        "--pop_ids",
        default=False,
        required=False,
        action="store_true",
        help="Toggles on population ID column (2nd col) in structure file",
    )

    ## Optional Arguments
    optional_args.add_argument(
        "-m",
        "--popmap",
        type=str,
        required=False,
        default=None,
        help="Two-column tab-separated population map file: inds\tpops. No header line",
    )
    optional_args.add_argument(
        "--prefix",
        type=str,
        required=False,
        default="output",
        help="Prefix for output files",
    )

    optional_args.add_argument(
        "--method",
        type=str,
        required=False,
        default="global",
        help="Imputation method. Must be one of: global, populations, phylo, phyloq, phyloqr, or nmf",
    )

    # Add help menu
    optional_args.add_argument(
        "-h", "--help", action="help", help="Displays this help menu"
    )

    # If no command-line arguments are called then exit and call help menu.
    if len(sys.argv) == 1:
        print("\nExiting because no command-line options were called.\n")
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
