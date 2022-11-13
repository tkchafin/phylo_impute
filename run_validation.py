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


	#for each of random, nonrandom, and non-random weighted, do 100 replicates and get avg accuracy
	out_string=str(args.output) + "\t" + str(args.method) + "\t" + str(args.prop) + "\t"
	output=list()
	if args.reps > 0:
		for missing_type in ["random", "nonrandom", "nonrandom_weighted"]:
			for i in range(1,args.reps+1):
				print(missing_type, " rep ", str(i))
				out_i=out_string + str(missing_type) + "\t" + str(i) + "\t"
				sim=SimGenotypeData(data, prop_missing=args.prop, subset=0.1, strategy=missing_type)
				#print(sim)
				if args.method=="phylo":
					sim.q = None
					sim.site_rates=None
					imputed=ImputePhylo(genotype_data=sim, write_output=False)
				elif args.method=="phyloq":
					sim.site_rates=None
					imputed=ImputePhylo(genotype_data=sim, write_output=False)
				elif args.method=="phyloqr":
					imputed=ImputePhylo(genotype_data=sim, write_output=False)
				elif args.method=="global":
					imputed=ImputeAlleleFreq(genotype_data=sim, by_populations=False, write_output=False)
				elif "pop" in args.method:
					imputed=ImputeAlleleFreq(genotype_data=sim, by_populations=True, write_output=False)
				elif args.method=="nmf":
					imputed=ImputeNMF(genotype_data=sim, latent_features=2, max_iter=1000, n_fail=100, write_output=False)
				else:
					print("No imputation method selected")
					sys.exit()
				accuracy=sim.accuracy(imputed)
				out_i=out_i + str(accuracy) + "\n"
				output.append(out_i)
				#print(out_i)
				#sys.exit()

	with open((str(args.output) + ".impute.out"), "w") as fh:
		fh.writelines(output)

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
		"--reps",
		type=int,
		required=False,
		default=100,
		help="Number of replicates for calculating accuracy (set to 0 to skip this step)",
	)

	optional_args.add_argument(
		"--method",
		type=str,
		required=False,
		default="global",
		help="Imputation method. Must be one of: global, populations, phylo, phyloq, phyloqr, or nmf",
	)

	optional_args.add_argument(
		"--prop",
		type=float,
		required=False,
		default=0.2,
		help="Proportion of missing data to simulate for computing imputation accuracy (only if --reps > 0)",
	)

	optional_args.add_argument(
		"--resume_imputed",
		type=str,
		required=False,
		help="Read in imputed data from a file instead of doing the imputation",
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
