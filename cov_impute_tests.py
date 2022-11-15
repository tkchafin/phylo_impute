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

sys.setrecursionlimit(10**7)

def main():
	data = GenotypeData(
		filename="/home/tkchafin/cov_impute/lanfear_tree/all_seqs.align_targets.gapfix.phylip",
		filetype="phylip",
		siterates_iqtree="/home/tkchafin/cov_impute/lanfear_tree/all_seqs.align_targets.phylip.rate",
		qmatrix_iqtree="/home/tkchafin/cov_impute/lanfear_tree/all_seqs.align_targets.phylip.iqtree",
		guidetree="/home/tkchafin/cov_impute/lanfear_tree/all_seqs.align_targets.phylip.treefile"
	)
	sim=SimGenotypeData(data, prop_missing=0.2, strategy="random")
	imputed=ImputePhylo(genotype_data=sim, prefix="test_output")
	print(sim.accuracy(imputed))
	print(sim.accuracy_by_site(imputed))
	print(sim.missingness_by_site())
	print(sim.missingness_by_site(mask=True))

if __name__ == "__main__":
    main()
