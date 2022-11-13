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
	data = GenotypeData(
		filename="/Users/tyler/programs/scripts/test_files/internal_gaps.phylip",
		filetype="phylip",
		popmapfile="/Users/tyler/programs/scripts/test_files/internal_gaps.popmap"
	)
	sim=SimGenotypeData(data, prop_missing=0.2, strategy="random")
	imputed=ImputeAlleleFreq(genotype_data=sim, by_populations=True, prefix="test_output")
	print(sim.accuracy(imputed))
	print(sim.accuracy_by_site(imputed))
	print(sim.missingness_by_site())
	print(sim.missingness_by_site(mask=True))

if __name__ == "__main__":
    main()
