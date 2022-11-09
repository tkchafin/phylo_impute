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
	print(data.snps)
	imputed=ImputeAlleleFreq(genotype_data=data, by_populations=False, prefix="test_output")

if __name__ == "__main__":
    main()
