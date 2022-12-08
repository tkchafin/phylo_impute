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
from impute.simple_imputers.simple_imputers import ImputeReference

def main():
	prefix="test"
	data = GenotypeData(
		filename="/Users/tyler/programs/scripts/test_files/internal_gaps.phylip",
		filetype="phylip",
		popmapfile="/Users/tyler/programs/scripts/test_files/internal_gaps.popmap"
	)
	sim=SimGenotypeData(data, prop_missing=0.2, strategy="random")
	ref_data=data.get_sample_012("C2.TEST")
	imputed=ImputeReference(genotype_data=sim, reference=ref_data, prefix=prefix+"_phylo")
	np.savetxt(prefix+"_acc.tsv", np.array([sim.accuracy(imputed)]), fmt='%-10.5f')
	np.savetxt(prefix+"_accBySite.tsv", sim.accuracy_by_site(imputed), fmt='%-10.5f')
	np.savetxt(prefix+"_missBySite.tsv", sim.missingness_by_site(mask=False), fmt='%-10.5f')
	np.savetxt(prefix+"_maskBySite.tsv", sim.missingness_by_site(mask=True), fmt='%-10.5f')
	sim.decode_imputed(sim.genotypes012_df, prefix=prefix+"_sim")
	np.savetxt(prefix+"_mask.tsv", sim.mask, fmt='%-1d')

if __name__ == "__main__":
    main()
