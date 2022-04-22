#!/usr/bin/bash


doit() {

  infile=$1
  phylip=`echo $infile | sed 's/.RData/.phy/'`
  popmap=`echo $infile | sed 's/.RData/.popmap/'` 
  threads=$2

  #make msa
  Rscript --vanilla ./sim2phylip.R $infile

  #make popmap
  cat $phylip | tail -n +2 | awk '{print $1}' | awk 'BEGIN{FS="."}{print $0 "\t" $1}' > $popmap

  #run iqtree
  iqtree -s $phylip -m MFP -wsr -safe -redo -nt $threads
  treefile=$phylip".treefile"
  rate=$phylip.".rate"
  iqtree=$phylip".iqtree"

  #get imputation accuracy per method
  python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate

  #run pca

}

export -f doit

doit missing_data_PCA/MISSdata/p3_SNP_rand_miss0.2.RData 1

