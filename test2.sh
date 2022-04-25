#!/usr/bin/bash


doit() {

  infile=$1
  phylip=`echo $infile | sed 's/.RData/.phy/'`
  popmap=`echo $infile | sed 's/.RData/.popmap/'` 
  popmap2=`echo $infile | sed 's/.RData/.popmap2/'`
  output=`echo $infile | sed 's/.RData//'`
  threads=1

  #make msa
  Rscript --vanilla ./sim2phylip.R $infile

  #make popmap
  #cat $phylip | tail -n +2 | awk '{print $1}' | awk 'BEGIN{FS="."}{print $0 "\t" $1}' > $popmap
  #cat $popmap | sed 's/\tpop2/\tpop3/g' > $popmap2

  #get imputation accuracy per method
  python3 ./run_imputer.py -p $phylip -m $popmap --method "global" -o $output"_global"
}

export -f doit

for file in ./missing_data_PCA/MISSdata/*RData
do
  doit $file 
done
#ls missing_data_PCA/simulation/*replicate*RData | parallel doit {}

