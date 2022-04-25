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
  cat $phylip | tail -n +2 | awk '{print $1}' | awk 'BEGIN{FS="."}{print $0 "\t" $1}' > $popmap
  cat $popmap | sed 's/\tpop2/\tpop3/g' > $popmap2

  #make smaller (just for testing)
  #./nremover.pl -r 200 -t phylip -f $phylip
  #mv $phylip".out" $phylip

  #run iqtree to get guidetree
  iqtree -s $phylip -m MFP -wsr -st DNA -safe -redo -nt $threads
  treefile=$phylip".treefile"
  rate=$phylip".rate"
  iqtree=$phylip".iqtree"

  #get imputation accuracy per method
  python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "global" -o $output"_global"
  python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "population" -o $output"_pop"
  python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "nmf" -o $output"_nmf"
  python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "phylo" -o $output"_phylo"
  python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "phyloq" -o $output"_phyloq"
  python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "phyloqr" -o $output"_phyloqr"
  python3 ./run_imputer.py -p $phylip -m $popmap2 -t $treefile -i $iqtree -r $rate --method "population2" -o $output"_popWrong"

}

export -f doit

#doit missing_data_PCA/MISSdata/cline_mig50_SNP_biasINDV_miss0.01.RData 
ls missing_data_PCA/MISSdata/*.RData | parallel -j 4 doit {}

