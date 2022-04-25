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

  #truncate alignment to make things go faster (taking 1000bp each)
  cp $phylip t
  awk 'BEGIN{OFS=FS="\t"} {$2=substr($2,1,1000)}1' t > $phylip
  rm -rf t
  sed -i 's/\t5000/\t1000/g' $phylip

  #run iqtree
  iqtree -s $phylip -m MFP -wsr -st DNA -safe -redo -nt $threads
  treefile=$phylip".treefile"
  rate=$phylip".rate"
  iqtree=$phylip".iqtree"

  #get imputation accuracy per method
  python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "global" -o $output"_global"
  python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "population" -o $output"_pop"
  python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "nmf" -o $output"_nmf"
  python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "phylo" -o $output"_phylo"
  python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "phyloq" -o $output"_phyloq"
  python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "phyloqr" -o $output"_phyloqr"
  python3 ./run_validation.py -p $phylip -m $popmap2 -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "population2" -o $output"_popWrong"

}

export -f doit

ls missing_data_PCA/simulation/*replicate*RData | parallel doit {}

