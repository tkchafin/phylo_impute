#!/bin/bash


doit() {

  infile=$1
  phylip=$infile
  popmap=`echo $infile | sed 's/.phylip/.popmap/'`
  popmap2=`echo $infile | sed 's/.phylip/.popmap2/'`
  output=`echo $infile | sed 's/.phylip//'`
  threads=1

	echo $phylip
  #make msa
  #Rscript --vanilla ./sim2phylip.R $infile

  #make popmap
  cat $phylip | tail -n +2 | awk '{print $1}' | awk 'BEGIN{FS="."}{print $0 "\t" $1}' > $popmap
  cat $popmap | sed 's/\tpop2/\tpop3/g' > $popmap2

  #truncate alignment to make things go faster (taking 1000bp each)
  #./nremover.pl -i 1.0 -c 0.5 -m -b -r 1000 -t phylip -f $phylip

  #run iqtree
  #iqtree -s $phylip -m MFP -wsr -st DNA -safe -redo -nt $threads
  treefile=$phylip".treefile"
  rate=$phylip".mlrate"
  iqtree=$phylip".iqtree"
	#
  # #get imputation accuracy per method
  python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "global" -o $output"_global"
  #python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "population" -o $output"_pop"
  #python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "nmf" -o $output"_nmf"
  #python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "phylo" -o $output"_phylo"
  #python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "phyloq" -o $output"_phyloq"
  #python3 ./run_validation.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "phyloqr" -o $output"_phyloqr"
  #python3 ./run_validation.py -p $phylip -m $popmap2 -t $treefile -i $iqtree -r $rate --reps 20 --prop 0.2 --method "population2" -o $output"_popWrong"

}

export -f doit

ls simulation/*/*/*.phylip | xargs -n 1 -P 8 -I {} bash -c 'doit "$@"' _ {}
