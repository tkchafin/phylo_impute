#!/bin/bash


doit() {

  infile=$1
  phylip=$infile
  popmap=`echo $infile | sed 's/.phylip/.popmap/'`
  popmap2=`echo $infile | sed 's/.phylip/.popmap2/'`
  output=`echo $infile | sed 's/.phylip//'`
  threads=1

  #make msa
  # Rscript --vanilla ./scripts/sim2phylip.R $infile

  #make popmap
  cat $phylip | tail -n +2 | awk '{print $1}' | awk 'BEGIN{FS="."}{print $0 "\t" $1}' > $popmap
  cat $popmap | sed 's/\tpop2/\tpop3/g' > $popmap2

  #make smaller (just for testing)
  #./scripts/nremover.pl -r 200 -t phylip -f $phylip
  #mv $phylip".out" $phylip

  #run iqtree to get guidetree
  # Add -redo to overwrite
  #iqtree -s $phylip -m MFP -wsr -st DNA -safe -nt $threads
  treefile=$phylip".rooted.tre"
  rate=$phylip".mlrate"
  iqtree=$phylip".iqtree"

	echo $phylip
	echo $popmap

  #get imputation accuracy per method
  python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "global" -o $output"_global"
  #python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "population" -o $output"_pop"
  #python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "nmf" -o $output"_nmf"
  #python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "phylo" -o $output"_phylo"
  #python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "phyloq" -o $output"_phyloq"
  #python3 ./run_imputer.py -p $phylip -m $popmap -t $treefile -i $iqtree -r $rate --method "phyloqr" -o $output"_phyloqr"
  #python3 ./run_imputer.py -p $phylip -m $popmap2 -t $treefile -i $iqtree -r $rate --method "population2" -o $output"_popWrong"


  #make structure files (easier to import to adegenet)
  #./scripts/phylip2structure.pl -i $output"_global_imputed.phy" -o $output"_global_imputed.str" -p $popmap
  #./scripts/phylip2structure.pl -i $output"_pop_imputed.phy" -o $output"_pop_imputed.str" -p $popmap
  #./scripts/phylip2structure.pl -i $output"_nmf_imputed.phy" -o $output"_nmf_imputed.str" -p $popmap
  #./scripts/phylip2structure.pl -i $output"_phylo_imputed.phy" -o $output"_phylo_imputed.str" -p $popmap
  #./scripts/phylip2structure.pl -i $output"_phyloq_imputed.phy" -o $output"_phyloq_imputed.str" -p $popmap
  #./scripts/phylip2structure.pl -i $output"_phyloqr_imputed.phy" -o $output"_phyloqr_imputed.str" -p $popmap
  #./scripts/phylip2structure.pl -i $output"_popWrong_imputed.phy" -o $output"_popWrong_imputed.str" -p $popmap

}

export -f doit

#doit missing_data_PCA/MISSdata/cline_mig50_SNP_biasINDV_miss0.01.RData
ls simulation/*/*/*.phylip |head -1 | parallel -j 8 doit {}
