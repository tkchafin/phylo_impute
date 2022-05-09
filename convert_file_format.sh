#!/bin/bash

cd data/test_runs

for f in *imputed.phy;
do
  base1=`echo $f | sed "s/_imputed.phy//g"`
  base2=`echo $base1 | sed -E 's/(.*)_.*/\1/'`
  #echo $base1 $base2
  popmap=$base2".popmap"
  output=$base1".str"
  ../../phylip2structure.pl -i $f -p $popmap -o $output
done

for f in *_popWrong_imputed.phy;
do
  base1=`echo $f | sed "s/_imputed.phy//g"`
  base2=`echo $base1 | sed -E 's/(.*)_.*/\1/'`
  #echo $base1 $base2
  popmap=$base2".popmap2"
  output=$base1".str"
  ../../phylip2structure.pl -i $f -p $popmap -o $output
done



