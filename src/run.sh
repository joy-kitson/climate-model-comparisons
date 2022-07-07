#!/usr/bin/bash

# This script runs all scripts needed to generate the plots and data files for the paper

./plot_rankings_diffs.py -o ../data
./plot_cos_sim.py -o ../data

for prog in circo dot sfdp ; do
  ./plot_network.py ../data/model_weighted_data_cos_sim_region*.csv -t .8 -a -p ${prog}
done
