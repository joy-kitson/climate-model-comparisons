#!/usr/bin/bash

# This script runs all scripts needed to generate the plots and data files
# for the paper

# Command line arguments:
#   (1) DATA_DIR - then directory in which to save/load data. ../data by default
# Run this script as ./run.sh [<DAT_DIR>]
if [ -z ${1} ]; then
  DATA_DIR="../data/"
else
  DATA_DIR=${1}
fi

# Echos the command before running it, similar to make
run () {
  echo $@
  $@
  echo
}

# Generates suplimental data files used by other scripts from
# unweighted_data_37gcms_66metrics_0708.nc
run ncl generate_data.ncl dir\='"'${DATA_DIR}'"'

run ncl radius_of_similarity.ncl dir\='"'${DATA_DIR}'"'

run ./plot_rankings_diffs.py -o ${DATA_DIR} -w ww
run ./plot_cos_sim.py -o ${DATA_DIR}

for prog in circo dot sfdp ; do
  run ./plot_network.py ${DATA_DIR}/model_weighted_data_cos_sim_region*.csv \
    -t .8 -a -p ${prog}
done

# Run this last since it takes a while
run ncl heatmap_sample.ncl dir\='"'${DATA_DIR}'"'
