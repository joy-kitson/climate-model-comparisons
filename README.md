# Climate Model Comparisions
Plotting scripts for "Evaluation of CMIP6 GCMs over the CONUS for downscaling studies"

## Setup

### Data
Start by cloning this repository locally. Once you have a local copy `cd` into the top level of this repo, and create a directory to store the data in, like so:
```mkdir data```

If you wish to put data in another directory, you may do so, as long as you pass the path to our run script, `run.sh`, later as descrbed in the running section.

Once you have chosen and created your data directory, download and copy the following files into it:
1. `data_cor_ss_ww.nc`
2. `gcms_names.txt`
3. `metrics_list.txt`

### Dependencies
Our scripts are writtne in both the NCAR Command Language (NCL) and Python. Start by making sure you have a working install of both languages. Note that while we used NCL version 6.6.2 and Python version 3.10.4, other version may still work.

For Python, we recommend using an Anaconda installation, as the included package management tools will made installing the various Python packages much easier. We use the following packages in our Python scripts:
1. pandas
2. numpy
3. xarray
4. networkx
5. statsmodels
6. matplotlib
7. seaborn
8. pygraphviz

## Running
To generate the plots and data files for the paper, just `cd` into `src` and run
```./run.sh <data_dir>```
if you decided to put the input data in a directory other than `data`. Otherwise you can leave off the `<data_dir>` argument, and run with:
```./run.sh```
This will run all the neccessary scripts.

## Output Files
