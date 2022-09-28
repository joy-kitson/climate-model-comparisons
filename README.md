# Climate Model Comparisons
Plotting scripts for "Evaluation of CMIP6 GCMs over the CONUS for downscaling studies"

## Setup

### Data
We include the starting data in this repo in the `data` directory. If you wish to put data in another directory, you may do so, as long as you pass the path to our run script, `run.sh`, later as descrbed in the running section and put *all* the input data there.

The input data files are as follows:
1. `unweighted_data_37gcms_66metrics_0708.nc` contains the unweighted values for each GCM on each metric. The order of the GCMs and metrics are described by the other two files, and the regions are in the order north, east, west, and south.
2. `gcms_names.txt` contains the names of the GCMs in the file above, in the same order as they are presented there; the $i$-th line of this file identifies the $i$-th GCM in that file
3. `metrics_list.txt` contains the names of the metrics in the first file, in the same order as they are presented there; the $i$-th line of this file identifies the $i$-th metric in that file

### Dependencies
Our scripts are written in both the NCAR Command Language (NCL) and Python. Start by making sure you have a working install of both languages. Note that while we used NCL version 6.6.2 and Python version 3.10.4, other versions may still work.

For Python, we recommend using an Anaconda installation, as the included package management tools will made installing the various Python packages much easier. We use the following packages in our Python scripts:
1. pandas
2. numpy
3. xarray
4. networkx
5. statsmodels
6. matplotlib
7. seaborn
8. pygraphviz
For convenience, we provide a file containing a list of all the packages installed in our local Anaconda environment, `environment.yml`. This file can be used to create and activate a new Anaconda environment containign the neccessary dependencies by `cd`ing into the `src` directory and running:
```conda env create environment.yml```
Note that this may take a while. Once this is done, the environment can be activated like so
```conda activate gcms```

## Running
To generate the plots and data files for the paper, just `cd` into `src` and run
```./run.sh <data_dir>```
if you decided to put the input data in a directory other than `data`. Otherwise you can leave off the `<data_dir>` argument, and run with:
```./run.sh```
This will run all the neccessary scripts. By default, this will put all of the output date in `data` alongside the input data in this repo. 

## Output Files
