#!/usr/bin/env python3

# Based on example from
# /ccs/home/jkitson/.conda/envs/python-netcdf/lib/python3.10/site-packages/netCDF4/__init__.py

import xarray as xr
import pandas as pd

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--in-file',
            default=os.path.join('..','data','metrics.nc'),
            help='The path to the input file')
    parser.add_argument('-o', '--out-file-format',
            default='{in_dir}/metrics_{num_metrics}_heaviest.nc',
            help='The format of the path where to put the output file')
    parser.add_argument('-n', '--num-metrics', type=int, default=10,
            help='The number of metrics to retain (the n metrics with the heaviest weights ' + \
                 'will be retained)')

    return parser.parse_args()

def select_weights(ds, num_to_select=10):
    df = ds.to_dataframe()
    
    filtered_df = df.sort_values(['weights']) \
            .groupby(level=['regions','gcms']) \
            .apply(pd.DataFrame.tail, n=num_to_select) \
            .droplevel(level=[1,2])

    return filtered_df.to_xarray()

def main():
    args = parse_args()
    in_file = os.path.realpath(args.in_file)
    in_dir = os.path.dirname(os.path.abspath(in_file))
    out_file_format = args.out_file_format
    num_metrics = args.num_metrics

    ds = xr.open_dataset(in_file, mode='r')
    filtered_ds = select_weights(ds, num_to_select=num_metrics)

    out_file = out_file_format.format(**locals())
    filtered_ds.to_netcdf(out_file, format='NETCDF4_CLASSIC')

if __name__ == '__main__':
    main()
