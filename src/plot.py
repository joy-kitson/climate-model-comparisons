#!/usr/bin/env python3

# Based on example from
# /ccs/home/jkitson/.conda/envs/python-netcdf/lib/python3.10/site-packages/netCDF4/__init__.py

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import argparse
import functools
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--in-file',
            default=os.path.join('..','data','metrics.nc')
            help='The path to the input file')
    parser.add_argument('-o', '--out-file',
            default='metrics.pdf',
            help='The path to the output file')

    return parser.parse_args()

def main():
    args = parse_args()
    in_file = os.path.realpath(args.in_file)
    out_file = args.out_file

    data = xr.open_dataset(in_file, mode='r')

if __name__ == '__main__':
    main()
