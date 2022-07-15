#!/usr/bin/env python3

# Based on example from
# /ccs/home/jkitson/.conda/envs/python-netcdf/lib/python3.10/site-packages/netCDF4/__init__.py

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import save_or_show, read_lines, write_lines

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--in-file',
            default=os.path.join('..','data','metrics.nc'),
            help='The path to the input file')
    parser.add_argument('-m', '--models-file',
            default=os.path.join('..','data','gred.dat'),
            help='The path to a file containing the names of the GCMS')
    parser.add_argument('-o', '--out-dir', default=None,
            help='The optional path to a directory to save the output file')
    parser.add_argument('-u', '--use-unweighted', action='store_true',
            help='Pass this flag to use the unweighted data')
    parser.add_argument('-r', '--plot-regions', action='store_true',
            help='Pass this flag to plot eahc reigon`s data seperately')

    return parser.parse_args()

def get_cosine_similarity(da, dim, axis=0):
    num_to_compare = da[dim].shape[0]
    
    dot_prods = np.zeros((num_to_compare, num_to_compare))
    for i in range(num_to_compare):
        for j in range(num_to_compare):
            dot_prods[i,j] = xr.dot(da.sel({dim: i}), da.sel({dim: j}))

    norms = np.linalg.norm(da, axis=axis)

    return dot_prods / np.outer(norms, norms)

def plot_similarity(cos_sim, models, out_dir=None, var='weighted', region=None):
    sns.set(rc={"figure.figsize": (8, 8)})
    plt.figure()
    fig = sns.heatmap(cos_sim, cmap='rocket', cbar_kws={'label': 'Cosine Similarity'},
            xticklabels=models, yticklabels=models, square=True)
    fig.set_xticklabels(models, size=10)
    fig.set_yticklabels(models, size=10)
    plt.xlabel('Global Climate Model')
    plt.ylabel('Global Climate Model')
    title_var = var.replace('_',' ')
    plt.title(f'Cosine similarity between models for {title_var}')
    plt.tight_layout()

    out_file = None
    if out_dir is not None:
        if region is not None:
            out_file = os.path.join(out_dir, f'model_{var}_cos_sim_region_{region}.pdf')
        else:
            out_file = os.path.join(out_dir, f'model_{var}_cos_sim.pdf')
    save_or_show(out_file)

def plot_distribution(cos_sim, models, out_dir=None, var='weighted', region=None):
    flat_cos_sim = np.triu(cos_sim, k=1) \
            .flatten()
    cos_sim_values = flat_cos_sim[flat_cos_sim > 0]
    
    sns.set(rc={"figure.figsize": (8, 8)})
    plt.figure()
    fig = sns.histplot(cos_sim_values, kde=True, bins=50)
    title_var = var.replace('_',' ')
    plt.title(f'Cosine similarity distribution between models for {title_var}')
    plt.xlabel('Cosine similarity')
    plt.tight_layout()

    out_file = None
    if out_dir is not None:
        if region is not None:
            out_file = os.path.join(out_dir, f'model_{var}_cos_sim_distrib_region_{region}.pdf')
        else:
            out_file = os.path.join(out_dir, f'model_{var}_cos_sim_distrib.pdf')
    save_or_show(out_file)
    
    if out_dir is not None:
        if region is not None:
            out_file = os.path.join(out_dir, f'model_{var}_cos_sim_distrib_region_{region}.csv')
        else:
            out_file = os.path.join(out_dir, f'model_{var}_cos_sim_distrib.csv')
        write_lines(cos_sim_values, out_file)


def main():
    args = parse_args()
    in_file = os.path.realpath(args.in_file)
    in_dir = os.path.dirname(os.path.abspath(in_file))
    models_file = args.models_file
    out_dir = args.out_dir

    var = 'weighted_data'
    if args.use_unweighted:
        var = 'unweighted_data'

    ds = xr.open_dataset(in_file, mode='r')

    if args.plot_regions:
        groups = ds[var].groupby('regions')
        for region, da in groups:
            cos_sim = get_cosine_similarity(da, 'gcms')
            models = read_lines(models_file)
            plot_similarity(cos_sim, models, out_dir=out_dir, var=var, region=region)
            plot_distribution(cos_sim, models, out_dir=out_dir, var=var, region=region)

            if out_dir is not None:
                out_file = os.path.join(out_dir, f'model_{var}_cos_sim_region_{region}.csv')
                np.savetxt(out_file, cos_sim, delimiter=',')
    
    cos_sim = get_cosine_similarity(ds[var], 'gcms', axis=(0,1))
    models = read_lines(models_file)
    plot_similarity(cos_sim, models, out_dir=out_dir, var=var)
    plot_distribution(cos_sim, models, out_dir=out_dir, var=var)

    if out_dir is not None:
        out_file = os.path.join(out_dir, f'model_{var}_cos_sim_region.csv')
        np.savetxt(out_file, cos_sim, delimiter=',')

if __name__ == '__main__':
    main()
