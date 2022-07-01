#!/usr/bin/env python3

# Based on example from
# /ccs/home/jkitson/.conda/envs/python-netcdf/lib/python3.10/site-packages/netCDF4/__init__.py

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    #parser.add_argument('-n', '--num-metrics', type=int, nargs='+', default=[10],
    #        help='The number of metrics to retain (the n metrics with the heaviest weights ' + \
    #             'will be retained)')

    return parser.parse_args()

def select_metrics(ds, num_to_select=10):
    df = ds.to_dataframe()
    
    filtered_df = df.sort_values(['weights']) \
            .groupby(level=['regions','gcms']) \
            .apply(pd.DataFrame.tail, n=num_to_select) \
            .droplevel(level=[1,2])

    return filtered_df.to_xarray()

def rank_models(ds):
    model_scores = ds['weighted_data'].mean(['regions','metrics'])
    ranked_df = model_scores.sortby(model_scores) \
            .to_dataframe()
    #ranked_df.reset_index(inplace=True)
    ranked_df['rank'] = range(ranked_df.shape[0])
    #print(ranked_df)

    return ranked_df

def find_rank_diffs(df):
    df = df.reset_index()
    df.set_index(['num_metrics','gcms'], inplace=True)
    ranks_df = df['rank'].unstack('num_metrics')
    return ranks_df.diff(axis=1)

def read_lines(path):
    with open(path, 'r') as in_file:
        return [l.strip() for l in in_file]

def save_or_show(save_as=None):
    if save_as is None:
        plt.show()
    else:
        plt.savefig(save_as)

def plot_diffs(df, models, out_dir=None):
    sns.heatmap(df, cmap='vlag', yticklabels=models)
    plt.xlabel('Number of Included Metrics')
    plt.ylabel('Global Climate Models')
    plt.title('Change in GCM rank with metric inclusion')
    plt.tight_layout()

    out_file = None
    if out_dir is not None:
        out_file = os.path.join(out_dir, 'rank_diffs_vs_num_metrics.pdf')
    save_or_show(out_file)

def main():
    args = parse_args()
    in_file = os.path.realpath(args.in_file)
    in_dir = os.path.dirname(os.path.abspath(in_file))
    models_file = args.models_file
    out_dir = args.out_dir
    #num_metrics = args.num_metrics

    ds = xr.open_dataset(in_file, mode='r')
    dfs = []
    for num_metrics in range(1,ds['metrics'].shape[0]):
        filtered_ds = select_metrics(ds, num_to_select=num_metrics)
        ranked_df = rank_models(filtered_ds)
        
        ranked_df['num_metrics'] = num_metrics
        dfs.append(ranked_df)

    df = pd.concat(dfs)
    rank_diffs = find_rank_diffs(df)
    models = read_lines(models_file)
    #plot_diffs(rank_diffs, models, out_dir=out_dir)

    if out_dir is not None:
        rank_diffs_df = pd.DataFrame({'rank_diff': rank_diffs.stack('num_metrics')})
        rank_diffs_ds = rank_diffs_df.to_xarray()
        out_file = os.path.join(out_dir, 'rank_diffs.nc')
        rank_diffs_ds.to_netcdf(out_file, format='NETCDF4_CLASSIC')

if __name__ == '__main__':
    main()
