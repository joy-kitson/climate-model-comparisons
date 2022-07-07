#!/usr/bin/env python3

# Based on example from
# /ccs/home/jkitson/.conda/envs/python-netcdf/lib/python3.10/site-packages/netCDF4/__init__.py

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import save_or_show, read_lines
from statsmodels.stats.weightstats import ttost_paired

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
    parser.add_argument('-e', '--epsilon', type=float, default=.05,
            help='For the paired t-tests, the maximum difference between the means of the ' + \
                 'two samples under the alternate hypothesis')
    parser.add_argument('-u', '--use-unweighted', action='store_true',
            help='Pass this flag to use the unweighted data')
    #parser.add_argument('-n', '--num-metrics', type=int, nargs='+', default=[10],
    #        help='The number of metrics to retain (the n metrics with the heaviest weights ' + \
    #             'will be retained)')

    return parser.parse_args()

def select_metrics(ds, by='weights', var='weighted_data', num_to_select=10, region=0):
    df = ds.isel(regions=region)[[var, by]] \
            .to_dataframe()
   
    filtered_df = df.sort_values(by)
    
    # No need to do anything more than sort if we're keeping all the vlaues
    if num_to_select < ds['metrics'].shape[0]:
        filtered_df = filtered_df.groupby(level=['gcms']) \
            .apply(pd.DataFrame.tail, n=num_to_select) \
            .droplevel(level=0)

    return filtered_df.to_xarray()

def rank_models(ds, var='weighted_data'):
    model_scores = ds[var].mean(['metrics'])
    ranked_df = model_scores.sortby(model_scores, ascending=False) \
            .to_dataframe()
    ranked_df['rank'] = range(ranked_df.shape[0])
    #print(ranked_df)

    return ranked_df

def run_t_test(ranked_df, ds, epsilon=.05, var='weighted_data'):
    # Ranked_df has the model order, which ds needs
    ranked_ds = ds.sel(gcms=ranked_df.index)[var]
    
    p_values = np.zeros(ranked_ds['gcms'].shape)
    p_values[:] = np.nan

    last_model = None
    for model in ranked_ds['gcms']:
        if last_model is not None:
            weighted = ranked_ds.sel(gcms=model)
            last_weighted = ranked_ds.sel(gcms=last_model)
            p, lower_stats, upper_stats = ttost_paired(weighted, last_weighted, -epsilon,
                    epsilon)
            p_values[model] = p

            #print(weighted)
            #print(last_weighted)
            #print(p)
            #print(*lower_stats)
            #print(*upper_stats)
            #return
        last_model = model
    return p_values

def find_rank_diffs(df):
    df = df.reset_index()
    df.set_index(['num_metrics','gcms'], inplace=True)

    ranks_df = df['rank'].unstack('num_metrics')
    return ranks_df.diff(axis=1), None

def plot_diffs(df, models, out_dir=None, by='weights', region=0):
    plt.figure()
    sns.heatmap(df, cmap='vlag', yticklabels=models, vmin=-10, vmax=10)
    plt.xlabel('Number of Included Metrics')
    plt.ylabel('Global Climate Models')
    plt.title(f'Change in GCM rank with metric inclusion (ranked by {by}, for region {region})')
    plt.tight_layout()

    out_file = None
    if out_dir is not None:
        out_file = os.path.join(out_dir, f'rank_diffs_by_{by}_vs_num_metrics_region_{region}.pdf')
    save_or_show(out_file)

def plot_p_values(df, out_dir=None, by='weights', region=0, epsilon=.05):
    df = df.reset_index()
    df.set_index(['num_metrics','rank'], inplace=True)

    p_value_df = df['p_value'].unstack('num_metrics')
    print(p_value_df)

    plt.figure()
    sns.heatmap(p_value_df, cmap='vlag', vmin=0, vmax=.2)
    plt.xlabel('Number of Included Metrics')
    plt.ylabel('Rank')
    plt.title(f'P-value for paired t-test between subsequent ranks\n (ranked by {by}, for region {region})')
    plt.tight_layout()
    
    out_file = None
    if out_dir is not None:
        out_file = os.path.join(out_dir, f'rank_diff_p_values_by_{by}_vs_num_metrics_region_{region}.pdf')
    save_or_show(out_file)

def main():
    args = parse_args()
    in_file = os.path.realpath(args.in_file)
    in_dir = os.path.dirname(os.path.abspath(in_file))
    models_file = args.models_file
    out_dir = args.out_dir
    epsilon = args.epsilon
    
    var = 'weighted_data'
    if args.use_unweighted:
        var = 'unweighted_data'

    ds = xr.open_dataset(in_file, mode='r')
    ds['std'] = ds[var].std( 'gcms')

    for by in ['weights', 'std']:
        for region in range(ds['regions'].shape[0]):
            dfs = []
            for num_metrics in range(1, ds['metrics'].shape[0] + 1):
                filtered_ds = select_metrics(ds, by=by, var=var, region=region,
                        num_to_select=num_metrics)
                if num_metrics == ds['metrics'].shape[0]:
                    print(filtered_ds)
                ranked_df = rank_models(filtered_ds, var=var)
                
                ranked_df['num_metrics'] = num_metrics
                
                # Shouldn't run paired T-test with zero degrees of freedom
                if num_metrics == 1:
                    ranked_df['p_value'] = np.nan
                else:
                    ranked_df['p_value'] = run_t_test(ranked_df, filtered_ds, epsilon=epsilon,
                            var=var)
                dfs.append(ranked_df)

            df = pd.concat(dfs)
            rank_diffs, p_vals = find_rank_diffs(df)
            models = read_lines(models_file)
            plot_diffs(rank_diffs, models, out_dir=out_dir, by=by, region=region)
            plot_p_values(df, out_dir=out_dir, by=by, region=region, epsilon=epsilon)

            if out_dir is not None:
                rank_diffs_df = pd.DataFrame({'rank_diff': rank_diffs.stack('num_metrics')})
                rank_diffs_ds = rank_diffs_df.to_xarray()
                out_file = os.path.join(out_dir, f'rank_diffs_by_{by}_region_{region}.nc')
                rank_diffs_ds.to_netcdf(out_file, format='NETCDF4_CLASSIC')

                df.sort_values(['num_metrics','rank'], inplace=True)
                out_file = os.path.join(out_dir, f'ranked_by_{by}_region_{region}.csv')
                df[['num_metrics','rank','weighted_data', 'p_value']].to_csv(out_file)

if __name__ == '__main__':
    main()
