#!/usr/bin/env python3

# Based on example from
# /ccs/home/jkitson/.conda/envs/python-netcdf/lib/python3.10/site-packages/netCDF4/__init__.py

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import save_or_show, read_lines, write_lines
from statsmodels.stats.weightstats import ttost_paired, DescrStatsW
from pprint import pprint

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--in-file',
            default=os.path.join('..','data','data_cor_ss_ww.nc'),
            help='The path to the input file')
    parser.add_argument('-m', '--models-file',
            default=os.path.join('..','data','gcms_names.txt'),
            help='The path to a file containing the names of the GCMS')
    parser.add_argument('-M', '--metrics-file',
            default=os.path.join('..','data','metrics_list.txt'),
            help='The path to a file containing the names of the metrics')
    parser.add_argument('-o', '--out-dir', default=None,
            help='The optional path to a directory to save the output file')
    parser.add_argument('-w', '--weights-var', default='weights',
            help='The name of the variable used as weights for the weighted data')
    parser.add_argument('-u', '--use-unweighted', action='store_true',
            help='Pass this flag to use the unweighted data')
    #parser.add_argument('-n', '--num-metrics', type=int, nargs='+', default=[10],
    #        help='The number of metrics to retain (the n metrics with the heaviest weights ' + \
    #             'will be retained)')

    return parser.parse_args()

def select_metrics(ds, by='weights', var='weighted_data', num_to_select=10, region=0):
    df = ds.isel(regions=region)[[var, by]] \
            .to_dataframe()
  
    # The variable we sort by shouldn't depend on the models, so when we sort by it here,
    # the ordering will be the same for each metric when we perform the groupby (which
    # preserves the with-in group order)
    filtered_df = df.sort_values(by)
    
    # No need to do anything more than sort if we're keeping all the vlaues
    if num_to_select < ds['metrics'].shape[0]:
        filtered_df = filtered_df.groupby(level=['gcms']) \
            .apply(pd.DataFrame.tail, n=num_to_select) \
            .droplevel(level=0)

    return filtered_df

def rank_models(ds, var='weighted_data'):
    scores = ds[var].mean(['metrics'])
    scores = scores.sortby(scores, ascending=False)
    
    ranked_df = scores.to_dataframe()
    ranked_df['rank'] = range(ranked_df.shape[0])
    ranked_df['score'] = scores

    return ranked_df

def save_metrics_order(df, metrics, out_dir=None, by='weights', region=0, var=''):
    # It's hard to extract the first part of the index cleanly without distrubing the order,
    # so we group and sort to make sure we get it right
    sorted_df = df.droplevel('gcms') \
            .groupby('metrics') \
            .mean() \
            .sort_values(by, ascending=False)
    sorted_df['name'] = [metrics[i] for i in sorted_df.index]

    if out_dir is not None:
        out_file = f'{var}_sorted_metrics_by_{by}_region_{region}.csv'
        out_file = os.path.join(out_dir, out_file)
        sorted_df[['name', by]].to_csv(out_file)

def save_models_order(sorted_df, models, out_dir=None, by='weights', region=0, var=''):
    sorted_ids = sorted_df.index
    sorted_df = sorted_df.copy()
    sorted_df['name'] = [models[i] for i in sorted_df.index]

    if out_dir is not None:
        out_file = f'{var}_sorted_models_by_{by}_region_{region}.csv'
        out_file = os.path.join(out_dir, out_file)
        sorted_df[['name', 'score']].to_csv(out_file)

def to_array_like(df, values, cols='num_metrics', rows='gcms'):
    df = df.reset_index()
    df.set_index([cols, rows], inplace=True)
    return df[values].unstack(cols)

def find_relative(df, var, col=-1):
    arr_df = to_array_like(df, var)
    tmp = arr_df.T - arr_df.iloc[:,col].T
    return tmp.T

def find_diffs(df, var):
    arr_df = to_array_like(df, var)
    return arr_df.diff(axis=1)

def run_t_test(ranked_df, ds, var='weighted_data'):
    # Ranked_df has the model order, which ds needs
    ranked_ds = ds.sel(gcms=ranked_df.index)[var]
    
    p_values = np.zeros(ranked_ds['gcms'].shape)
    p_values[:] = np.nan

    last_model = None
    for rank, model in enumerate(ranked_ds['gcms']):
        if last_model is not None:
            weighted = ranked_ds.sel(gcms=model)
            last_weighted = ranked_ds.sel(gcms=last_model)
            descr = DescrStatsW(weighted - last_weighted)
            t, p, df = descr.ttest_mean()
            p_values[rank] = p

        last_model = model
    return p_values

def plot_relative_ranks(df, models, out_dir=None, by='weights', region=0, var=''):
    plt.figure()
    sns.heatmap(df, cmap='vlag', yticklabels=models) #, vmin=-10, vmax=10)
    plt.xlabel('Number of Included Metrics')
    plt.ylabel('Global Climate Models')
    plt.title(f'Difference between current and final GCM rank\n(ranked by {by}, for region {region})')
    plt.tight_layout()

    out_file = None
    if out_dir is not None:
        out_file = os.path.join(out_dir, f'{var}_relative_rank_by_{by}_vs_num_metrics_region_{region}.pdf')
    save_or_show(out_file)
    plt.close()

def plot_diffs(df, models, out_dir=None, by='weights', region=0, var=''):
    plt.figure()
    sns.heatmap(df, cmap='vlag', yticklabels=models, vmin=-10, vmax=10)
    plt.xlabel('Number of Included Metrics')
    plt.ylabel('Global Climate Models')
    plt.title(f'Change in GCM rank with metric inclusion\n(ranked by {by}, for region {region})')
    plt.tight_layout()

    out_file = None
    if out_dir is not None:
        out_file = os.path.join(out_dir, f'{var}_rank_diffs_by_{by}_vs_num_metrics_region_{region}.pdf')
    save_or_show(out_file)
    plt.close()

def plot_scores(df, models, out_dir=None, by='weights', region=0, var=''):
    plt.figure()
    sns.heatmap(df, cmap='rocket', yticklabels=models)
    plt.xlabel('Number of Included Metrics')
    plt.ylabel('Weighted Score')
    plt.title(f'GCM scores for top ranked metrics \n(ranked by {by}, for region {region})')
    plt.tight_layout()
    
    out_file = None
    if out_dir is not None:
        out_file = os.path.join(out_dir, f'{var}_rank_scores_by_{by}_vs_num_metrics_region_{region}.pdf')
    save_or_show(out_file)
    plt.close()

def plot_p_values(df, out_dir=None, by='weights', region=0, var=''):
    plt.figure()
    sns.heatmap(df, cmap='vlag', vmin=0, vmax=1)
    plt.xlabel('Number of Included Metrics')
    plt.ylabel('Rank')
    plt.title(f'P-value for paired t-test between subsequent ranks\n(ranked by {by}, for region {region})')
    plt.tight_layout()
    
    out_file = None
    if out_dir is not None:
        out_file = os.path.join(out_dir, f'{var}_rank_diff_p_values_by_{by}_vs_num_metrics_region_{region}.pdf')
    save_or_show(out_file)
    plt.close()

def main():
    args = parse_args()
    in_file = os.path.realpath(args.in_file)
    in_dir = os.path.dirname(os.path.abspath(in_file))
    models_file = args.models_file
    metrics_file = args.metrics_file
    out_dir = args.out_dir
    
    var = 'weighted_data'
    if args.use_unweighted:
        var = 'unweighted_data'

    ds = xr.open_dataset(in_file, mode='r')
    if args.weights_var != 'weights':
        ds = ds.rename({args.weights_var: 'weights'})

    #print(ds['weights'])
    ds['std'] = ds['unweighted_data'].std( 'gcms')
    #print(ds['std'])
    ds['weighted_std'] = ds['weights'] * ds['std']
    #print(ds['weighted_std'])
            
    models = read_lines(models_file)
    metrics = read_lines(metrics_file)
    total_metrics = ds['metrics'].shape[0]

    for by in ['weighted_std']: #, 'weights', 'std']:
        for region in range(ds['regions'].shape[0]):
            dfs = []
            for num_metrics in range(1, total_metrics + 1):
                filtered_df = select_metrics(ds, by=by, var=var, region=region,
                        num_to_select=num_metrics)
                filtered_ds = filtered_df.to_xarray()
                ranked_df = rank_models(filtered_ds, var=var)
                
                ranked_df['num_metrics'] = num_metrics
                
                # Shouldn't run paired T-test with zero degrees of freedom
                if num_metrics == 1:
                    ranked_df['p_value'] = np.nan
                else:
                    ranked_df['p_value'] = run_t_test(ranked_df, filtered_ds, var=var)
                
                if num_metrics == total_metrics:
                    save_metrics_order(filtered_df, metrics, out_dir=out_dir, by=by, 
                            region=region, var=var)
                    save_models_order(ranked_df, models, out_dir=out_dir, by=by, 
                            region=region, var=var)
                
                dfs.append(ranked_df)

            df = pd.concat(dfs)
            
            relative_ranks = find_relative(df, 'rank')
            plot_relative_ranks(relative_ranks, models, out_dir=out_dir, by=by, region=region,
                    var=var)
            
            rank_diffs = find_diffs(df, 'rank')
            plot_diffs(rank_diffs, models, out_dir=out_dir, by=by, region=region, var=var)
            
            scores = to_array_like(df, 'score')
            plot_scores(scores, models, out_dir=out_dir, by=by, region=region, var=var)
            
            p_value_df = to_array_like(df, 'p_value', rows='rank')
            plot_p_values(p_value_df, out_dir=out_dir, by=by, region=region, var=var)

            if out_dir is not None:
                relative_ranks_df = pd.DataFrame({'rank_diff': relative_ranks.stack('num_metrics')})
                relative_ranks_ds = relative_ranks_df.to_xarray()
                out_file = os.path.join(out_dir, f'{var}_relative_ranks_by_{by}_region_{region}.nc')
                relative_ranks_ds.to_netcdf(out_file, format='NETCDF4_CLASSIC')
                
                rank_diffs_df = pd.DataFrame({'rank_diff': rank_diffs.stack('num_metrics')})
                rank_diffs_ds = rank_diffs_df.to_xarray()
                out_file = os.path.join(out_dir, f'{var}_rank_diffs_by_{by}_region_{region}.nc')
                rank_diffs_ds.to_netcdf(out_file, format='NETCDF4_CLASSIC')
                
                #score_ds = score_df.stack('num_metrics').to_xarray()
                #out_file = os.path.join(out_dir, f'scores_by_{by}_region_{region}.nc')
                #score_ds.to_netcdf(out_file, format='NETCDF4_CLASSIC')
                
                p_value_ds = p_value_df.stack('num_metrics').to_xarray()
                out_file = os.path.join(out_dir, f'{var}_p_values_by_{by}_region_{region}.nc')
                p_value_ds.to_netcdf(out_file, format='NETCDF4_CLASSIC')

                df.sort_values(['num_metrics','rank'], inplace=True)
                out_file = os.path.join(out_dir, f'{var}_ranked_by_{by}_region_{region}.csv')
                df[['num_metrics','rank',var, 'p_value']].to_csv(out_file)

if __name__ == '__main__':
    main()
