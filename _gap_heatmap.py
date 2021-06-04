#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v(1) 2021-05-28
autor: Daniel Siervo
e-mail: emetdan@gmail.com

Crea heatmap del máximo gap azimutal teórico usando la grilla de salida
del programa azim_gap.py nombrados como yyyy_sep_grid.csv (2020_0.5_grid.csv)
puede ser ejecutado sobre un directorio que contenga las grillas de gap.
"""

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import os
import click


@click.command()
@click.option('-g', "--grids_dir", required=True, prompt=True, help='Waveform file name')
@click.option('-o', "--output_dir", required=True, prompt=True, help='Dataless file name')
@click.option('-m', "--pool_mode", default="avg", help='units to convert. Can be: DIS, VEL, ACC')
@click.option('-p', "--plot", default=False, type=bool, help='Define if waveform will be plotted')


def main(grids_dir, output_dir, pool_mode, plot):
    
    grids = {}
    grids['big'] = [[3, 8, -77, -73], [7, 9, -76, -73], [9, 11, -75, -73],
                    [1, 3, -78, -74], [11, 12, -73, -72], [1, 2, -79, -78]]
    
    grids['center'] = [[3, 5, -77, -74], [5, 7, -77, -73], [7, 8, -76, -74]]
    
    # creating output file if doesn't exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    c = 0
    for region in grids:
        print('\n\t', 'region:', region)
        sub_output_dir = os.path.join(output_dir, region)
        # creating output file if doesn't exist
        if not os.path.exists(sub_output_dir):
            os.mkdir(sub_output_dir)

        grid = grids[region]
        # iterating over all azimutal gap files in folder gaps (generated with azim_gap.py)
        for i in glob.glob(os.path.join(grids_dir, '*grid.csv')):
            splited_file_name = os.path.basename(i).split('_')
            sep = float(splited_file_name[1])
            year = splited_file_name[0]
            print('\n\tyear:', year, '- grid:', sep)
            df = pd.read_csv(i)
            # only plot contours once
            if c == 0:
                iso_gap_map(df, year, main_dir=output_dir, plot=plot)
            
            map_and_grids(df, year, grid, sep,
                          pool_mode=pool_mode,
                          output_dir=sub_output_dir,
                          plot=plot)
        c += 1

    print('\n\tArchivos de salida en la carpeta: %s'%output_dir)


def iso_gap_map(df, year, main_dir='mapas', plot=False):
    lon_bins = np.arange(df['LON'].min(), df['LON'].max(), 0.25) 
    lat_bins = np.arange(df['LAT'].min(), df['LAT'].max(), 0.25) 

    X, Y = np.meshgrid(lon_bins, lat_bins)
    gap_grid = griddata((df['LON'], df['LAT']),
                        df['GAP'], (X, Y), method='linear')
    fig = plt.figure(figsize=(30,30))

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent([-82, -65, -2, 14])

    ax.contour(X, Y, gap_grid, colors='black', linewidths=0.5,
                transform=ccrs.PlateCarree())
    g = ax.contourf(X, Y, gap_grid,
                transform=ccrs.PlateCarree())

    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # add colorbar
    cbar = fig.colorbar(g, orientation='horizontal', shrink=0.625, aspect=20,
                        fraction=0.2, pad=0.05)
    cbar.set_label('GAP',size=14)

    output_dir = os.path.join(main_dir, 'contours')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    output_path = os.path.join(output_dir, 'contour_%s'%year)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()
    fig.clf()


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max', 'avg' or 'min'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'min':
        return A_w.min(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)


def make_grid(lat_min=2, lat_max=7, lon_min=-77,
              lon_max=-72, sep=1, df=None, pool_mode='avg'):

    f = round(1/sep)
    lon_bins_ = np.arange(lon_min, lon_max+1, sep)
    lat_bins_ = np.arange(lat_min, lat_max+1, sep)
    X_, Y_ = np.meshgrid(lon_bins_, lat_bins_)

    lon_bins = np.arange(lon_min, lon_max+1, 1)
    lat_bins = np.arange(lat_min, lat_max+1, 1)
    X, Y = np.meshgrid(lon_bins, lat_bins) # X, Y grid with 1 of sep (for the plot)

    gap_grid_ = griddata((df['LON'], df['LAT']),
                         df['GAP'], (X_, Y_), method='linear')

    gap_grid = pool2d(gap_grid_, kernel_size=f, stride=f, padding=0,
                      pool_mode=pool_mode)
    # gap_grid = np.pad(gap_grid, 1, mode='constant')[1:,1:]

    return X, Y, gap_grid


def get_mean_st(grids):
    l_flat = []
    for g in grids:
        l_flat += g[:-1, :-1].flatten().tolist()
    return np.mean(l_flat), np.std(l_flat)


def map_and_grids(df, year, cuadrants, sep=0.5,
                  pool_mode='avg', output_dir='mapas/heatmaps/', plot=False):

    grids_list = []
    d = 0.5  # distance to put correctly numbers on heatmap plot
    fig = plt.figure(figsize=(25, 25), clear=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-82, -65, -2, 14])

    for cuad in cuadrants:
        assert isinstance(cuad, list), "\n\t CUIDADO el cuadrante %s no es una lista\n"%cuad
        assert len(cuad) == 4, "\n\t CUIDADO el cuadrante %s no tiene 4 puntos.\n"%cuad

        lat_min, lat_max, lon_min, lon_max = cuad
        assert lat_min < lat_max and lon_min < lon_max

        # sep, separation between points on the grid
        X, Y, gap_grid = make_grid(lat_min, lat_max, lon_min, lon_max,
                                   sep, df, pool_mode)
        grids_list.append(gap_grid)

        # grid 1
        gap_plot = ax.pcolormesh(X, Y, gap_grid,
                                 transform=ccrs.PlateCarree(),
                                 vmin=df['GAP'].min(), vmax=df['GAP'].max())

        xn, yn = gap_grid.shape
        for i in range(xn-1):
            for j in range(yn-1):
                plt.text(X[i, j]+d, Y[i, j]+d, '%.1f' % gap_grid[i, j],
                         horizontalalignment='center', color='white',
                         verticalalignment='center')

    # add colorbar
    cbar = fig.colorbar(gap_plot, orientation='horizontal', shrink=0.625,
                        aspect=20, fraction=0.2, pad=0.05)
    cbar.set_label('GAP', size=14)

    ax.coastlines(resolution='10m')
    # ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    # ax.stock_img()
    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')
    ax.add_feature(states_provinces)
    ax.set_xticks(range(-82, -65), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-2, 15), crs=ccrs.PlateCarree())

    mean, std = get_mean_st(grids_list)

    plt.figtext(0.5, 0.87, 'GAP promedio %s: $%.1f \pm %.1f$'%(year, mean, std),
                fontsize=18, ha='center')

    output_dir2 = os.path.join(output_dir, 'heatmaps')
    if not os.path.exists(output_dir2):
        os.mkdir(output_dir2)
    output_path = os.path.join(output_dir2, 'heatmap_%s'%year)
    plt.savefig(output_path, bbox_inches='tight',
                pad_inches=0)
    if plot:
        plt.show()
    fig.clf()


if __name__ == '__main__':
    
    main()
    
    """params = read_params('params_gap_heatmap.inp')
    
    data_dir = params['data_dir']
    output_dir = params['output_dir']
    
    grid = params['grid']
    
    #grid_all = [[3, 8, -77, -73], [7, 9, -76, -73], [9, 11, -75, -73],
    #            [1, 3, -78, -74], [11, 12, -73, -72], [1, 2, -79, -78]]
    
    #grid_center = [[3, 5, -77, -74], [5, 7, -77, -73], [7, 8, -76, -74]]
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # iterating over all azimutal gap files in folder gaps (generated with azim_gap.py)
    for i in glob.glob(os.path.join(data_dir, '*grid.csv')):
        splited_file_name = os.path.basename(i).split('_')
        sep = float(splited_file_name[1])
        year = splited_file_name[0]
        print('\n\t', year, ', grid:', sep)
        df = pd.read_csv(i)
        iso_gap_map(df, year, main_dir=output_dir)
        map_and_grids(df, year, grid_center, sep,
                      pool_mode='avg', output_dir=output_dir)"""
