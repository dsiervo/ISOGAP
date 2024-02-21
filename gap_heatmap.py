#!/home/dsiervo/miniconda3/bin/python
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
from azim_gap import azim_gap
import glob
from datetime import datetime
import matplotlib as mpl
import os
import click
import sys
import pickle

# append the path of the folder where the module is located
sys.path.append('/home/seiscomp/py_seiscomp2/')
from get_eq_data_sql_bna import EqData
import json
from icecream import ic

@click.group()
def main():
    pass

@main.command()
@click.option('-sd', "--stations_dir", required=True, default='default',
              prompt="Stations directory. Press enter if you don't have one",
              help='Directory containing the csv files with the stations coordinates, "no" if you want to use the default ones')
@click.option('-gs', "--grid_step", required=True,
              default=0.25, type=float, prompt=True, help='Grid step in degrees')
@click.option('-c', "--custom_quad", is_flag=True, prompt=True, help='Choose if you want to change de default quadrant: lats = -3,14 and lons = -80,-67')
@click.option('-pm', "--pool_mode", prompt=True, default="avg", type=click.Choice(['avg', 'max', 'min'], 
              case_sensitive=False), help='How to aggregate the data, averagin, maximum or minimum')
@click.option('-s', "--show", is_flag=True, prompt=True, help='Define if the maps will be show while they are being created')
@click.option('-gd', "--grids_dir", required=False, default='grids', help='Grids directory')
@click.option('-lo', "--lons", required=False, prompt=True, default='-80,-67', help='Longitude range like "-80,-67"')
@click.option('-la', "--lats", required=False, prompt=True, default='-3,14', help='Latitude range like "-3,14"')
@click.option('-dth', "--dist_thr", required=False, prompt="Distance threshold in km to consider a station",
              default=100, help='Distance threshold in km to consider a station')
@click.option('-gf', "--grids_file", required=True, prompt="File with the grids coordinates to make the heatmaps",
              help='File with the grids coordinates to make the heatmaps', type=click.Path(exists=True, readable=True),
              default='region_coordinates.json')
def g_h(stations_dir, grid_step, custom_quad, grids_dir, pool_mode, show, lons, lats, dist_thr, grids_file):
    # computing
    azim_gap(stations_dir, grid_step, custom_quad, grids_dir, lons, lats, dist_thr)
    #grids_dir = 'grids'
    make_heatmaps(grids_dir, pool_mode, show, lons, lats, grids_file, stations_dir)


@main.command()
@click.option('-sd', "--stations_dir", required=True, default='default',
              prompt="Stations directory. Press enter if you don't have one",
              help='Directory containing the csv files with the stations coordinates, "default" if you want to use the default ones')
@click.option('-lo', "--lons", required=False, prompt=True, default='-80,-67', help='Longitude range like "-80,-67"')
@click.option('-la', "--lats", required=False, prompt=True, default='-3,14', help='Latitude range like "-3,14"')
@click.option('-gd', "--grids_dir", required=False, default='grids', help='Grids directory')
@click.option('-gs', "--grid_step", required=True,
              default=0.25, type=float, prompt=True, help='Grid step in degrees')
@click.option('-c', "--custom_quad", is_flag=True, prompt=True, help='Choose if you want to change de default quadrant: lats = -3,14 and lons = -80,-67')
@click.option('-dth', "--dist_thr", required=False, prompt=True, default=100, help='Distance threshold in km to consider a station')
def grids(stations_dir, grid_step, custom_quad, grids_dir, lons, lat):
    azim_gap(stations_dir, grid_step, custom_quad, grids_dir, lons, lat)


@main.command()
@click.option('-gd', "--grids_dir", required=True, prompt=True, help='Grids directory')
@click.option('-pm', "--pool_mode", prompt=True, default="avg", type=click.Choice(['avg', 'max', 'min'], 
              case_sensitive=False), help='How to aggregate the data, averagin, maximum or minimum')
@click.option('-s', "--show", is_flag=True, prompt=True, help='Define if the maps will be show while they are being created')
@click.option('-lo', "--lons", required=False, prompt=True, default='-80,-67', help='Longitude range like "-80,-67"')
@click.option('-la', "--lats", required=False, prompt=True, default='-3,14', help='Latitude range like "-3,14"')
@click.option('-gf', "--region_coords", required=True, prompt="File with the grids coordinates to make the heatmaps",
              help='File with the grids coordinates to make the heatmaps', type=click.Path(exists=True, readable=True),
              default='region_coordinates.json')
@click.option('-sd', "--stations_dir", required=True, default=None,
              prompt="Directory containing csv files with stations coordinates, to plot them. Press enter if you don't want to plot the stations",
              help='Directory containing the csv files with the stations coordinates to plot them')
def heatmaps(grids_dir, pool_mode, show, lons, lats, region_coords, stations_dir):
    make_heatmaps(grids_dir, pool_mode, show, lons, lats, region_coords, stations_dir)


def make_heatmaps(grids_dir, pool_mode, show, lons, lats, region_coords, stations_dir):

    output_dir = 'output_maps'
    # reading region coordinates
    with open(region_coords, 'r') as f:
        grids_ = json.load(f)

    # Remove the comment key
    grids = {k: v for k, v in grids_.items() if not k.startswith('_comment')}
    # make sure grids directory exist
    if not os.path.exists(grids_dir):
        sta_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'station_coordinates')
        message = f'''
        ¡No existe el directorio "{grids_dir}" con las grillas de gap teórico!
        
        Si no tiene grillas de gap teórico puede generarlas con los archivos
        de coordenadas de estaciones por defecto, que se encuentran en la ruta
        {sta_dir}
        y hacer los respectivos heatmaps dando "enter" con el comando:
        
        gap_heatmap.py g-h
        
        '''
        print(message)
        sys.exit()

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
            ic(i)
            splited_file_name = os.path.basename(i).split('_')
            print(splited_file_name)
            sep = float(splited_file_name[-2])
            file_prefix = '_'.join(splited_file_name[:-2])
            print('\n\tfile_prefix:', file_prefix, '- grid:', sep)
            df = pd.read_csv(i)
            # only plot contours once
            if c == 0:
                iso_gap_map(df, file_prefix, lons, lats, sep, main_dir=output_dir, plot=show, stations_dir=stations_dir)

            map_and_grids(df, file_prefix, lons, lats, grid, sep,
                          pool_mode=pool_mode,
                          output_dir=sub_output_dir,
                          plot=show, stations_dir=stations_dir)
        c += 1

    print('\n\n\tArchivos de salida en la carpeta: %s\n'%output_dir)


"""def iso_gap_map(df, file_prefix, lons, lats, main_dir='mapas', plot=False):
    lon_bins = np.arange(df['LON'].min(), df['LON'].max(), 0.25) 
    lat_bins = np.arange(df['LAT'].min(), df['LAT'].max(), 0.25) 

    X, Y = np.meshgrid(lon_bins, lat_bins)
    gap_grid = griddata((df['LON'], df['LAT']),
                        df['GAP'], (X, Y), method='linear')
    fig = plt.figure(figsize=(30,30))

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    #ax.set_extent([-82, -65, -2, 14])
    ax.set_extent([float(lons.split(',')[0]), float(lons.split(',')[1]),
                   float(lats.split(',')[0]), float(lats.split(',')[1])])

    ax.contour(X, Y, gap_grid, colors='black', linewidths=0.5,
                transform=ccrs.PlateCarree())
    g = ax.contourf(X, Y, gap_grid,
                transform=ccrs.PlateCarree())

    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # Add US states
    ax.add_feature(cfeature.STATES.with_scale('10m'))

    # add colorbar
    cbar = fig.colorbar(g, orientation='horizontal', shrink=0.625, aspect=20,
                        fraction=0.2, pad=0.05)
    cbar.set_label('GAP',size=14)

    output_dir = os.path.join(main_dir, 'contours')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    output_path = os.path.join(output_dir, 'contour_%s'%file_prefix)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()
    fig.clf()"""

def iso_gap_map(df, file_prefix, lons, lats, sep, main_dir='mapas', plot=False, stations_dir=None):


    output_path = os.path.join(main_dir, 'contour_%s_%s.png'%(file_prefix, sep))
    pickle_path = output_path.replace('.png', '.pkl')
    if os.path.exists(pickle_path):
        # Ask if the user wants to load the figure from the pickle file or recalculate it
        answer = input(f'File {pickle_path} already exists. Do you want to load the figure from the pickle file? (y/n): ')
        if answer.lower() == 'y':
            print('Loading figure from pickle file...')
            with open(pickle_path, 'rb') as f:
                fig = pickle.load(f)
            plt.show()
            return None
        else:
            print('Recalculating figure...')
        
    # Get earthquake data
    print('Getting earthquake data...')
    # get current time in UTC
    endtime = datetime.utcnow()
    # get the last 12 months
    starttime = endtime - pd.Timedelta(360, unit='D')
    
    # Columns:['Eventid', 'OriginDate', 'OriginTime', 'lat', 'lon', 'depth', 'depth_unc', 'magnitude', 'mag_unc', 'min_sta_dis', 'sec_min_sta_dis', 'az_gap', 'n_picks_p', 'n_picks_s', 'station_count', 'region']
    df_eq = EqData(
                #bna_folder='/home/seiscomp/fixed_BNA20230328',
                bna_folder='/home/seiscomp/.seiscomp3/bna/del',
                starttime=starttime.strftime('%Y-%m-%d %H:%M:%S'),
                endtime=endtime.strftime('%Y-%m-%d %H:%M:%S'),
                status="'preliminary', 'final'",
                overwrite=False,
                status_in_output_filename=True,
                host='scdb.beg.utexas.edu',
                user='sysro', passwd='0niReady',
                database='seiscomp').get_df_eqs()
    
    lon_bins = np.arange(df['LON'].min(), df['LON'].max(), 0.25) 
    lat_bins = np.arange(df['LAT'].min(), df['LAT'].max(), 0.25) 

    X, Y = np.meshgrid(lon_bins, lat_bins)
    gap_grid = griddata((df['LON'], df['LAT']),
                        df['GAP'], (X, Y), method='linear')
    fig = plt.figure(figsize=(30,30))

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent([float(lons.split(',')[0]), float(lons.split(',')[1]),
                   float(lats.split(',')[0]), float(lats.split(',')[1])])

    if stations_dir:
        df_stations = pd.read_csv(os.path.join(stations_dir, file_prefix+'.csv'))
        ic(df_stations.columns)
        # Rename 'Longitude (WGS84)', 'Latitude (WGS84)' and 'Station Code' to 'lon', 'lat' and 'sta'
        df_stations.rename(columns={'Longitude (WGS84)': 'lon', 'Latitude (WGS84)': 'lat', 'Station Code': 'sta'}, inplace=True)
        # keep only the stations that are between lats and lons
        df_stations = df_stations[(df_stations['lon'] >= float(lons.split(',')[0])) &
                                    (df_stations['lon'] <= float(lons.split(',')[1])) &
                                    (df_stations['lat'] >= float(lats.split(',')[0])) &
                                    (df_stations['lat'] <= float(lats.split(',')[1]))]
        # Add stations as triangles and station names
        ax.scatter(df_stations['lon'], df_stations['lat'], marker='^', color='blue', transform=ccrs.PlateCarree())
        for i, row in df_stations.iterrows():
            ax.text(row['lon'], row['lat'], row['sta'], transform=ccrs.PlateCarree(), fontsize=8)

    # Add earthquakes as circles with opacity of 0.4
    # filter the earthquakes that are between Lons and Lats
    df_eq = df_eq[(df_eq['lon'] >= float(lons.split(',')[0])) &
                    (df_eq['lon'] <= float(lons.split(',')[1])) &
                    (df_eq['lat'] >= float(lats.split(',')[0])) &
                    (df_eq['lat'] <= float(lats.split(',')[1]))]
    ax.scatter(df_eq['lon'], df_eq['lat'], s=10, color='black', alpha=0.8, transform=ccrs.PlateCarree())

    #ax.contour(X, Y, gap_grid, colors='black', linewidths=0.5, transform=ccrs.PlateCarree())
    levels = [0, 60, 90, 120, 360]
    #cmap = mpl.cm.get_cmap('viridis_r', len(levels) - 1)
    #cmap = mpl.cm.viridis_r
    cmap = mpl.cm.Dark2_r
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    g = ax.contourf(X, Y, gap_grid, alpha=0.8, levels=levels, cmap=cmap, norm=norm,
                transform=ccrs.PlateCarree())
    #g = ax.contour(X, Y, gap_grid, levels=levels, transform=ccrs.PlateCarree(), colors=('tab:blue', 'tab:green', 'tab:orange', 'tab:red'))
    #ax.clabel(g, inline=True, fontsize=10, fmt='%d')

    ax.coastlines(resolution='50m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES.with_scale('10m'))

    #cbar = fig.colorbar(g, orientation='horizontal', shrink=0.5, aspect=20,
    #                    fraction=0.15)#, pad=0.05)
    #cbar.set_label('GAP',size=14)
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(fig, f)
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


def map_and_grids(df, file_prefix, lons, lats, cuadrants, sep=0.5,
                  pool_mode='avg', output_dir='mapas/heatmaps/', plot=False, stations_dir=None):

    ic(file_prefix)
    grids_list = []
    d = 0.5  # distance to put correctly numbers on heatmap plot
    fig = plt.figure(figsize=(25, 25), clear=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    #ax.set_extent([-82, -65, -2, 14])
    ax.set_extent([float(lons.split(',')[0]), float(lons.split(',')[1]),
                   float(lats.split(',')[0]), float(lats.split(',')[1])])
    

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
        # Debido a que matplotlib a partir de la versión 3.3 no permite usar shading='flat'
        # cuando C (gap_grid) tiene la misma dimensión de X y Y, para que no coloque
        # shading = 'nearest' se debe poner explícitamente este atributo y quitarle una dimensión
        # a C en cada eje. Se escoje quitar la última dimensión en cada eje debido a que por
        # defecto toma el valor de la esquina superior izquierda.
        gap_plot = ax.pcolormesh(X, Y, gap_grid[:-1, :-1], shading='flat',
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
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray')
    ax.add_feature(cfeature.BORDERS.with_scale('10m'))
    # ax.stock_img()
    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='10m',
        facecolor='none')
    ax.add_feature(states_provinces)
    #ax.set_xticks(range(-82, -65), crs=ccrs.PlateCarree())
    lons_int = [int(round(float(i))) for i in lons.split(',')]
    lat_int = [int(round(float(i))) for i in lats.split(',')]
    
    ax.set_xticks(range(lons_int[0], lons_int[1]), crs=ccrs.PlateCarree())
    #ax.set_yticks(range(-2, 15), crs=ccrs.PlateCarree())
    ax.set_yticks(range(lat_int[0], lat_int[1]), crs=ccrs.PlateCarree())

    if stations_dir:
        df_stations = pd.read_csv(os.path.join(stations_dir, file_prefix+'.csv'))
        # Rename 'Longitude (WGS84)', 'Latitude (WGS84)' and 'Station Code' to 'lon', 'lat' and 'sta'
        df_stations.rename(columns={'Longitude (WGS84)': 'lon', 'Latitude (WGS84)': 'lat', 'Station Code': 'sta'}, inplace=True)
        # keep only the stations that are between lats and lons
        df_stations = df_stations[(df_stations['lon'] >= float(lons.split(',')[0])) &
                                    (df_stations['lon'] <= float(lons.split(',')[1])) &
                                    (df_stations['lat'] >= float(lats.split(',')[0])) &
                                    (df_stations['lat'] <= float(lats.split(',')[1]))]
        print(df_stations.head())
        # Add stations as triangles and station names
        ax.scatter(df_stations['lon'], df_stations['lat'], marker='^', color='red', transform=ccrs.PlateCarree())
        for i, row in df_stations.iterrows():
            ax.text(row['lon'], row['lat'], row['sta'], transform=ccrs.PlateCarree())

    mean, std = get_mean_st(grids_list)
    print('\n\tGAP promedio %s: %.1f ± %.1f'%(file_prefix, mean, std))

    #plt.figtext(0.5, 0.87, 'GAP promedio %s: $%.1f \\pm %.1f$' % (file_prefix, mean, std))

    #output_dir2 = os.path.join(output_dir, 'heatmaps')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, f'heatmap_%s_%s.png' % (file_prefix, sep))
    plt.savefig(output_path, bbox_inches='tight',
                pad_inches=0)
    if plot:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close('all')


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
