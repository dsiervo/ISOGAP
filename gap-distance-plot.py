#!/home/seiscomp/anaconda3/envs/gapheatmap/bin/python
# -*- coding: utf-8 -*-
"""
Author: Daniel Siervo, 2024
"""
import sys
# append the path of the folder where the module is located
sys.path.append('/home/seiscomp/py_seiscomp2/')
import matplotlib.pyplot as plt
import os
import numpy as np
from distance_countour_map import plot_distance_countour_map
import pandas as pd
from icecream import ic
from scipy.interpolate import griddata
import cartopy.crs as ccrs
from azim_gap import azim_gap
import pickle
import matplotlib.patches as patches
import re

def plot_gap_distance(station_file_path, lons, lats, grid_step, dist_threshold, starttime, endtime, overwrite_grid=False, grid_dir='test_texnet_grids'):
    """
    From a list of stations given, creates a rectangular grid between lons and lats and calculates
    how many stations are within a certain distance threshold from each point of the grid.

    Later, it plots a contour map with the number of stations within the distance threshold.
    
    Parameters
    ----------
    station_file : str
        Path to the file with the stations. The file should be a csv file with the following columns:
        Code,Station Code,Longitude (WGS84),Latitude (WGS84),Affiliation,Archive,Location Description,Place,Elevation,Start Date,End Date
    lons : str
        Comma separated list of the corners of the rectangle to create the grided area. Like: lon1,lon2
    lats : str
        Comma separated list of the corners of the rectangle to create the grided area. Like: lat1,lat2
    grid_step : float
        Step for the grid in degrees.
    dist_threshold : float
        Distance threshold in kilometers.
    starttime : str
        Start time for the earthquakes. Format: 'YYYY-MM-DD HH:MM:SS'
    endtime : str
        End time for the earthquakes. Format: 'YYYY-MM-DD HH:MM:SS'
    overwrite_grid : bool
        If True, the grid will be recalculated and saved to a csv file. If False, the grid will be read from a csv file if it exists.
    """
    output_dir = 'gap_distance_map'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_prefix = os.path.basename(station_file_path).split('.')[0]
    ic(file_prefix)
    output_path = os.path.join(output_dir, f'gap_distance_map_{file_prefix}_{starttime.replace(" ", "_")}-{endtime.replace(" ", "_")}-{dist_threshold}km.png')
    pickle_path = output_path.replace('.png', '.pickle')
    ic(pickle_path)
    if os.path.exists(pickle_path):
        # Ask if the user wants to load the figure from the pickle file or recalculate it
        answer = input(f'Figure {pickle_path} already exists. Do you want to load it? (y/n) ')
        if answer.lower() == 'y':
            print(f'Loading figure from {pickle_path}...')
            with open(pickle_path, 'rb') as f:
                fig = pickle.load(f)
            plt.show()
            return fig
    print("Recalculating figure...")
    grid_file_path = os.path.join(grid_dir, f'{file_prefix}_{grid_step}_grid.csv')
    ic(grid_file_path)
    
    fig = plot_distance_countour_map(station_file_path, lons, lats, grid_step, dist_threshold, starttime, endtime, overwrite_grid, ask_to_load=False)
    
    ax = fig.get_axes()[0]
    
    # check if grid file exists
    if not os.path.exists(grid_file_path) or overwrite_grid:
        ic('Calculating gap grid...')
        # get directory of the station file
        sta_dir = os.path.dirname(station_file_path)
        azim_gap(sta_dir, grid_step, custom=True, output_dir=output_dir, lons=lons, lats=lats, dist_thr=dist_threshold)
        
    X, Y, gap_grid = contour_gap_grid(grid_file_path)
    
    levels = [0, 60, 90, 120, 360]
    g = ax.contour(X, Y, gap_grid, levels=levels, transform=ccrs.PlateCarree(), colors=('tab:blue', 'tab:green', 'tab:orange', 'tab:red'))
    ax.clabel(g, inline=True, fontsize=10, fmt='%d')

    # Add polygons to the plot
    bna_file_path = '/home/seiscomp/.seiscomp3/bna/del/del.bna'
    add_polygon_to_plot(bna_file_path, ax)

    ax.legend(handles=g.legend_elements()[0][1:-1], labels=[f'GAP = {level}' for level in levels[1:-1]], loc='lower left', fancybox=True)
    
    fig.savefig(output_path)
    ic(f'Figure saved to {output_path}')
    with open(pickle_path, 'wb') as f:
        pickle.dump(fig, f)
    ic(f'Figure saved to {pickle_path}')
    plt.show()

def contour_gap_grid(grid_path):
    df = pd.read_csv(grid_path)
    lon_bins = np.arange(df['LON'].min(), df['LON'].max(), 0.25) 
    lat_bins = np.arange(df['LAT'].min(), df['LAT'].max(), 0.25) 

    X, Y = np.meshgrid(lon_bins, lat_bins)
    gap_grid = griddata((df['LON'], df['LAT']),
                        df['GAP'], (X, Y), method='linear')
    return X, Y, gap_grid

def add_polygon_to_plot(filename, ax):
    with open(filename, 'r') as file:
        header = file.readline().split(',')
        polygon_name = header[0].strip('"')
        num_points = int(header[2])
        
        points = []
        for _ in range(num_points):
            line = file.readline()
            points.append(list(map(float, line.split(','))))
        
        polygon = patches.Polygon(points, fill=None, label=polygon_name)
        ax.add_patch(polygon)
        plt.legend()
if __name__ == '__main__':
    # Example usage
    station_file = 'texnet_stations/texnet_stations_2024.csv'
    lons = "-106,-94"
    lats = "28.2,36.5"
    grid_step = 0.05
    dist_threshold = 5
    starttime = '2019-01-01 00:00:00'
    endtime = '2024-12-31 23:59:59'
    preload_distance_map = False
    plot_gap_distance(station_file, lons, lats, grid_step, dist_threshold, starttime, endtime)