#!/home/seiscomp/anaconda3/envs/gapheatmap/bin/python
# -*- coding: utf-8 -*-
"""
Author: Daniel Siervo, March 2024
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
import glob
import json
import datetime
import time
import logging
log_file_name = "log_" + os.path.basename(__file__).replace('.py', '.log')
logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format='%(message)s')

def loggin_print(s):
    logging.debug(s)
    print(s, file=sys.stderr)

#ic.configureOutput(prefix=time_ic_debug, outputFunction=loggin_print)

def plot_gap_distance(station_file_path, lons, lats, grid_step, dist_threshold, starttime, endtime, overwrite_grid=False, grid_dir='test_texnet_grids', polygons_dir=None, ask_to_load=False, polygons_kmz_dir=None, load_dis_pickle=True):
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
                print(f'Figure loaded from {pickle_path}')
                print(fig)
            plt.show()
            return fig
    print("Recalculating figure...")
    grid_file_path = os.path.join(grid_dir, f'{file_prefix}_{grid_step}_grid.csv')
    ic(grid_file_path)
    
    
    dist_time_start = time.time()
    print('Calculating distance map...')
    fig = plot_distance_countour_map(station_file_path, lons, lats, grid_step, dist_threshold, starttime, endtime, overwrite_grid, ask_to_load=ask_to_load, load_pickle=load_dis_pickle)
    dist_time_end = time.time()
    distance_time = dist_time_end - dist_time_start
    loggin_print(f'\nTime to calculate distance map: {distance_time} seconds\n')
    
    
    ax = fig.get_axes()[0]
    
    gap_time_start = time.time()
    # check if grid file for GAPS exists
    if not os.path.exists(grid_file_path) or overwrite_grid:
        ic('Calculating gap grid...')
        # get directory of the station file
        sta_dir = os.path.dirname(station_file_path)
        azim_gap(sta_dir, grid_step, custom=True, output_dir=output_dir, lons=lons, lats=lats, dist_thr=dist_threshold)
    gap_time_end = time.time()
    gap_time = gap_time_end - gap_time_start
        
    X, Y, gap_grid = contour_gap_grid(grid_file_path)
    
    levels = [0, 60, 90, 120, 360]
    g = ax.contour(X, Y, gap_grid, levels=levels, transform=ccrs.PlateCarree(), colors=('tab:blue', 'tab:green', 'tab:orange', 'tab:red'))
    ax.clabel(g, inline=True, fontsize=10, fmt='%d')
    loggin_print(f'\nTime to calculate gap map: {gap_time} seconds\n')
    # Add polygons to the plot
    #bna_file_path = '/home/seiscomp/.seiscomp3/bna/del/del.bna'
    #add_polygon_to_plot(bna_file_path, ax)

    # Add polygons to the plot
    if polygons_dir:
        add_polygon_from_esri_json(polygons_dir, ax)
    
    if polygons_kmz_dir:
        kmz_files = glob.glob(os.path.join(polygons_kmz_dir, '*.kmz'))
        for kmz_file in kmz_files:
            add_polygon_from_kmz(kmz_file, ax)

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

def add_polygon_from_esri_json(polygons_dir, ax):
    from pyproj import Proj, transform
    # Define the projection transformation. From Web Mercator (EPSG:3857) to WGS84 (EPSG:4326)
    proj_from = Proj(init='epsg:3857')
    proj_to = Proj(init='epsg:4326')
    
    geojson_polygons = glob.glob(os.path.join(polygons_dir, '*.json'))
    for polygon in geojson_polygons:
        ic(polygon)
        with open(polygon, 'r') as f:
            data = json.load(f)
        for ring in data['rings']:
            lon_orig, lat_orig = zip(*ring)
            # Transform the coordinates to WGS84
            lon, lat = transform(proj_from, proj_to, lon_orig, lat_orig)
            ax.plot(lon, lat, color='black', linewidth=1, transform=ccrs.PlateCarree())

def add_polygon_from_kmz(kmz_file, ax):
    """
    Add polygons to the plot from a KMZ file.
    """
    import zipfile
    import tempfile
    from pykml import parser

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Unzip the KMZ file
        with zipfile.ZipFile(kmz_file, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)

        # KMZ files usually contain a single KML file named "doc.kml"
        kml_file = os.path.join(tmpdirname, "doc.kml")

        with open(kml_file, 'r') as f:
            doc = parser.parse(f).getroot()

        for placemark in doc.Document.Placemark:
            if hasattr(placemark, 'Polygon'):
                coords_text = placemark.Polygon.outerBoundaryIs.LinearRing.coordinates.text
                coords = [tuple(map(float, coord.split(','))) for coord in coords_text.split()]
                lon, lat = zip(*[(coord[0], coord[1]) for coord in coords])  # Only take the first two values
                ax.plot(lon, lat, color='black', linewidth=1, transform=ccrs.PlateCarree())


def add_polygon_from_bna(filename, ax):
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
    lons = "-106,-93"
    lats = "28.2,36.5"
    grid_step = 0.05
    #grid_step = 0.5
    dist_threshold = 5
    starttime = '2019-01-01 00:00:00'
    endtime = '2024-12-31 23:59:59'
    #preload_distance_map = False
    #polygons_dir = '/home/seiscomp/ISOGAP/areas_add_stations'
    polygons_dir = None
    #polygons_kmz_dir = '/home/seiscomp/ISOGAP/kmz_polygons'd
    polygons_kmz_dir = None
    load_piclke_distance_map = False
    overwrite_grid = True
    grid_dir = 'gap_distance_map'
    fig = plot_gap_distance(station_file, lons, lats, grid_step, dist_threshold, starttime, endtime,
                            grid_dir=grid_dir, load_dis_pickle=load_piclke_distance_map,
                            polygons_dir=polygons_dir, overwrite_grid=overwrite_grid,
                            polygons_kmz_dir=polygons_kmz_dir)
    plt.plot()
