#!/home/dsiervo/miniconda3/bin/python
# -*- coding: utf-8 -*-
"""
Author: Daniel Siervo
"""
import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import pickle
from icecream import ic
ic.disable()
# append the path of the folder where the module is located
sys.path.append('/home/seiscomp/py_seiscomp2/')
from get_eq_data_sql_bna import EqData
import os


def plot_distance_countour_map(station_file, lons, lats, grid_step, dist_threshold, starttime, endtime, overwrite_grid=False, ask_to_load=True, load_pickle=True):
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
    output_dir = 'output_dist_contour_maps'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'station_density_map_{starttime.replace(" ", "_")}-{endtime.replace(" ", "_")}-{dist_threshold}km.png')
    pickle_path = output_path.replace('.png', '.pickle')
    if os.path.exists(pickle_path):
        if ask_to_load:
            # Ask if the user wants to load the figure from the pickle file or recalculate it
            answer = input(f'Figure {pickle_path} already exists. Do you want to load it instead of recalculating it? (y/n) ')
            plot = True if answer.lower() == 'y' else False
        else:
            plot = False
            answer = 'y' if load_pickle else 'n'
        if answer.lower() == 'y':
            print(f'Loading figure from {pickle_path}...')
            with open(pickle_path, 'rb') as f:
                fig = pickle.load(f)
            if plot:
                plt.show()
            return fig
        else:
            print(f'Recalculating figure and saving it to {pickle_path}...')
    # Columns:['Eventid', 'OriginDate', 'OriginTime', 'lat', 'lon', 'depth', 'depth_unc', 'magnitude', 'mag_unc', 'min_sta_dis', 'sec_min_sta_dis', 'az_gap', 'n_picks_p', 'n_picks_s', 'station_count', 'region']
    print(f'Getting earthquake data from {starttime} to {endtime}...')
    df_eq = EqData(
                #bna_folder='/home/seiscomp/fixed_BNA20230328',
                bna_folder='/home/seiscomp/.seiscomp3/bna/del',
                starttime=starttime,
                endtime=endtime,
                status="'preliminary', 'final'",
                overwrite=False,
                status_in_output_filename=True,
                host='scdb.beg.utexas.edu',
                user='sysro', passwd='0niReady',
                database='seiscomp').get_df_eqs()

    print(f'Reading station data from {station_file}...')
    # Read station data
    df = pd.read_csv(station_file)
    # filter out stations that belong to the network code AM
    df = df[df['Network Code'] != 'AM']
    ic(df.columns)
    station_coords = df[['Longitude (WGS84)', 'Latitude (WGS84)']].values

    # Split lons and lats
    lon1, lon2 = map(float, lons.split(','))
    lat1, lat2 = map(float, lats.split(','))

    grid_dir = 'grids'
    grid_filename = f'{grid_dir}/grid_{lon1}_{lon2}_{lat1}_{lat2}_{grid_step}_{dist_threshold}_{os.path.basename(station_file).split(".")[0]}.csv'
    if not os.path.exists(grid_dir):
        os.makedirs(grid_dir)

    if overwrite_grid or not os.path.exists(grid_filename):
        print(f'Creating grid with step {grid_step} degrees...')
        # Create grid
        grid_lons, grid_lats = np.meshgrid(np.arange(lon1, lon2, grid_step), np.arange(lat1, lat2, grid_step))

        # Calculate distances and count stations within threshold
        count = np.zeros_like(grid_lons, dtype=int)
        for i in range(grid_lons.shape[0]):
            for j in range(grid_lons.shape[1]):
                grid_point = (grid_lats[i, j], grid_lons[i, j])
                distances = [gps2dist_azimuth(grid_point[0], grid_point[1], station[1], station[0])[0] / 1000 for station in station_coords]
                count[i, j] = sum(d <= dist_threshold for d in distances)

        print(f'Writing grid to file {grid_filename}...')
        write_grid_to_csv(grid_lons, grid_lats, count, grid_filename)
    else:
        print(f'Reading grid from file {grid_filename}...')
        grid_lons, grid_lats, count = read_grid_from_csv(grid_filename)
        
    print(f'Plotting station density map...')
    # Plot
    fig = plt.figure(figsize=(30, 20))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon1, lon2, lat1, lat2], crs=ccrs.PlateCarree())

    # Add earthquakes as circles with opacity of 0.7
    # filter the earthquakes that are between Lons and Lats
    df_eq = df_eq[(df_eq['lon'] >= lon1) & (df_eq['lon'] <= lon2) & (df_eq['lat'] >= lat1) & (df_eq['lat'] <= lat2)]
    ax.scatter(df_eq['lon'], df_eq['lat'], color='black', marker='o', alpha=0.4, transform=ccrs.PlateCarree())

    # Plot stations
    # keep only the stations that are between Lons and Lats
    df = df[(df['Longitude (WGS84)'] >= lon1) & (df['Longitude (WGS84)'] <= lon2) & (df['Latitude (WGS84)'] >= lat1) & (df['Latitude (WGS84)'] <= lat2)]
    # Add stations as triangles and station names
    ax.scatter(df['Longitude (WGS84)'], df['Latitude (WGS84)'], color='blue', marker='^', transform=ccrs.PlateCarree())
    for index, row in df.iterrows():
        ax.text(row['Longitude (WGS84)'], row['Latitude (WGS84)'], row['Station Code'], transform=ccrs.PlateCarree(), fontsize=8)

    # Plot contour map
    levels = [0, 1, 2, 3]
    g = ax.contourf(grid_lons, grid_lats, count, cmap='viridis_r', transform=ccrs.PlateCarree(), alpha=0.8, levels=levels, extend='both', vmin=0, vmax=levels[-1])

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='lightgrey')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.LAKES, color='lightblue')
    ax.add_feature(cfeature.RIVERS, color='lightblue')
    ax.add_feature(cfeature.STATES, linestyle=':')
    """cbar = plt.colorbar(g, orientation='horizontal', ticks=[0, 1, 2], shrink=0.5, aspect=20, fraction=0.1, pad=0.05)
    #cbar.ax.set_xticklabels(['0', '1', '2', '2+'])  # Set custom tick labels
    cbar.set_label(f'Number of stations within threshold ({dist_threshold} km)', size=14)"""
    # Calculate midpoints of levels for colorbar ticks
    midpoints = [(a + b) / 2 for a, b in zip(levels[:-1], levels[1:])]

    cbar = plt.colorbar(g, orientation='horizontal', shrink=0.5, aspect=20, fraction=0.1, pad=0.05)
    cbar.set_ticks(midpoints)  # Set ticks at midpoints
    cbar.set_ticklabels(['1', '2', '3'])  # Set custom tick labels
    cbar.set_label(f'Number of stations within {dist_threshold} km', size=14)
    
    plt.title(f'Station Density within {dist_threshold} km.\nEQ from {starttime} to {endtime}. Grid step: {grid_step} degrees.')

    plt.savefig(output_path)
    # Save the figure
    with open(pickle_path, 'wb') as f:
        pickle.dump(fig, f)
    
    plt.show()
    return fig

def write_grid_to_csv(grid_lons, grid_lats, count, filename):
    """
    Write the grid and count to a csv file.
    """
    with open(filename, 'w') as f:
        # Write shape information as the first row
        f.write(f'shape,{grid_lons.shape[0]},{grid_lons.shape[1]}\n')
        for i in range(grid_lons.shape[0]):
            for j in range(grid_lons.shape[1]):
                # Write each grid point's longitude, latitude, and count
                f.write(f'{grid_lons[i, j]},{grid_lats[i, j]},{count[i, j]}\n')


def read_grid_from_csv(filename):
    """
    Read the grid and count from a csv file, reconstructing the original shape.
    """
    with open(filename, 'r') as f:
        # Read the first line to get the shape
        shape_line = f.readline()
        shape = tuple(map(int, shape_line.strip().split(',')[1:]))
        
        # Now read the rest of the file into a DataFrame
        df = pd.read_csv(f, names=['lon', 'lat', 'count'])
    
    # Reshape the columns based on the extracted shape
    grid_lons = df['lon'].values.reshape(shape)
    grid_lats = df['lat'].values.reshape(shape)
    count = df['count'].values.reshape(shape)
    
    return grid_lons, grid_lats, count

if __name__ == '__main__':
    # Example usage
    station_file = 'texnet_stations/texnet_stations_2024.csv'
    lons = "-106,-94"
    lats = "28.2,36.5"
    grid_step = 0.05
    dist_threshold = 5
    starttime = '2023-01-01 00:00:00'
    endtime = '2024-12-31 23:59:59'
    plot_distance_countour_map(station_file, lons, lats, grid_step, dist_threshold, starttime, endtime, overwrite_grid=False)