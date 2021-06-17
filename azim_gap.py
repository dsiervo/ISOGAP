#!/home/dsiervo/miniconda3/bin/python
# -*- coding: utf-8 -*-

#                           ISOGAP 0.1
#This script calculates the primary azimuthal gap for a certain network geometry
#Input:
#- A csv file with station name, longitude and latitude columns
#- Corners of a rectangle to create the grided area
#- Grid step
## Author: Nelson David Perez e-mail:ndperezg@gmail.com 


import numpy as np
import nvector as nv
from obspy.geodetics import gps2dist_azimuth
import sys
import os
import multiprocessing as mp
import glob
"""import click


@click.command()
@click.option('-g', "--grids_dir", required=True, prompt=True, help='Waveform file name')
@click.option('-o', "--output_dir", required=True, prompt=True, help='Dataless file name')
@click.option('-m', "--pool_mode", default="avg", help='units to convert. Can be: DIS, VEL, ACC')
@click.option('-p', "--plot", default=False, type=bool, help='Define if waveform will be plotted')"""

#Calculates gaps for a list of azimuths 
def gaps(Azz):
    azz = sorted(Azz)
    gaps_ = []
    for i in range(len(azz)):
        if i != 0:
            alpha = azz[i]-azz[i-1]
        else:
            alpha = azz[0]+ 360 - azz[-1]
        gaps_.append(alpha)
    return gaps_

#Calculates azimuths between two points
def azimuth(Lon1,Lat1,Lon2,Lat2):
    """wgs84 = nv.FrameE(name='WGS84')
    pointA = wgs84.GeoPoint(latitude=Lat1, longitude=Lon1, z=0, degrees=True)
    pointB = wgs84.GeoPoint(latitude=Lat2, longitude=Lon2, z=0, degrees=True)
    p_AB_N = pointA.delta_to(pointB)
    azim = p_AB_N.azimuth_deg"""
    azim = gps2dist_azimuth(Lat1, Lon1, Lat2, Lon2)[1]
    if azim < 0:
        azim += 360
    else:
        pass
    return azim

#calculates isogap for each point
def each_gap(lon,lat,net):
    azz=[]
    for sta in net:
        azim = azimuth(lon,lat,net[sta][0], net[sta][1])
        azz.append(azim)
    GAP = max(gaps(azz))
    return GAP


#reads stations file
def read_stations(arc):
    with open(arc) as fl:
        count = 0
        NET = {}
        for line in fl.readlines():
            point=[]
            if count > 0:
                sta = line.strip().split(',')[0]
                lon = float(line.strip().split(',')[1])
                lat = float(line.strip().split(',')[2])
                point.append(lon)
                point.append(lat)
                NET[sta] = point
            count += 1
    return NET


#Ask for longitudes and latitudes for the study area
def input_area(custom=False):
    if custom:
        lons = input("Enter min and max longitudes separated by a comma: ")
        lats = input("Enter min and max latitudes separated by a comma: ")
    else:
        lons = '-80,-67'
        lats = '-3,14'
    if len(lons.split(',')) != 2 or len(lats.split(',')) != 2:
        print("Bad input, try again\n")
        sys.exit()
    minlon = float(lons.split(',')[0])
    maxlon = float(lons.split(',')[1])
    minlat = float(lats.split(',')[0])
    maxlat = float(lats.split(',')[1])
    if (minlon >= maxlon) or (minlat >= maxlat):
        print("Wrong values, try again\n")
        sys.exit()
    return minlon, maxlon, minlat, maxlat


def azim_gap(sta_dir, grid, custom=False):

    if sta_dir in ['no', 'n', 'NO', 'No', 'N', 'default', 'dfalut']:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        sta_dir = os.path.join(scripts_dir, 'station_coordinates')
    #sta_dir = input('\nEnter the directory containing the csv files with the stations coordinates:\n\t--> ')
    #grid = float(input('Enter grid step in degrees:\n\t--> '))
    minlon, maxlon, minlat, maxlat = input_area(custom)
    Lons = np.arange(minlon, maxlon, grid)
    Lats = np.arange(minlat, maxlat, grid)

    # Creatig grids folder if doesn't exist
    output_dir = 'grids'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    sta_files = glob.glob(os.path.join(sta_dir, '*.csv'))
    n = len(sta_files)

    with mp.Pool(processes=mp.cpu_count()) as p:
        p.map(write_grid,
              zip([grid]*n, [Lons]*n, [Lats]*n, [output_dir]*n, sta_files))


def write_grid(arguments):
    grid, Lons, Lats, output_dir, arc = arguments
    NET = read_stations(arc)
    basename = os.path.basename(arc).split('.')[0]
    output_file = os.path.join(output_dir, '%s_%s_grid.csv' % (basename, grid))
    out = open(output_file, 'w')
    out.write('LON,LAT,GAP\n')
    for i in Lons:
        for j in Lats:
            az_gap = each_gap(i, j, NET)
            print(i, j, az_gap)
            out.write('%s,%s,%4.2f\n' % (i, j, az_gap))
    out.close()
    print('\n\t%s was created\n' % output_file)


if __name__ == '__main__':
    
    azim_gap(False)
