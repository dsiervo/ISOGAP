# gap_heatmap.py

## Introduction
The gap_heatmap.py program allows quantitative monitoring of the evolution of the coverage of the National Seismological Network of Colombia in the areas of greatest interest to the Group of Evaluation and Monitoring of Seismic Activity of the Colombian Geological Survey, by calculating the maximum azimuthal GAP average in these areas and the generation of heat and contour maps.

The main script (gap_heatmap.py) has 3 commands that allow to perform the following tasks:
1. [**grids**] Compute azimuthal GAP over a grid (in Colombia) taking into account the distribution of the seismological network (in this particular case, the Colombian National Seismological Network)
2. [**heatmaps**] With those grids created in the previous step, generates a heatmap and a contour plot over Colombian geographical map.
3. [**g-h**] Perform the 2 above.

### Heatmap example
![heatmap](output_maps/big/heatmaps/heatmap_2015.png)

### Countour map example
![contour](output_maps/contours/contour_2015.png)

## Installation
### Prerequisites
* python 3.7 - 3.8

For non anaconda python installations:
```
$ apt-get install libproj-dev proj-data proj-bin  
$ apt-get install libgeos-dev  
$ pip install cython 
```

### Procedure
Clone or download this repository and then:

#### Non anaconda users

```
$ pip install -r requirements.txt
```

#### Anaconda users
If you have an anaconda installation:

```
$ conda install -c scitools cartopy
$ pip install -r requirements.txt
```

## Usage

The program has 3 commands that allow you to carry out the following specific tasks:
1. grids: Calculate the azimuthal gap in a grid over Colombia taking into account the distribution of the stations that are entered through a .csv file with their coordinates or by default, with the stations of the National Seismological Network of Colombia for the years 2015-2020.
2. heatmaps: With the grids calculated in the previous step, generate a heat map and a contour map on the geographic map of Colombia.
3. g-h: Perform 1 and 2.

To get the command names, type:
```
$ python gap_heatmap.py --help
```

To get help about each command, type:
```
$ python gap_heatmap.py [command] --help
```
