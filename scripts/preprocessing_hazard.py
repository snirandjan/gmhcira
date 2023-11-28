"""
  
@Author: Sadhana Nirandjan & Elco Koks  - Institute for Environmental studies, VU University Amsterdam
"""

################################################################
                ## Load package and set path ##
################################################################
import os,sys
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
import rioxarray as rxr
import xarray as xr
#!pip install geopy
#!pip install boltons
from pathlib import Path
from geofeather.pygeos import to_geofeather, from_geofeather
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.plot import show
from IPython.display import display #when printing geodataframes, put it in columns -> use display(df)
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, LinearLocator, MaxNLocator)
import pygeos
from pgpkg import Geopackage
import matplotlib.pyplot as plt
import copy
from multiprocessing import Pool,cpu_count

#sys.path.append("C:\Projects\gmhcira\scripts")
import functions

plt.rcParams['figure.figsize'] = [20, 20]

from osgeo import gdal
gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join("..","osmconf.ini"))


################################################################
                    ## Set pathways ##
################################################################

local_path = os.path.join('/scistor','ivm','snn490')

# Set path to inputdata
hazard_data_path = os.path.abspath(os.path.join('/scistor','ivm','data_catalogue','open_street_map','global_hazards'))
fathom_data_path = os.path.abspath(os.path.join('/scistor','ivm','eks510','fathom-global'))

shapes_file = 'global_countries_advanced.geofeather'
country_shapefile_path = os.path.abspath(os.path.join(local_path, 'Datasets', 'Administrative_boundaries', 'global_countries_buffer', shapes_file))

# path to our python scripts
sys.path.append(os.path.join('..','scripts'))

################################################################
                    ## set variables ##
################################################################

#hazard_dct = {'coastal_flooding': [2, 5, 10, 25, 50, 100, 250, 500, 1000], 
#              'fluvial_flooding': [5, 10, 20, 50, 75, 100, 200, 250, 500, 1000], 
#              'earthquakes': [250, 475, 975, 1500, 2475], 
#              'tropical_cyclones': [10, 20 , 50, 100, 200, 1000, 2000, 5000, 10000], 
#              'landslides': [None]} #keys are hazards, lists are return periods associated with hazard

hazard_dct = {'coastal_flooding': [2]} #, 5, 10, 25, 50, 100, 250, 500, 1000]} #keys are hazards, lists are return periods associated with hazard
country = 'NLD' #temporary

################################################################
                    ## hazard data preprocessing functions##
################################################################

def transform_raster_to_vectorgrid(hazard_file,hazard_type):             
    """load raster in polygon format

    Args:
        hazard_file (string): pathway to file
        hazard_type (string): hazard type that is being analyzed

    Returns:
        dataframe: dataframe containing hazard intensities in format of polygon data of hazard type
    """    
    hazard_point_df = load_raster_as_dataframe(hazard_file,hazard_type)
    hazard_df = add_buffer(hazard_type, hazard_point_df)
    
    return hazard_df

def load_raster_as_dataframe(file_path,hazard_type):
    """load raster and transform to dataframe

    Args:
        hazard_file (string): pathway to file
        hazard_type (string): hazard type that is being analyzed

    Returns:
        dataframe: dataframe containing hazard intensities in format of point data of hazard type
    """    
    with xr.open_rasterio(file_path) as ds:
        
        if hazard_type == 'pluvial':
            df_ds = ds.to_dataframe(name='hazard_intensity').reset_index()    
        elif hazard_type == 'fluvial':
            df_ds = ds.to_dataframe(name='hazard_intensity').reset_index() 
        elif hazard_type == 'coastal_flooding':
            df_ds = ds.to_dataframe(name='hazard_intensity').reset_index() 

        df_ds = df_ds.loc[(df_ds.hazard_intensity != 999) & (df_ds.hazard_intensity != -9999) & (df_ds.hazard_intensity != 0)].reset_index(drop=True)
        df_ds['geometry'] = pygeos.points(df_ds.x.values,y=df_ds.y.values)
        df_ds = df_ds.drop(['band','y','x'],axis=1)

    return df_ds


def add_buffer(hazard_type, hazard_df):             
    """function to create square buffer around point data

    Args:
        hazard_type (string): hazard type that is being analyzed
        hazard_df (dataframe): dataframe containing hazard intensities in format of point data of hazard type

    Returns:
        dataframe: dataframe containing hazard intensities in format of polygon data of hazard type
    """    
    # load hazard data and create spatial index
    if hazard_type in ['pluvial','fluvial']:
        grid_size = 0.0008333333333333333868
    elif hazard_type == 'coastal_flooding':
        grid_size = 0.008333333333333333218
    elif hazard_type == 'landslides':
        grid_size = 0.008333333332237680136
        
    # create squares again from flood points
    hazard_df.geometry =  pygeos.buffer(hazard_df.geometry.values,radius=grid_size/2,cap_style='square')

    return hazard_df