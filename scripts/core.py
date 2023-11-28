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
from multiprocessing import Pool,cpu_count
from itertools import repeat
import pygeos
from pgpkg import Geopackage
import matplotlib.pyplot as plt
import copy

#sys.path.append(os.path.join('..','scripts')) # path to our python scripts
#sys.path.append("C:\Projects\gmhcira\scripts")
import functions
import preprocessing_hazard

plt.rcParams['figure.figsize'] = [20, 20]

from osgeo import gdal
gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join("..","osmconf.ini"))

#def run_all(goal_area = 'Netherlands', local_path = 'C:/Users/snn490/surfdrive'):
def run_all(countries, goal_area = 'Global', local_path = os.path.join('/scistor','ivm','snn490')):
    """Function to manage and run the model (in parts). 
    Args:
        *areas* ([str]): list with areas (e.g. list of countries)
        *goal_area* (str, optional): area that will be analyzed. Defaults to "Global". 
        *local_path* ([str], optional): Local pathway. Defaults to os.path.join('/scistor','ivm','snn490').
    """    

    hazard_preprocessing(local_path)
    #base_calculations(local_path) 
    #base_calculations_global(local_path) #if base calcs per area already exist
    #cisi_calculation(local_path,goal_area)



################################################################
                    ## set variables ##
################################################################

def set_variables():
    """Function to set the variables that are necessary as input to the model
    Returns:
        hazard_dct (dictionary): overview of hazards and return periods 
    """    
    #hazard_dct = {'coastal_flooding': [2, 5, 10, 25, 50, 100, 250, 500, 1000], 
    #              'fluvial_flooding': [5, 10, 20, 50, 75, 100, 200, 250, 500, 1000], 
    #              'earthquakes': [250, 475, 975, 1500, 2475], 
    #              'tropical_cyclones': [10, 20 , 50, 100, 200, 1000, 2000, 5000, 10000], 
    #              'landslides': [None]} #keys are hazards, lists are return periods associated with hazard

    hazard_dct = {'coastal_flooding': [2, 5, 10, 25, 50, 100, 250, 500, 1000]} #keys are hazards, lists are return periods associated with hazard

    return [hazard_dct]

################################################################
                    ## Set pathways ##
################################################################

def set_paths(local_path = 'C:/Data/CISI',hazard_preprocessing=False):
    """Function to specify required pathways for inputs and outputs
    Args:
        *local_path* (str, optional): local path. Defaults to 'C:/Data/CISI'.
        *hazard_preprocessing* (bool, optional): True if hazard preprocessing part of model should be activated. Defaults to False.
    Returns:
        *hazard_data_path* (str): directory to coastal flood, landslide, earthquake and cyclone hazard data
        *fathom_data_path* (str): directory to fathom hazard data
        *country_shapefile_path* (str): directory to file with country boundaries
    """ 
    # Set path to inputdata
    hazard_data_path = os.path.abspath(os.path.join('/scistor','ivm','data_catalogue','open_street_map','global_hazards'))
    fathom_data_path = os.path.abspath(os.path.join('/scistor','ivm','eks510','fathom-global'))

    shapes_file = 'global_countries_advanced.geofeather'
    country_shapefile_path = os.path.abspath(os.path.join(local_path, 'Datasets', 'Administrative_boundaries', 'global_countries_buffer', shapes_file))

    if hazard_preprocessing:
        return [hazard_data_path,fathom_data_path,country_shapefile_path]

################################################################
 ## Step 1: hazard_preprocessing  ##
################################################################    

def hazard_preprocessing_per_area(country,hazard_dct,hazard_data_path,country_shapefile_path,hazard_type,file_path_lst,file_path):
    """function to extract infrastrastructure for an area 
    Args:
        *area* (str): area to be analyzed
        *osm_data_path* (str): directory to osm data
        *fetched_infra_path* (str): directory to output location of the extracted infrastructure data 
        *country_shapes_path* (str): directory to dataset with administrative boundaries (e.g. of countries)
    """
    shape_countries = from_geofeather(country_shapefile_path) #open as geofeather
    hazard_df = from_geofeather(file_path.replace('tif', 'feather')) #open as geofeather
    rp = str(hazard_dct[hazard_type][file_path_lst.index(file_path)]) #get return period using index

    if hazard_type in ['coastal_flooding', 'earthquakes']:
        if os.path.isfile(os.path.join(hazard_data_path, hazard_type, 'rp_{}'.format(rp), '{}_rp{}_{}.feather'.format(hazard_type, rp, country))) == False: #if feather file does not already exists 
            #soft overlay of hazard data with countries
            country_shape = shape_countries[shape_countries['ISO_3digit'] == country]
            if country_shape.empty == False: #if ISO_3digit in shape_countries
                print("Time to overlay and output '{}' hazard data for the following country: {}".format(hazard_type, country))
                spat_tree = pygeos.STRtree(hazard_df.geometry)
                hazard_country_df = (hazard_df.loc[spat_tree.query(country_shape.geometry.iloc[0],predicate='intersects').tolist()]).sort_index(ascending=True) #get grids that overlap with country
                hazard_country_df = hazard_country_df.reset_index() #.rename(columns = {'index':'grid_number'}) #get index as column and name column grid_number

                #save data
                Path(os.path.abspath(os.path.join(hazard_data_path, hazard_type, 'rp_{}'.format(rp)))).mkdir(parents=True, exist_ok=True) 
                to_geofeather(hazard_country_df, os.path.join(hazard_data_path, hazard_type, 'rp_{}'.format(rp), '{}_rp{}_{}.feather'.format(hazard_type, rp, country)), crs="EPSG:4326") #save as geofeather #save file for each country
                #to_geofeather(hazard_country_df, os.path.join(hazard_data_path, hazard_type, rp, '{}_{}_{}.feather'.format(hazard_type, rp, country)), crs="EPSG:4326") #save as geofeather #save file for each country
                temp_df = functions.transform_to_gpd(hazard_country_df) #transform df to gpd with shapely geometries
                temp_df.to_file(os.path.join(hazard_data_path, hazard_type, 'rp_{}'.format(rp), '{}_rp{}_{}.gpkg'.format(hazard_type, rp, country)), layer=' ', driver="GPKG")              
            else:
                print("Country '{}' not specified in file containing shapefiles of countries with ISO_3digit codes. Please check inconsistency".format(country))
        else:
            print('Hazard data already exists at country level, file: {}_rp{}_{}'.format(hazard_type,rp,country))

    elif hazard_type == 'fluvial_flooding':  ###STIL NEED TO WORK ON THIS PART
        file_path_lst = os.path.abspath(os.path.join(hazard_data_path, country, 'fluvial_undefended', 'FU_1in{}'.format(rp))) #pathway to file
        #hazard_country_df = preprocessing_hazard.transform_raster_to_vectorgrid(file_path, hazard_type) 
    
    else:
        print("def preprocess_hazard_per_area has not been defined for the following hazard type: {}".format(hazard_type))



def hazard_preprocessing(local_path):
    """function to disaggregate hazard data at country level, parallel processing 
    Args:
        *local_path*: Local pathway. Defaults to os.path.join('/scistor','ivm','snn490').
    """
    # get paths
    hazard_data_path,fathom_data_path,country_shapefile_path = set_paths(local_path,hazard_preprocessing=True)

    # get settings
    hazard_dct = set_variables()[0]

    #import hazard data
    for hazard_type in hazard_dct:
        #data-preprocessing for coastal floods and earthquakes
        if hazard_type in ['coastal_flooding', 'earthquakes']:
            if hazard_type == 'coastal_flooding':
                file_path_lst = [os.path.abspath(os.path.join(hazard_data_path, hazard_type, 'inuncoast_historical_nosub_hist_rp{:0>4d}_0.tif'.format(rp))) for rp in hazard_dct[hazard_type]] #pathway to file
            elif hazard_type == 'earthquakes':
                file_path_lst = [os.path.abspath(os.path.join(hazard_data_path, hazard_type, 'rp_'.format(rp), 'gar17pga{}.tif'.format(rp))) for rp in hazard_dct[hazard_type]] #pathway to file

            #transform to vector data
            for file_path in file_path_lst:
                if os.path.isfile('{}'.format(file_path.replace('tif', 'feather'))) == False: #if feather file does not already exists
                    print('Time to transform global hazard data into polygon format for the following hazard type: {}, rp {}'.format(hazard_type, hazard_dct[hazard_type][file_path_lst.index(file_path)]))
                    hazard_df = preprocessing_hazard.transform_raster_to_vectorgrid(file_path,hazard_type)
                    to_geofeather(hazard_df, '{}'.format(file_path.replace('tif', 'feather')), crs="EPSG:4326") #save as geofeather  
                    print('Global hazard data has been transformed and outputted in polygon format for the following hazard type: {}, rp {}'.format(hazard_type, hazard_dct[hazard_type][file_path_lst.index(file_path)]))
                else: 
                    print('Transformed global hazard data in polygon format already exists for the following hazard type: {}, rp {}'.format(hazard_type, hazard_dct[hazard_type][file_path_lst.index(file_path)]))

                # run the extract parallel per country
                print('Time to start hazard data preprocessing for the following countries: {}'.format(countries))
                with Pool(cpu_count()-1) as pool: 
                    pool.starmap(hazard_preprocessing_per_area,zip(countries,
                                                                    repeat(hazard_dct,len(countries)),
                                                                    repeat(hazard_data_path,len(countries)),
                                                                    repeat(country_shapefile_path,len(countries)),
                                                                    repeat(hazard_type,len(countries)),
                                                                    repeat(file_path_lst,len(countries)),
                                                                    repeat(file_path,len(countries))),
                                                                    chunksize=1)
    
        #data preprocessing for fluvial flooding ###STIL NEED TO WORK ON THIS PART
        elif hazard_type == 'fluvial_flooding':
            # run the extract parallel per country
            print('Time to start hazard data preprocessing for the following countries: {}'.format(countries))
            with Pool(cpu_count()-1) as pool: 
                pool.starmap(hazard_preprocessing_per_area,zip(countries,
                                                                repeat(hazard_dct,len(countries)),
                                                                repeat(fathom_data_path,len(countries)),
                                                                repeat(hazard_df,len(countries)),
                                                                repeat(shape_countries,len(countries)),
                                                                repeat(hazard_type,len(countries)),
                                                                repeat(file_path_lst,len(countries))),
                                                                chunksize=1)

################################################################
        ## Step 2: Damage assessment ##
################################################################    
          
def regional_analysis(country,reg_index,asset_type='roads',hazard='pluvial'):
    
    
    
    
def global_hazard_analysis():
    """function to perform , parallel processing 
    Args:
        *local_path*: Local pathway. Defaults to os.path.join('/scistor','ivm','snn490').
    """
    
        if hazard == 'pluvial':
            haz_short = 'P'
            hazard_data_path = os.path.join('..','hazard_data','pluvial')
            return_periods = [5,10,20,50,75,100,200,250,500,1000]

        elif hazard == 'fluvial':
            haz_short = 'FD'
            hazard_data_path = os.path.join('..','hazard_data','fluvial_defended')
            return_periods = [5,10,20,50,75,100,200,250,500,1000]

        elif hazard == 'coastal':
            haz_short = 'CF'
            hazard_data_path = os.path.join('..','hazard_data','coastal_flooding')
            return_periods = #[XX,XX]

        elif hazard == 'earthquake':
            haz_short = 'EQ'
            hazard_data_path = os.path.join('..','hazard_data','coastal_flooding')
            return_periods = #[XX,XX]

        elif hazard == 'tropical_cyclones':
            haz_short = 'TC'
            hazard_data_path = os.path.join('..','hazard_data','tropical_cyclones')
            return_periods = #[XX,XX]           

        elif hazard == 'landslides':
            haz_short = 'LS'
            hazard_data_path = os.path.join('..','hazard_data','landslides')
            return_periods = [1]


        #check if file is already finished
        if hazard in ['pluvial','fluvial','coastal','earthquake','landslide','tropical_cyclones']:
            if os.path.exists(os.path.join('..','{}_damage'.format(hazard),'{}_{}_{}.csv'.format(country,reg_index,asset_type))):
                return None

        elif hazard in ['wildfires','temperature']:
            if os.path.exists(os.path.join('..','{}_exposure'.format(hazard),'{}_{}_{}.csv'.format(country,reg_index,asset_type))):
                return None  
    
    
    
    

if __name__ == '__main__':
    #receive nothing, run area below
    if (len(sys.argv) == 1):    
        countries = ['Galveston_Bay']#['Zuid-Holland']
        run_all(countries)
    else:
        # receive ISO code, run one country
        if (len(sys.argv) > 1) & (len(sys.argv[1]) == 3):    
            countries = []
            countries.append(sys.argv[1])
            run_all(countries)
        #receive multiple ISO-codes in a list, run specified countries
        elif '[' in sys.argv[1]:
            if ']' in sys.argv[1]:
                countries = sys.argv[1].strip('][').split('.') 
                run_all(countries)
            else:
                print('FAILED: Please write list without space between list-items. Example: [NLD.LUX.BEL]')
        #receive global, run all countries in the world
        elif (len(sys.argv) > 1) & (sys.argv[1] == 'global'):    
            #glob_info = pd.read_excel(os.path.join('/scistor','ivm','snn490','Datasets','global_information_short.xlsx'))
            #glob_info = pd.read_excel(os.path.join(r'C:\Users\snn490\surfdrive\Datasets','global_information_test.xlsx'))
            glob_info = pd.read_excel(os.path.join('/scistor','ivm','snn490','Datasets','global_information_advanced.xlsx'))
            countries = list(glob_info.ISO_3digit) 
            if len(countries) == 0:
                print('FAILED: Please check file with global information')
            else:
                run_all(countries)
        #receive continent, run all countries in continent
        elif (len(sys.argv) > 1) & (len(sys.argv[1]) > 3):    
            #glob_info = pd.read_excel(os.path.join('/scistor','ivm','snn490','Datasets','global_information_short.xlsx'))
            #glob_info = pd.read_excel(os.path.join(r'C:\Users\snn490\surfdrive\Datasets','global_information_test.xlsx'))
            glob_info = pd.read_excel(os.path.join('/scistor','ivm','snn490','Datasets','global_information_advanced.xlsx'))
            glob_info = glob_info.loc[glob_info.continent==sys.argv[1]]
            countries = list(glob_info.ISO_3digit) 
            if len(countries) == 0:
                print('FAILED: Please write the continents as follows: Africa, Asia, Central-America, Europe, North-America,Oceania, South-America') 
            else:
                run_all(countries)
        else:
            print('FAILED: Either provide an ISO3 country name or a continent name')
