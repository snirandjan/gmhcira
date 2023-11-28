"""
  
@Author: Sadhana Nirandjan & Elco Koks  - Institute for Environmental studies, VU University Amsterdam
"""

import os,sys
from osgeo import ogr,gdal
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
import rasterio
import xarray as xr
import pygeos
from tqdm import tqdm
import pyproj
from collections import defaultdict
from multiprocessing import Pool,cpu_count
from functools import partial

#sys.path.append(os.path.join( '..','scripts'))
#from core import *
osm_data_path = os.path.join('..','osm_data')
gadm_data_path = os.path.join('..','gadm_data')

hazard_data_path = os.path.join('..','hazard_data')
import itertools

tqdm.pandas()

gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join('..',"osmconf.ini"))

def default_factory():
    return 'nodata'

def query_b(geoType,keyCol,**valConstraint):
    """
    This function builds an SQL query from the values passed to the retrieve() function.
    Arguments:
         *geoType* : Type of geometry (osm layer) to search for.
         *keyCol* : A list of keys/columns that should be selected from the layer.
         ***valConstraint* : A dictionary of constraints for the values. e.g. WHERE 'value'>20 or 'value'='constraint'
    Returns:
        *string: : a SQL query string.
    """
    query = "SELECT " + "osm_id"
    for a in keyCol: query+= ","+ a  
    query += " FROM " + geoType + " WHERE "
    # If there are values in the dictionary, add constraint clauses
    if valConstraint: 
        for a in [*valConstraint]:
            # For each value of the key, add the constraint
            for b in valConstraint[a]: query += a + b
        query+= " AND "
    # Always ensures the first key/col provided is not Null.
    query+= ""+str(keyCol[0]) +" IS NOT NULL" 
    return query 

def buffer_assets(assets,buffer_size=100):
    """[summary]

    Args:
        assets ([type]): [description]
        buffer_size (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """    
    assets['buffered'] = pygeos.buffer(assets.geometry.values,buffer_size)# assets.geometry.progress_apply(lambda x: pygeos.buffer(x,buffer_size)) 

    return assets

def overlay_hazard_assets(df_ds,assets):
    """[summary]

    Args:
        df_ds ([type]): [description]
        assets ([type]): [description]

    Returns:
        [type]: [description]
    """
    #overlay 
    flooded_tree = pygeos.STRtree(df_ds.geometry.values)
    return  flooded_tree.query_bulk(assets.geometry,predicate='intersects')    

def save_as_feather(df,path,file):
    df.geometry = pygeos.to_wkb(df.geometry)
    df.to_feather(os.path.join(path,file))
    
def load_feather(path,file):
    df = pd.read_feather(os.path.join(path,file))        
    df.geometry = pygeos.from_wkb(df.geometry)
    return df

def regional_analysis(country,reg_index,asset_type='roads',hazard='pluvial'):
    """[summary]

    Args:
        country ([type]): [description]
        reg_index ([type]): [description]
        asset_type (str, optional): [description]. Defaults to 'roads'.
        hazard (str, optional): [description]. Defaults to 'pluvial'.

    Returns:
        [type]: [description]
    """ 

    try:

        print('{} {} started!'.format(country,reg_index))

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

        # specify list of sub systems
        point_assets = ['health','telecom']
        polygon_assets = ['airports','educational_facilities','waste_solid','waste_water','water_supply']
        line_assets = ['railways','roads']        

        # load curves and maxdam
        curves,maxdam = load_curves_maxdam(data_path=os.path.join('..','data','infra_vulnerability_data.xlsx'))

        #load regions
        lev1_gadm = pd.DataFrame(gpd.read_file(os.path.join(gadm_data_path,'gadm36_levels.gpkg'),layer='level1'))
        lev1_gadm = lev1_gadm.loc[lev1_gadm.GID_0 == country].reset_index()
        lev1_gadm.geometry = pygeos.from_shapely(lev1_gadm.geometry)
        
        #load asset data
        assets = load_assets(osm_data_path,asset_type,country)
        assets = pygeos.intersection(assets.geometry,lev1_gadm.iloc[reg_index].geometry)
    
        if (asset_type in line_assets) | (asset_type in point_assets):
            ### get line assets, clip and buffer them
            assets = buffer_assets(assets,buffer_size=100)
        
        elif asset_type == 'power':
            power_lines = buffer_assets(assets.loc[assets.asset.isin(['cable','minor_cable','line','minor_line'])],buffer_size=100).reset_index(drop=True)
            power_poly = assets.loc[assets.asset.isin(['plant','substation'])].reset_index(drop=True)
            power_points = buffer_assets(assets.loc[assets.asset.isin(['power_tower','power_pole'])],buffer_size=100).reset_index(drop=True)

        print('Asset data loaded for {} in {}'.format(asset_type,country))
      
        if len(assets) == 0:
            assets.to_csv(os.path.join('..','{}_damage'.format(hazard),'{}_{}.csv'.format(reg_index,asset_type)))
            return None
        

        #reproject roads
        assets_reg.geometry = reproject_assets(assets_reg,current_crs="epsg:4326",approximate_crs ="epsg:3857")
        
        #rename roads
        del df_asset

        print('NOTE: {} loaded and clipped to region {}'.format(asset_type,reg_index))
        
        # set damage presets
        water_depths = np.arange(0,1100,100)
        fragility_values = np.arange(0,1.1,0.1)
        
        #hazard impacts
        for return_period in tqdm(return_periods,total=len(return_periods),desc='Assessment for {} {} {} {}'.format(country,reg_index,hazard,asset_type)):
        
            # load hazard data and create spatial index
            if hazard in ['pluvial','fluvial']:
                df_ds = load_feather(os.path.join(hazard_data_path),country,'{}_1in{}.ft'.format(haz_short,return_period))
                grid_size = 0.0008333333333333333868
            elif hazard == 'landslides':
                df_ds = load_feather(os.path.join(hazard_data_path),'landslide_rainfall_trigger.ft')
                grid_size = 0.008333333332237680136
            elif hazard == 'wildfires':
                df_ds = load_feather(os.path.join(hazard_data_path),'csiro_wf_max_fwi_rp{}.ft'.format(return_period))
                grid_size = 0.5
            elif hazard == 'temperature':
                df_ds = load_feather(os.path.join(hazard_data_path),'VITO-EH-{}y.ft'.format(return_period))
                grid_size = 0.08333000000000000129

            sindex_hazard = pygeos.STRtree(df_ds.geometry)

            #get regional hazard data
            hazard_reg = df_ds.iloc[sindex_hazard.query(lev1_gadm.iloc[reg_index].geometry,predicate='intersects')].reset_index() #NOTE lev1 --> grid
            
            # create squares again from flood points
            hazard_reg.geometry =  pygeos.buffer(hazard_reg.geometry.values,radius=grid_size/2,cap_style='square')
            
            #reproject
            hazard_reg.geometry = reproject_assets(hazard_reg,current_crs="epsg:4326",approximate_crs ="epsg:3857")
            
            del df_ds

            #overlay hazards and roads    
            hazard_overlay = overlay_hazard_assets(hazard_reg,assets_reg)
            
            #calculate damage
            asset_geoms = assets_reg.geometry
            flood_geoms = hazard_reg.geometry.values
            flood_intensities = hazard_reg.hazard_intensity.values

            # load paved file
            paved_ratios = pd.read_csv(os.path.join('..','input_data','paved_ratios.csv'),index_col=[0,1]).loc[country]
            paved_dict = (paved_ratios['paved']/100).to_dict()

            asset_types = assets_reg.assets.values

            # prepare sampling with all possible combinations
            sampling = [[1,2,3,4,5],[0.75,1,1.25],[0,0.2,0.4,0.6,0.8,1]]
            sample_set = (list(itertools.product(*sampling)))

            # simple fragility curve used in Koks et al. (2019)
            if (asset_type == 'roads') | (asset_type == 'railway'): 
                water_depths = np.asarray([0,10,25,50,75,100,150])
                fragility_values = np.asarray([0,0,0.01,0.02,0.05,0.1,0.2])

            if asset_type == 'roads':                
                # paved costs for europe and central asia
                Paved_4L_costs = 1718347/1e3 
                Paved_2L_costs = 1587911/1e3
                Gravel_cost = 26546/1e3

            elif asset_type == 'railways':
                electric = 1000000/1e3 
                nonelectric = 750000/1e3

            if hazard in ['pluvial','fluvial']:

                collect_damage = {}
                for asset_id in (np.unique(hazard_overlay[0])):
                    asset_geom = asset_geoms[asset_id]
                    road_type = asset_types[asset_id]

                    match_hazard_overlays = hazard_overlay[1][hazard_overlay[0]==asset_id]

                    local_hazard_geoms = flood_geoms[match_hazard_overlays]
                    local_hazard_intensities = flood_intensities[match_hazard_overlays]
                    
                    overlay_meters = pygeos.length(pygeos.intersection(local_hazard_geoms,asset_geom))

                    uncer_output = []
                    for sample in sample_set:
                        damage_ratios = np.interp(local_hazard_intensities*100,water_depths,fragility_values*sample[0])

                        if asset_type == 'roads':
                            uncer_output.append(np.sum(paved_dict[road_type]*(damage_ratios*Paved_4L_costs*overlay_meters*sample[1]*(1-sample[2]) +
                                        damage_ratios*Paved_2L_costs*overlay_meters*sample[1]*(sample[2])) +
                        (1-paved_dict[road_type])*damage_ratios*Gravel_cost*overlay_meters*sample[1]))

                        elif asset_type == 'railways':
                            uncer_output.append(np.sum((damage_ratios*electric*overlay_meters*sample[1]*(1-sample[2]) +
                                        damage_ratios*nonelectric*overlay_meters*sample[1]*(sample[2]))))                     

                    uncer_output = np.asarray(uncer_output)

                    collect_damage[asset_id] = np.percentile(uncer_output,[0,20,40,50,60,80,100],axis=0)    
        
                df_damages = pd.DataFrame.from_dict(collect_damage,orient='index',columns = ['{}_perc_1in{}'.format(x,return_period) for x in [0,20,40,50,60,80,100]])
                assets_reg = assets_reg.join(df_damages)

            elif hazard == 'landslides':
                collect_probabilities = {}
                for asset_id in (np.unique(hazard_overlay[0])):
                    asset_geom = asset_geoms[asset_id]
                    road_type = asset_types[asset_id]

                    match_hazard_overlays = hazard_overlay[1][hazard_overlay[0]==asset_id]

                    local_hazard_intensities = flood_intensities[match_hazard_overlays]

                    collect_probabilities[asset_id] = np.mean(local_hazard_intensities),np.min(local_hazard_intensities),np.max(local_hazard_intensities)
                
                df_exposure = pd.DataFrame.from_dict(collect_probabilities,orient='index',columns = ['mean','min','max'])
                assets_reg = assets_reg.join(df_exposure)

            elif hazard in ['wildfires','temperature']:
                
                collect_probabilities = {}
                for asset_id in (np.unique(hazard_overlay[0])):
                    asset_geom = asset_geoms[asset_id]
                    road_type = asset_types[asset_id]

                    match_hazard_overlays = hazard_overlay[1][hazard_overlay[0]==asset_id]
                    
                    local_hazard_intensities = flood_intensities[match_hazard_overlays]
                    
                    collect_probabilities[asset_id] = np.mean(local_hazard_intensities),np.min(local_hazard_intensities),np.max(local_hazard_intensities)
                
                df_exposure = pd.DataFrame.from_dict(collect_probabilities,orient='index',columns = ['mean_{}'.format(return_period),'min_{}'.format(return_period),'max_{}'.format(return_period)])
                assets_reg = assets_reg.join(df_exposure)

        if hazard in ['pluvial','fluvial']:
            assets_reg.to_csv(os.path.join('..','{}_damage'.format(hazard),'{}_{}.csv'.format(reg_index,asset_type)))
        
        elif (hazard == 'landslides') | (hazard == 'wildfires') | (hazard == 'temperature') :
            assets_reg.to_csv(os.path.join('..','{}_exposure'.format(hazard),'{}_{}.csv'.format(reg_index,asset_type)))

    except Exception as e: 
        print('{} failed because of {}'.format(reg_index,e))


