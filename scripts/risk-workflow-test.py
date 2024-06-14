"""

  
@Author: Sadhana Nirandjan & Elco Koks  - Institute for Environmental studies, VU University Amsterdam
"""

import os
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np 
import shapely 
import csv
import ast

import osm_flex.download as dl
import osm_flex.extract as ex
from osm_flex.simplify import remove_contained_points,remove_exact_duplicates,remove_contained_polys
from osm_flex.config import OSM_DATA_DIR,DICT_GEOFABRIK

from tqdm import tqdm

from lonboard import viz
from lonboard.colormap import apply_continuous_cmap
from palettable.colorbrewer.sequential import Blues_9

from pathlib import Path
import pathlib


########################################################################################################################
################          define paths          #############################################################
########################################################################################################################
p = Path('..')
data_path = Path(pathlib.Path('Z:') / 'snn490' / 'Projects' / 'gmhcira' / 'data')  #should contain folder 'Vulnerability' with vulnerability data
flood_data_path = Path(pathlib.Path('Z:') / 'eks510' / 'fathom-global') # Flood data
eq_data_path = Path(pathlib.Path('Z:') / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'earthquakes') # Earthquake data
landslide_data_path = Path(pathlib.Path('Z:') / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'landslides') # Landslide data
cyclone_data_path = Path(pathlib.Path('Z:') / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'tropical_cyclones') # Cyclone data


########################################################################################################################
################         OSM-flex adjustments          #############################################################
########################################################################################################################
import logging
import geopandas as gpd
from osgeo import ogr, gdal
import pandas as pd
from pathlib import Path
import shapely
from tqdm import tqdm

from osm_flex.config import DICT_CIS_OSM, OSM_CONFIG_FILE


LOGGER = logging.getLogger(__name__)
DATA_DIR = '' #TODO: dito, where & how to define
gdal.SetConfigOption("OSM_CONFIG_FILE", str(OSM_CONFIG_FILE))


def _query_builder(geo_type, constraint_dict):
    """
    This function builds an SQL query from the values passed to the extract()
    function.

    Parameters
    ---------
    geo_type : str
        Type of geometry to extract. One of [points, lines, multipolygons]
    constraint_dict :  dict

    Returns
    -------
    query : str
        an SQL query string.
    """
    # columns which to report in output
    query =  "SELECT osm_id"
    for key in constraint_dict['osm_keys']:
        query+= ","+ key
    # filter condition(s)
    if constraint_dict['osm_query'] is not None:
        query+= " FROM " + geo_type + " WHERE " + constraint_dict['osm_query']
    else:
        query += " FROM " + geo_type + f" WHERE {constraint_dict['osm_keys'][0]} IS NOT NULL"
    return query

def extract(osm_path, geo_type, osm_keys, osm_query=None):
    """
    Function to extract geometries and tag info for entires in the OSM file
    matching certain OSM keys, or key-value constraints.
    from an OpenStreetMap osm.pbf file.

    Parameters
    ----------
    osm_path : str or Path
        location of osm.pbf file from which to parse
    geo_type : str
        Type of geometry to extract. One of [points, lines, multipolygons]
    osm_keys : list
        a list with all the osm keys that should be reported as columns in
        the output gdf.
    osm_query : str
        optional. query string of the syntax
        "key='value' (and/or further queries)". If left empty, all objects
        for which the first entry of osm_keys is not Null will be parsed.
        See examples in DICT_CIS_OSM in case of doubt.

    Returns
    -------
    gpd.GeoDataFrame
        A gdf with all results from the osm.pbf file matching the
        specified constraints.

    Note
    ----
    1) The keys that are searchable are specified in the osmconf.ini file.
    Make sure that they exist in the attributes=... paragraph under the
    respective geometry section.
    For example, to extract multipolygons with building='yes',
    building must be in the attributes under
    the [multipolygons] section of the file. You can find it in the same
    folder as the osm_dataloader.py module is located.
    2) OSM keys that have : in their name must be changed to _ in the
    search dict, but not in the osmconf.ini
    E.g. tower:type is called tower_type, since it would interfere with the
    SQL syntax otherwise, but still tower:type in the osmconf.ini
    3) If the osm_query is left empty (None), then all objects will be parsed
    for which the first entry of osm_keys is not Null. E.g. if osm_keys =
    ['building', 'name'] and osm_query = None, then all items matching
    building=* will be parsed.

    See also
    --------
    https://taginfo.openstreetmap.org/ to check what keys and key/value
    pairs are valid.
    https://overpass-turbo.eu/ for a direct visual output of the query,
    and to quickly check the validity. The wizard can help you find the
    correct keys / values you are looking for.
    """
    if not Path(osm_path).is_file():
        raise ValueError(f"the given path is not a file: {osm_path}")

    osm_path = str(osm_path)
    constraint_dict = {
        'osm_keys' : osm_keys,
        'osm_query' : osm_query}

    driver = ogr.GetDriverByName('OSM')
    data = driver.Open(osm_path)
    query = _query_builder(geo_type, constraint_dict)
    LOGGER.debug("query: %s", query)
    sql_lyr = data.ExecuteSQL(query)
    features = []
    geometry = []
    if data is not None:
        LOGGER.info('query is finished, lets start the loop')
        for feature in tqdm(sql_lyr, desc=f'extract {geo_type}'):
            try:
                wkb = feature.geometry().ExportToWkb()
                geom = shapely.wkb.loads(bytes(wkb))
                if geom is None:
                    continue
                geometry.append(geom)
                fields = [
                    feature.GetField(key)
                    for key in ["osm_id", *constraint_dict["osm_keys"]]
                ]
                features.append(fields)
            except Exception as exc:
                LOGGER.info('%s - %s', exc.__class__, exc)
                LOGGER.warning("skipped OSM feature")
    else:
        LOGGER.error("""Nonetype error when requesting SQL. Check the
                     query and the OSM config file under the respective
                     geometry - perhaps key is unknown.""")

    return gpd.GeoDataFrame(
        features,
        columns=["osm_id", *constraint_dict['osm_keys']],
        geometry=geometry,
        crs="epsg:4326"
    )

# TODO: decide on name of wrapper, which categories included & what components fall under it.
def extract_cis(osm_path, ci_type):
    """
    A wrapper around extract() to conveniently extract map info for a
    selection of  critical infrastructure types from the given osm.pbf file.
    No need to search for osm key/value tags and relevant geometry types.
    Parameters
    ----------
    osm_path : str or Path
        location of osm.pbf file from which to parse
    ci_type : str
        one of DICT_CIS_OSM.keys(), i.e. 'education', 'healthcare',
        'water', 'telecom', 'road', 'rail', 'air', 'gas', 'oil', 'power',
        'wastewater', 'food'
    See also
    -------
    DICT_CIS_OSM for the keys and key/value tags queried for the respective
    CIs. Modify if desired.
    """
    # features consisting in points and multipolygon results:
    if ci_type in ['healthcare','education','food','buildings']:
        gdf = pd.concat([
            extract(osm_path, 'points', DICT_CIS_OSM[ci_type]['osm_keys'],
                    DICT_CIS_OSM[ci_type]['osm_query']),
            extract(osm_path, 'multipolygons', DICT_CIS_OSM[ci_type]['osm_keys'],
                    DICT_CIS_OSM[ci_type]['osm_query'])
            ])

    # features consisting in points, multipolygons and lines:
    elif ci_type in ['gas','oil', 'water','power']:
        gdf =  pd.concat([
            extract(osm_path, 'points', DICT_CIS_OSM[ci_type]['osm_keys'],
                    DICT_CIS_OSM[ci_type]['osm_query']),
            extract(osm_path, 'multipolygons', DICT_CIS_OSM[ci_type]['osm_keys'],
                             DICT_CIS_OSM[ci_type]['osm_query']),
            extract(osm_path, 'lines', DICT_CIS_OSM[ci_type]['osm_keys'],
                             DICT_CIS_OSM[ci_type]['osm_query'])
            ])

    # features consisting in multipolygons and lines:
    elif ci_type in ['air']:
        gdf =  pd.concat([
            extract(osm_path, 'multipolygons', DICT_CIS_OSM[ci_type]['osm_keys'],
                             DICT_CIS_OSM[ci_type]['osm_query']),
            extract(osm_path, 'lines', DICT_CIS_OSM[ci_type]['osm_keys'],
                             DICT_CIS_OSM[ci_type]['osm_query'])
            ])
    
    # features consisting in multiple datattypes, but only lines needed:
    elif ci_type in ['rail','road', 'main_road']:
        gdf =  pd.concat([
            extract(osm_path, 'lines', 
                    DICT_CIS_OSM[ci_type]['osm_keys'],
                    DICT_CIS_OSM[ci_type]['osm_query'])
            ])


    # features consisting in all data types, but only points and multipolygon needed:
    elif ci_type in ['telecom','wastewater','waste_solid','waste_water','water_supply']:
        gdf = pd.concat([
            extract(osm_path, 'points', DICT_CIS_OSM[ci_type]['osm_keys'],
                    DICT_CIS_OSM[ci_type]['osm_query']),
            extract(osm_path, 'multipolygons', DICT_CIS_OSM[ci_type]['osm_keys'],
                    DICT_CIS_OSM[ci_type]['osm_query'])
            ])
        
    else:
        LOGGER.warning('feature not in DICT_CIS_OSM. Returning empty gdf')
        gdf = gpd.GeoDataFrame()
    return gdf

DICT_CIS_OSM =  {
        'power' : {
              'osm_keys' : ['power','voltage','name'],
              'osm_query' : """power='line' or power='cable' or
                               power='minor_line' or power='minor_cable' or
                               power='plant' or power='generator' or
                               power='substation' or power='tower' or
                               power='pole' or power='portal'"""},
        'road' :  {
            'osm_keys' : ['highway','name','maxspeed','lanes','surface'],
            'osm_query' : """highway in ('motorway', 'motorway_link', 'motorway_junction', 'trunk', 'trunk_link',
                            'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 
                            'residential', 'road', 'unclassified', 'living_street', 'pedestrian', 'bus_guideway', 'escape', 'raceway', 
                            'cycleway', 'construction', 'bus_stop', 'crossing', 'mini_roundabout', 'passing_place', 'rest_area', 
                            'turning_circle', 'traffic_island', 'yes', 'emergency_bay', 'service')"""},
        'rail' : {
            'osm_keys' : ['railway','name','gauge','electrified','voltage'],
            'osm_query' : """railway='rail' or railway='narrow_gauge'"""},
         'air' : {
             'osm_keys' : ['aeroway','name'],
             'osm_query' : """aeroway='aerodrome' or aeroway='terminal' or aeroway='runway'"""}, 
        'telecom' : {
            'osm_keys' : ['man_made','tower_type','name'],
            'osm_query' : """tower_type='communication' or man_made='mast' or man_made='communications_tower'"""},
        'water_supply' : {
            'osm_keys' : ['man_made','name'],
            'osm_query' : """man_made='water_well' or man_made='water_works' or
                             man_made='water_tower' or
                             man_made='reservoir_covered' or
                             (man_made='storage_tank' and content='water')"""},
        'waste_solid' : {
              'osm_keys' : ['amenity','name'],
              'osm_query' : """amenity='waste_transfer_station'"""},
        'waste_water' : {
              'osm_keys' : ['man_made','name'],
              'osm_query' : """man_made='wastewater_plant'"""},
        'education' : {
            'osm_keys' : ['amenity','building','name'],
            'osm_query' : """building='school' or amenity='school' or
                             building='kindergarten' or 
                             amenity='kindergarten' or
                             building='college' or amenity='college' or
                             building='university' or amenity='university' or
                             building='library' or amenity='library'"""},
        'healthcare' : {
            'osm_keys' : ['amenity','building','healthcare','name'],
            'osm_query' : """amenity='hospital' or healthcare='hospital' or
                             building='hospital' or building='clinic' or
                             amenity='clinic' or healthcare='clinic' or 
                             amenity='doctors' or healthcare='doctors' or
                             amenity='dentist' or amenity='pharmacy' or 
                             healthcare='pharmacy' or healthcare='dentist' or
                             healthcare='physiotherapist' or healthcare='alternative' or 
                             healthcare='laboratory' or healthcare='optometrist' or 
                             healthcare='rehabilitation' or healthcare='blood_donation' or
                             healthcare='birthing_center'
                             """},
        'power_original' : {
              'osm_keys' : ['power','voltage','utility','name'],
              'osm_query' : """power='line' or power='cable' or
                               power='minor_line' or power='plant' or
                               power='generator' or power='substation' or
                               power='transformer' or
                               power='pole' or power='portal' or 
                               power='tower' or power='terminal' or 
                               power='switch' or power='catenary_mast' or
                               utility='power'"""},
         'gas' : {
             'osm_keys' : ['man_made','pipeline', 'utility','name'],
             'osm_query' : """(man_made='pipeline' and substance='gas') or
                              (pipeline='substation' and substance='gas') or
                              (man_made='storage_tank' and content='gas') or
                              utility='gas'"""},
        'oil' : {
             'osm_keys' : ['pipeline','man_made','amenity','name'],
             'osm_query' : """(pipeline='substation' and substance='oil') or
                              (man_made='pipeline' and substance='oil') or
                              man_made='petroleum_well' or 
                              man_made='oil_refinery' or
                              amenity='fuel'"""},
        'main_road' :  {
            'osm_keys' : ['highway','name','maxspeed','lanes','surface'],
            'osm_query' : """highway in ('primary', 'primary_link', 'secondary',
                             'secondary_link', 'tertiary', 'tertiary_link', 'trunk', 'trunk_link', 
                             'motorway', 'motorway_link')
                            """},
        'wastewater' : {
              'osm_keys' : ['man_made','amenity',
                            'name'],
              'osm_query' : """amenity='waste_transfer_station' or man_made='wastewater_plant'"""},
         'food' : {
             'osm_keys' : ['shop','name'],
             'osm_query' : """shop='supermarket' or shop='greengrocer' or
                              shop='grocery' or shop='general' or 
                              shop='bakery'"""},             
        'buildings' : {
            'osm_keys' : ['building','amenity','name'],
            'osm_query' : """building='yes' or building='house' or 
                            building='residential' or building='detached' or 
                            building='hut' or building='industrial' or 
                            building='shed' or building='apartments'"""}
                              }

########################################################################################################################
################         functions          #############################################################
########################################################################################################################

def country_download(iso3):
    """
    Download OpenStreetMap data for a specific country.
    Arguments:
        *iso3* (str): ISO 3166-1 alpha-3 country code.
    Returns:
        *Path*: The file path of the downloaded OpenStreetMap data file.
    """
    
    dl.get_country_geofabrik(iso3) # Use the download library to get the geofabrik data for the specified country
    data_loc = OSM_DATA_DIR.joinpath(f'{DICT_GEOFABRIK[iso3][1]}-latest.osm.pbf') # Specify the location of the OpenStreetMap (OSM) data file
    return data_loc

def overlay_hazard_assets(df_ds,assets):
    """
    Overlay hazard assets on a dataframe of spatial geometries.
    Arguments:
        *df_ds*: GeoDataFrame containing the spatial geometries of the hazard data. 
        *assets*: GeoDataFrame containing the infrastructure assets.
    Returns:
        *geopandas.GeoSeries*: A GeoSeries containing the spatial geometries of df_ds that intersect with the infrastructure assets.
    """
    
    #overlay 
    hazard_tree = shapely.STRtree(df_ds.geometry.values)
    if (shapely.get_type_id(assets.iloc[0].geometry) == 3) | (shapely.get_type_id(assets.iloc[0].geometry) == 6): # id types 3 and 6 stand for polygon and multipolygon
        return  hazard_tree.query(assets.geometry,predicate='intersects')    
    else:
        return  hazard_tree.query(assets.buffered,predicate='intersects')

def buffer_assets(assets,buffer_size=0.00083):
    """
    Buffer spatial assets in a GeoDataFrame.
    Arguments:
        *assets*: GeoDataFrame containing spatial geometries to be buffered.
        *buffer_size* (float, optional): The distance by which to buffer the geometries. Default is 0.00083.
    Returns:
        *GeoDataFrame*: A new GeoDataFrame with an additional 'buffered' column containing the buffered geometries.
    """
    assets['buffered'] = shapely.buffer(assets.geometry.values,distance=buffer_size)
    return assets

def get_damage_per_asset(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam_asset,unit_maxdam):
    """
    Calculate damage for a given asset based on hazard information.
    Arguments:
        *asset*: Tuple containing information about the asset. It includes:
            - Index or identifier of the asset (asset[0]).
            - The specific hazard points in which asset is exposed (asset[1]['hazard_point']).
        *hazard_numpified*: NumPy array representing hazard information.
        *asset_geom*: Shapely geometry representing the spatial coordinates of the asset.
        *hazard_intensity*: NumPy array representing the hazard intensities of the curve for the asset type.
        *fragility_values*: NumPy array representing the damage factors of the curve for the asset type.
        *maxdam_asset*: Maximum damage value for asset.
        *unit_maxdam*: The unit of maximum damage value for asset.
    Returns:
        *float*: The calculated damage for the specific asset.
    """
     
    # find the exact hazard overlays:
    get_hazard_points = hazard_numpified[asset[1]['hazard_point'].values] 
    get_hazard_points[shapely.intersects(get_hazard_points[:,1],asset_geom)]

    # estimate damage
    if len(get_hazard_points) == 0: # no overlay of asset with hazard
        return 0
    
    else:
        if asset_geom.geom_type == 'LineString':
            overlay_meters = shapely.length(shapely.intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell
            return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_meters*maxdam_asset) #return asset number, total damage for asset number (damage factor * meters * max. damage)
        elif asset_geom.geom_type in ['MultiPolygon','Polygon']:
            overlay_m2 = shapely.area(shapely.intersection(get_hazard_points[:,1],asset_geom))
            if '/unit' in unit_maxdam:
                converted_maxdam = maxdam_asset / shapely.area(asset_geom) #convert to maxdam/m2
                return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_m2*converted_maxdam)
            else:
                return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_m2*maxdam_asset) #return asset number, total damage for asset number (damage factor * meters * max. damage)
        elif asset_geom.geom_type == 'Point':
            return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*maxdam_asset)

def get_damage_per_asset_og(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam_asset):
    """
    Calculate damage for a given asset based on hazard information.
    Arguments:
        *asset*: Tuple containing information about the asset. It includes:
            - Index or identifier of the asset (asset[0]).
            - The specific hazard points in which asset is exposed (asset[1]['hazard_point']).
        *hazard_numpified*: NumPy array representing hazard information.
        *asset_geom*: Shapely geometry representing the spatial coordinates of the asset.
        *hazard_intensity*: NumPy array representing the hazard intensities of the curve for the asset type.
        *fragility_values*: NumPy array representing the damage factors of the curve for the asset type.
        *maxdam_asset*: Maximum damage value for asset.
    Returns:
        *float*: The calculated damage for the specific asset.
    """
     
    # find the exact hazard overlays:
    get_hazard_points = hazard_numpified[asset[1]['hazard_point'].values] 
    get_hazard_points[shapely.intersects(get_hazard_points[:,1],asset_geom)]

    # estimate damage
    if len(get_hazard_points) == 0: # no overlay of asset with hazard
        return 0
    
    else:
        if asset_geom.geom_type == 'LineString':
            overlay_meters = shapely.length(shapely.intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell
            return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_meters*maxdam_asset) #return asset number, total damage for asset number (damage factor * meters * max. damage)
        elif asset_geom.geom_type in ['MultiPolygon','Polygon']:
            overlay_m2 = shapely.area(shapely.intersection(get_hazard_points[:,1],asset_geom))
            return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_m2*maxdam_asset) #return asset number, total damage for asset number (damage factor * meters * max. damage)
        elif asset_geom.geom_type == 'Point':
            return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*maxdam_asset)

def create_pathway_dict(data_path, flood_data_path, eq_data_path, landslide_data_path, cyclone_data_path): 

    """
    Create a dictionary containing paths to various hazard datasets.
    Arguments:
        *data_path* (Path): Base directory path for general data.
        *flood_data_path* (Path): Path to flood hazard data.
        *eq_data_path* (Path): Path to earthquake hazard data.
        *landslide_data_path* (Path): Path to landslide hazard data.
        *cyclone_data_path* (Path): Path to tropical cyclone hazard data.
    Returns:
        *dict*: A dictionary where keys represent a general pathway and different hazard types and values are corresponding paths.
    """

    #create a dictionary
    pathway_dict = {'data_path': data_path, 
                    'fluvial': flood_data_path, 
                    'pluvial': flood_data_path, 
                    'windstorm': cyclone_data_path, 
                    'earthquake': eq_data_path, 
                    'landslides': landslide_data_path,}

    return pathway_dict

def read_hazard_data(hazard_data_path,data_path,hazard_type,ISO3):
    """
    Read hazard data files for a specific hazard type.
    Arguments:
        *hazard_data_path* (Path): Base directory path where hazard data is stored.
        *hazard_type* (str): Type of hazard for which data needs to be read ('fluvial', 'pluvial', 'windstorm', 'earthquake', 'landslides').
    
    Returns:
        *list*: A list of Path objects representing individual hazard data files for the specified hazard type.
    """  

    country_df = pd.read_excel(data_path / 'global_information_advanced_fathom_check.xlsx',sheet_name = 'Sheet1') # finalize this file and adjust name
    fathom_code = country_df.loc[country_df['ISO_3digit'] == country_code, 'Fathom_countries'].item()

    if hazard_type == 'fluvial':
        hazard_data = hazard_data_path / fathom_code / 'fluvial_undefended' 
        return list(hazard_data.iterdir())

    elif hazard_type == 'pluvial':
        hazard_data = hazard_data_path / fathom_code / 'pluvial' 
        return list(hazard_data.iterdir())
    
    elif hazard_type == 'windstorm':
        hazard_data = hazard_data_path 
        return list(hazard_data.iterdir())

    elif hazard_type == 'earthquake':
        hazard_data = hazard_data_path
        return list(hazard_data.iterdir())

    elif hazard_type == 'landslides':
        hazard_data = hazard_data_path 
        return list(hazard_data.iterdir())

def read_vul_maxdam_orginal(data_path,hazard_type,infra_type):
    """
    Read vulnerability curves and maximum damage data for a specific hazard and infrastructure type.
    Arguments:
        *data_path*: The base directory path where vulnerability and maximum damage data files are stored.
        *hazard_type*: The type of hazard in string format, such as 'pluvial', 'fluvial', or 'windstorm'.
        *infra_type*: The type of infrastructure in string format for which vulnerability curves and maximum damage data are needed.
    
    Returns:
        *tuple*: A tuple containing two DataFrames:
            - The first DataFrame contains vulnerability curves specific to the given hazard and infrastructure type.
            - The second DataFrame contains maximum damage data for the specified infrastructure type.
    """

    vul_data = data_path / 'Vulnerability'
    
    # Load assumptions file containing curve - maxdam combinations per infrastructure type
    assumptions = pd.read_excel(vul_data / 'S1_Assumptions_Test.xlsx',sheet_name = 'Flooding assumptions',header=[1])
    assumptions['Infrastructure type'] = assumptions['Infrastructure type'].str.lower()
    if "_" in infra_type: infra_type = infra_type.replace('_', ' ')
    assump_infra_type = assumptions[assumptions['Infrastructure type'] == infra_type]
    assump_curves = ast.literal_eval(assump_infra_type['Vulnerability ID number'].item())
    assump_maxdams = ast.literal_eval(assump_infra_type['Maximum damage ID number'].item())
    
    # Get curves
    if hazard_type in ['pluvial','fluvial']:  
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_converted.xlsx',sheet_name = 'F_Vuln_Depth',index_col=[0],header=[0,1,2,3,4])
    elif hazard_type == 'windstorm':
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_converted.xlsx',sheet_name = 'W_Vuln_V10m',index_col=[0],header=[0,1,2,3,4])
    
    infra_curves =  curves[assump_curves]
    
    # get maxdam
    maxdam = pd.read_excel(vul_data / 'Table_D3_Costs_V1.1.0_converted.xlsx', sheet_name='Cost_Database',index_col=[0])
    infra_maxdam = maxdam[maxdam.index.isin(assump_maxdams)]['Amount'].dropna()
    infra_maxdam = infra_maxdam[pd.to_numeric(infra_maxdam, errors='coerce').notnull()]

    return infra_curves,infra_maxdam

def read_vul_maxdam(data_path,hazard_type,infra_type):
    """
    Read vulnerability curves and maximum damage data for a specific hazard and infrastructure type.
    Arguments:
        *data_path*: The base directory path where vulnerability and maximum damage data files are stored.
        *hazard_type*: The type of hazard in string format, such as 'pluvial', 'fluvial', or 'windstorm'.
        *infra_type*: The type of infrastructure in string format for which vulnerability curves and maximum damage data are needed.
    
    Returns:
        *tuple*: A tuple containing two DataFrames:
            - The first DataFrame contains vulnerability curves specific to the given hazard and infrastructure type.
            - The second DataFrame contains maximum damage data for the specified infrastructure type.
    """

    vul_data = data_path / 'Vulnerability'
    
    # Load assumptions file containing curve - maxdam combinations per infrastructure type
    assumptions = pd.read_excel(vul_data / 'S1_Assumptions_Test.xlsx',sheet_name = 'Flooding assumptions',header=[1])
    assumptions['Infrastructure type'] = assumptions['Infrastructure type'].str.lower()
    if "_" in infra_type: infra_type = infra_type.replace('_', ' ')
    assump_infra_type = assumptions[assumptions['Infrastructure type'] == infra_type]
    assump_curves = ast.literal_eval(assump_infra_type['Vulnerability ID number'].item())
    assump_maxdams = ast.literal_eval(assump_infra_type['Maximum damage ID number'].item())
    
    # Get curves
    if hazard_type in ['pluvial','fluvial']:  
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0.xlsx',sheet_name = 'F_Vuln_Depth',index_col=[0],header=[0,1,2,3,4])
    elif hazard_type == 'windstorm':
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0.xlsx',sheet_name = 'W_Vuln_V10m',index_col=[0],header=[0,1,2,3,4])
    
    infra_curves =  curves[assump_curves]
    
    # get maxdam
    maxdam = pd.read_excel(vul_data / 'Table_D3_Costs_V1.1.0_converted.xlsx', sheet_name='Cost_Database',index_col=[0])
    infra_costs = maxdam[maxdam.index.isin(assump_maxdams)][['Amount', 'Unit']].dropna(subset=['Amount'])
    infra_maxdam = infra_costs['Amount'][pd.to_numeric(infra_costs['Amount'], errors='coerce').notnull()]
    infra_units = infra_costs['Unit'].filter(items=list(infra_maxdam.index), axis=0)

    return infra_curves,infra_maxdam,infra_units

def read_flood_map(flood_map_path,diameter_distance=0.00083/2):
    """
    Read flood map data from a NetCDF file and process it into a GeoDataFrame.
    Arguments:
        *flood_map_path* (Path): Path to the NetCDF file containing flood map data.
        *diameter_distance* (float, optional): The diameter distance used for creating square geometries around data points. Default is 0.00083/2.
    
    Returns:
        *geopandas.GeoDataFrame*: A GeoDataFrame representing the processed flood map data.
    """
    
    flood_map = xr.open_dataset(flood_map_path, engine="rasterio")

    flood_map_vector = flood_map['band_data'].to_dataframe().reset_index() #transform to dataframe
    
    #remove data that will not be used
    flood_map_vector = flood_map_vector.loc[(flood_map_vector.band_data > 0) & (flood_map_vector.band_data < 100)]
    
    # create geometry values and drop lat lon columns
    flood_map_vector['geometry'] = [shapely.points(x) for x in list(zip(flood_map_vector['x'],flood_map_vector['y']))]
    flood_map_vector = flood_map_vector.drop(['x','y','band','spatial_ref'],axis=1)
    
    # drop all non values to reduce size
    flood_map_vector = flood_map_vector.loc[~flood_map_vector['band_data'].isna()].reset_index(drop=True)
    
    # and turn them into squares again:
    flood_map_vector.geometry= shapely.buffer(flood_map_vector.geometry,distance=diameter_distance,cap_style='square').values 

    return flood_map_vector

def read_windstorm_map(windstorm_map_path,bbox):
     
    # load data from NetCDF file
    with xr.open_dataset(flood_map_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        #ds['band_data'] = ds['band_data']/0.88*1.11 #convert 10-min sustained wind speed to 3-s gust wind speed
    
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data < 100)]
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)
        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=0.1/2, cap_style='square').values
    
        return ds_vector

def combine_columns(a, b):
    """
    Combine values from two input arguments 'a' and 'b' into a single string.
    Arguments:
    - a (str or None): Value from column 'A'.
    - b (str or None): Value from column 'B'.

    Returns:
    - str or None: A string of 'a', 'b' or combination. If both 'a' and 'b' are None, return None.
    """
    
    if pd.notna(a) and pd.notna(b) == False: #if only a contains a string
        return f"{a}" 
    elif pd.notna(b) and pd.notna(a) == False: #if only b contains a string
        return f"{b}"
    elif pd.notna(a) and pd.notna(b):  #if both values contain a string
        if a == b: 
            return f"{a}"
        elif a == 'yes' or b == 'yes':
            if a == 'yes':
                return  f"{b}"
            elif b == 'yes':
                return  f"{a}"
        else: 
            return f"{a}" #f"{a}_{b}" # assuming that value from column A contains the more detailed information
    else: 
        None # Decision point: If nones are existent, decide on what to do with Nones. Are we sure that these are education facilities? Delete them? Provide another tag to them?

def filter_dataframe(assets, column_names_lst):
    """
    Filter a GeoDataFrame by combining information from two specified columns and removing selected columns.
    Args:
        assets (geopandas.GeoDataFrame): The input GeoDataFrame containing spatial geometries and columns to filter.
        column_names_lst (list): A list of two column names whose information needs to be combined to create a new 'asset' column.

    Returns:
        geopandas.GeoDataFrame: A filtered GeoDataFrame with a new 'asset' column and selected columns dropped, and points converted to polygons.
    """

    if len(column_names_lst) == 2:        
        assets['asset'] = assets.apply(lambda row: combine_columns(row[column_names_lst[0]], row[column_names_lst[1]]), axis=1) # create new column based on tag information provided in two columns
    elif len(column_names_lst) == 3:
        assets['asset_temp'] = assets.apply(lambda row: combine_columns(row[column_names_lst[0]], row[column_names_lst[1]]), axis=1) # create temp column based on tag information provided in two columns
        assets['asset'] = assets.apply(lambda row: combine_columns(row['asset_temp'], row[column_names_lst[2]]), axis=1) # create new column based on tag information provided in two columns
        column_names_lst.append('asset_temp')        
    else:
        print("Warning: column_names_lst should contain 2 or 3 items")

    assets = assets.drop(columns=column_names_lst, axis=1) # drop columns
    assets = remove_contained_assets_and_convert(assets)
    
    return assets

def delete_linestring_data(assets, infra_lst):
    """
    Filter and update a GeoDataFrame by excluding rows with LineString geometries.

    Parameters:
    - assets (geopandas.GeoDataFrame): The original GeoDataFrame.
    - infra_lst (lst): A list with the infrastructure typs to filter.

    Returns:
    - geopandas.GeoDataFrame: The updated GeoDataFrame with excluded LineString rows.
    """

    for infra_type in infra_lst:
        #create subset of data
        condition = assets['asset'] == infra_type
        subset = assets[condition]
        
        #delete line data if there is line data (assuming that this function is only for point and polygon data)
        subset = subset[subset['geometry'].geom_type.isin(['Point', 'MultiPoint', 'Polygon', 'MultiPolygon'])]  # Keep (multi-) points and polygon geometries
    
        #update the original Dataframe by excluding rows in the subset
        assets = assets[~condition | condition & subset['geometry'].notna()]

    return assets

def delete_point_and_polygons(assets, infra_lst):
    """
    Filter and update a GeoDataFrame by excluding rows with points and (multi-)polygon geometries.

    Parameters:
    - assets (geopandas.GeoDataFrame): The original GeoDataFrame.
    - infra_lst (lst): A list with the infrastructure typs to filter.

    Returns:
    - geopandas.GeoDataFrame: The updated GeoDataFrame with excluded points and (multi-)polygon rows.
    """

    for infra_type in infra_lst:
        #create subset of data
        condition = assets['asset'] == infra_type
        subset = assets[condition]
        
        #delete points and (multi-)polygon data if available
        subset = subset[subset['geometry'].geom_type.isin(['LineString', 'MultiLineString'])]  # Keep only LineString geometries


        #update the original Dataframe by excluding rows in the subset
        assets = assets[~condition | condition & subset['geometry'].notna()]

    return assets

def remove_polygons_with_contained_points(gdf):
    """
    Remove polygons in a GeoDataFrame if there is a point falling within them.
    Arguments:
        gdf : GeoDataFrame containing entries with point and (multi-)polygon geometry
    Returns:
    - geopandas.GeoDataFrame: GeoDataFrame containing entries with point and (multi-)polygon geometry, but without duplicates
    """
    gdf = gdf.reset_index(drop=True)
    
    ind_poly_with_points = np.unique(gpd.sjoin(gdf[gdf.geometry.type == 'Point'],
                                              gdf[gdf.geometry.type.isin(['MultiPolygon', 'Polygon'])],
                                              predicate='within').index_right)
    
    return gdf.drop(index=ind_poly_with_points).reset_index(drop=True)


def remove_contained_assets_and_convert(assets):
    """
    Process the geometry of assets, removing contained points and polygons, and converting points to polygons.
    Args:
        assets (geopandas.GeoDataFrame): Input GeoDataFrame containing asset geometries.

    Returns:
        geopandas.GeoDataFrame: Processed GeoDataFrame with updated asset geometries.
    """
    
    assets =  remove_contained_polys(remove_contained_points(assets)) #remove points and polygons within a (larger) polygon
    
    #convert points to polygons
    if (assets.loc[assets.geom_type == 'MultiPolygon']).empty:
        default_distance = 58.776
        assets.loc[assets.geom_type == 'Point','geometry'] = assets.loc[assets.geom_type == 'Point'].buffer(distance=default_distance, cap_style='square')
    else:    
        assets.loc[assets.geom_type == 'Point','geometry'] = assets.loc[assets.geom_type == 'Point'].buffer(
                                                                        distance=np.sqrt(assets.loc[assets.geom_type == 'MultiPolygon'].area.median())/2, cap_style='square')

    return assets

def create_point_from_polygon(gdf):
    """
    Transforms polygons into points
    Arguments:
        gdf: A geodataframe containing a column geometry
    Returns:
    - geopandas.GeoDataFrame: The updated GeoDataFrame without polygons but with only point geometries
    """
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: MultiPolygon([geom]) if geom.geom_type == 'Polygon' else geom) #convert to multipolygons in case polygons are in the df
    #gdf.loc[gdf.geom_type == 'MultiPolygon','geometry'] = gdf.loc[assets.geom_type == 'MultiPolygon'].centroid #convert polygon to point
    gdf.loc[gdf.geom_type == 'MultiPolygon','geometry'] = gdf.loc[gdf.geom_type == 'MultiPolygon'].centroid #convert polygon to point
    return gdf
    
def process_selected_assets(gdf, polygon_types, point_types):
    """
    Process the geometry of selected assets, removing contained points and polygons, and converting non-contained points to polygons.
    Args:
        gdf (geopandas.GeoDataFrame): Input GeoDataFrame containing asset geometries.
        selected_types (list): List of asset types to process.

    Returns:
        geopandas.GeoDataFrame: Processed GeoDataFrame with updated asset geometries.
    """
    asset_temp = gdf['asset'].tolist()
    gdf.insert(1, 'asset_temp', asset_temp) 
    
    # For assets that we need as (multi-)polygons: group by asset type and apply the processing function
    filtered_assets = gdf[gdf['asset'].isin(polygon_types)] # Filter only selected asset types
    polygon_gdf = (filtered_assets.groupby('asset_temp').apply(remove_contained_assets_and_convert, include_groups=False)).reset_index(drop=True)

    # For assets that we need as (multi-)points: group by asset type and apply the processing function
    filtered_assets = gdf[gdf['asset'].isin(point_types)] # Filter only selected asset types
    #point_gdf = (filtered_assets.groupby('asset').apply(create_point_from_polygon)).reset_index(drop=True)
    point_gdf = (filtered_assets.groupby('asset_temp').apply(lambda group: create_point_from_polygon(remove_polygons_with_contained_points(group)), include_groups=False)).reset_index(drop=True)
    
    # Concatenate the two dataframes along rows
    merged_gdf = pd.concat([polygon_gdf, point_gdf], ignore_index=True)
    
    return merged_gdf

def create_damage_csv(damage_output, hazard_type, pathway_dict, country_code, sub_system):
    """
    Create a CSV file containing damage information.
    Arguments:
        damage_output: A dictionary containing damage information.
        hazard_type: The type of hazard (e.g., 'earthquake', 'flood').
        pathway_dict: A dictionary containing file paths for different data.
        country_code: A string containing information about the country code
        sub_system: A string containing information about the subsystem considered

    Returns:
        None
    """
  
    hazard_output_path = pathway_dict['data_path'] / 'damage' / country_code
    
    # Check if the directory exists
    if not hazard_output_path.exists():
        # Create the directory
        hazard_output_path.mkdir(parents=True, exist_ok=True)
    
    csv_file_path = hazard_output_path / '{}_{}_{}.csv'.format(country_code, hazard_type, sub_system)
    
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write header
        csv_writer.writerow(['Country', 'Return period', 'Subsystem', 'Infrastructure type', 'Curve ID number', 'Damage ID number', 'Damage', 'Exposed assets'])
        
        # Write data
        for key, value in damage_output.items():
            csv_writer.writerow(list(key) + list(value))
    
    print(f"CSV file created at: {csv_file_path}")

def country_infrastructure_hazard(pathway_dict, country_code, sub_system, infra_type_lst, hazard_type):

    # get country osm data
    data_loc = country_download(country_code)
    
    # get infrastructure data:
    print(f'Time to extract OSM data for {sub_system}')
    assets = extract_cis(data_loc, sub_system)
    
    # convert assets to epsg3857 (system in meters)
    assets = gpd.GeoDataFrame(assets).set_crs(4326).to_crs(3857)
    
    if sub_system == 'power':
        assets = assets.rename(columns={'power' : 'asset'}).reset_index(drop=True)
        
        #reclassify assets 
        mapping_dict = {
            "cable" : "cable", 
            "minor_cable" : "cable",
            "line" : "transmission_line", 
            "minor_line" : "distribution_line", 
            "plant" : "plant", 
            "generator" : "plant", 
            "substation" : "substation", 
            "tower" : "power_tower",
            "pole" : "power_pole",
            "portal" : "power_tower",
            
        }
        assets['asset'] = assets.asset.apply(lambda x : mapping_dict[x])  #reclassification

        #filter dataframe
        infra_lst = ['plant', 'substation','power_tower','power_pole']
        assets = delete_linestring_data(assets, infra_lst) #check for linestring data for specific infrastructure types and delete
        infra_lst = ['transmission_line', 'distribution_line', 'cable'] 
        assets = delete_point_and_polygons(assets, infra_lst) #check for (multi-)polygon and point data and delete

        # process geometries according to infra type
        polygon_types = ['plant', 'substation']
        point_types = ['power_tower', 'power_pole']
        assets = process_selected_assets(assets, polygon_types, point_types)
    
    elif sub_system == 'road':
        assets = assets.rename(columns={'highway' : 'asset'})
        
        #reclassify assets 
        mapping_dict = {
            "motorway" : "motorway", 
            "motorway_link" : "motorway", 
            "motorway_junction" : "motorway",
            "trunk" : "trunk",
            "trunk_link" : "trunk",
            "primary" : "primary", 
            "primary_link" : "primary", 
            "secondary" : "secondary", 
            "secondary_link" : "secondary", 
            "tertiary" : "tertiary", 
            "tertiary_link" : "tertiary", 
            "residential" : "other",           
            "road" : "other", 
            "unclassified" : "other",
            "living_street" : "other", 
            "pedestrian" : "other", 
            "bus_guideway" : "other", 
            "escape" : "other", 
            "raceway" : "other", 
            "cycleway" : "other", 
            "construction" : "other", 
            "bus_stop" : "other", 
            "crossing" : "other", 
            "mini_roundabout" : "other", 
            "passing_place" : "other", 
            "rest_area" : "other", 
            "turning_circle" : "other",
            "traffic_island" : "other",
            "yes" : "other",
            "emergency_bay" : "other",
            "service" : "other"
        }
        assets['asset'] = assets.asset.apply(lambda x : mapping_dict[x])  #reclassification
    
    elif sub_system == 'rail':
        assets = assets.rename(columns={'railway' : 'asset'})
        
        #reclassify assets 
        mapping_dict = {
            "rail" : "railway", 
            "narrow_gauge" : "railway", 
        }
        assets['asset'] = assets.asset.apply(lambda x : mapping_dict[x])  #reclassification
    
    elif sub_system == 'air':
        assets = assets.rename(columns={'aeroway' : 'asset'})   

        #reclassify assets 
        mapping_dict = {
            "aerodrome" : "airport", 
            "terminal" : "terminal",
            "runway" : "runway"
            }
        assets['asset'] = assets.asset.apply(lambda x : mapping_dict[x])  #reclassification

    elif sub_system == 'telecom':
        #filter dataframe based on conditions 
        assets = (assets[(assets['man_made'] == 'tower') & (assets['tower_type'] == 'communication') |
                (assets['man_made'] == 'mast') & (assets['tower_type'] == 'communication') |
                (assets['man_made'] == 'communications_tower')|
                (assets['man_made'] == 'mast') & (assets['tower_type'].isna())]).reset_index(drop=True)
        assets = assets.drop(['tower_type'], axis=1) #drop columns that are of no further use

        #reclassify assets
        assets = assets.rename(columns={'man_made' : 'asset'})
        mapping_dict = {
            "tower" : "communication_tower", 
            "communications_tower" : "communication_tower",
            "mast" : "mast", 
        }
        assets['asset'] = assets.asset.apply(lambda x : mapping_dict[x])  #reclassification
        
        assets = create_point_from_polygon(remove_polygons_with_contained_points(assets)) #remove duplicates and transform polygons into points

    elif sub_system == 'water_supply':
        assets = assets.reset_index(drop=True)
        assets = assets.rename(columns={'man_made' : 'asset'})
        mapping_dict = {
            "water_tower" : "water_tower",
            "water_well" : "water_well",
            "reservoir_covered" : "reservoir_covered",
            "water_works" : "water_treatment_plant",
            "storage_tank" : "water_storage_tank"
        }
        assets['asset'] = assets.asset.apply(lambda x : mapping_dict[x])  #reclassification
        
        # process geometries according to infra type
        polygon_types = ['reservoir_covered', 'water_treatment_plant']
        point_types = ['water_tower', 'water_well', 'storage_tank']
        assets = process_selected_assets(assets, polygon_types, point_types)

    elif sub_system == 'water':
        assets = assets.reset_index(drop=True)
        assets = assets.drop(assets[assets['emergency'] == 'fire_hydrant'].index).reset_index(drop=True) #drop linestrings
        assets = assets.rename(columns={'man_made' : 'asset'})   

    elif sub_system == 'waste_solid':
        assets = assets.rename(columns={'amenity' : 'asset'})
        assets = remove_contained_assets_and_convert(assets)

    elif sub_system == 'waste_water':
        assets = assets.rename(columns={'man_made' : 'asset'})
        mapping_dict = {
            "wastewater_plant" : "wastewater_treatment_plant", 
        }
        assets['asset'] = assets.asset.apply(lambda x : mapping_dict[x])  #reclassification
        assets = remove_contained_assets_and_convert(assets)

    elif sub_system == 'wastewater':
        column_names_lst = ['man_made', 'amenity']
        assets = filter_dataframe(assets, column_names_lst)
    
    elif sub_system == 'healthcare':
        column_names_lst = ['amenity' , 'building', 'healthcare']
        assets = filter_dataframe(assets, column_names_lst)
        list_of_assets_to_keep = ["clinic", "doctors", "hospital", "dentist", "pharmacy", 
                        "physiotherapist", "alternative", "laboratory", "optometrist", "rehabilitation", 
                        "blood_donation", "birthing_center"]
        assets = assets.loc[assets.asset.isin(list_of_assets_to_keep)].reset_index(drop=True)
    
    elif sub_system == 'education':
        column_names_lst = ['amenity' , 'building']
        assets = filter_dataframe(assets, column_names_lst)
        list_of_assets_to_keep =["college", "kindergarten", "library", "school", "university"]
        assets = assets.loc[assets.asset.isin(list_of_assets_to_keep)].reset_index(drop=True)
    
    # read hazard data
    hazard_data_path = pathway_dict[hazard_type]
    data_path = pathway_dict['data_path']
    hazard_data_list = read_hazard_data(hazard_data_path,data_path,hazard_type,country_code)

    # start analysis 
    print(f'{country_code} runs for {sub_system} for {hazard_type} for {len(hazard_data_list)} maps')

    if hazard_type in ['windstorm','earthquake','landslide']:
        # load country geometry file and create geometry to clip
        ne_countries = gpd.read_file(data_path / "natural_earth" / "ne_10m_admin_0_countries.shp") #https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
        bbox = ne_countries.loc[ne_countries['ISO_A3']==country_code].geometry.envelope.values[0].bounds
        
    collect_output = {}
    for single_footprint in hazard_data_list: #tqdm(hazard_data_list,total=len(hazard_data_list)):
    
        hazard_name = single_footprint.parts[-1].split('.')[0]
        
        # load hazard map
        if hazard_type in ['pluvial','fluvial']:
            hazard_map = read_flood_map(single_footprint)
        elif hazard_type in ['windstorm']:
             hazard_map = read_windstorm_map(single_footprint,bbox)
        elif hazard_type in ['earthquake']:
             hazard_map = read_earthquake_map(single_footprint)
        elif hazard_type in ['landslide']:
             hazard_map = read_landslide_map(single_footprint)
         
        # convert hazard data to epsg 3857
        hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)

        # Loop through unique infrastructure types within the subsystem
        for infra_type in infra_type_lst: 
            assets_infra_type = assets[assets['asset'] == infra_type].copy().reset_index(drop=True)
        
            # create dicts for quicker lookup
            geom_dict = assets_infra_type['geometry'].to_dict()
            type_dict = assets_infra_type['asset'].to_dict()

            # read vulnerability and maxdam data:
            infra_curves,maxdams,infra_units = read_vul_maxdam(data_path,hazard_type, infra_type)

            # start analysis 
            print(f'{country_code} runs for {infra_type} for {hazard_type} for {hazard_name} map for {len(infra_curves.T)*len(maxdams)} combinations')
    
            if not assets_infra_type.empty:
                # overlay assets
                overlay_assets = pd.DataFrame(overlay_hazard_assets(hazard_map,buffer_assets(assets_infra_type)).T,columns=['asset','hazard_point'])
            else: 
                overlay_assets = pd.DataFrame(columns=['asset','hazard_point']) #empty dataframe
    
            # convert dataframe to numpy array
            hazard_numpified = hazard_map.to_numpy() 

            for infra_curve in infra_curves:
                # get curves
                curve = infra_curves[infra_curve[0]]
                hazard_intensity = curve.index.values
                fragility_values = (np.nan_to_num(curve.values,nan=(np.nanmax(curve.values)))).flatten()

                for maxdam in maxdams:
                    collect_inb = []
                    collect_geom = []
                    unit_maxdam = infra_units[maxdams[maxdams == maxdam].index[0]] #get unit maxdam
                    
                    for asset in tqdm(overlay_assets.groupby('asset'),total=len(overlay_assets.asset.unique())): #group asset items for different hazard points per asset and get total number of unique assets
                        asset_geom = geom_dict[asset[0]]
                        collect_geom.append(asset_geom.wkt)
                        if np.max(fragility_values) == 0: #if exposure does not lead to damage
                            collect_inb.append(0)  
                        else:
                            #collect_inb.append(get_damage_per_asset_og(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam))
                            collect_inb.append(get_damage_per_asset(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam,unit_maxdam)) #get list of damages for specific asset
                    collect_output[country_code, hazard_name, sub_system, infra_type, infra_curve[0], ((maxdams[maxdams == maxdam]).index)[0]] = np.sum(collect_inb), collect_geom # dictionary to store results for various combinations of hazard maps, infrastructure curves, and maximum damage values.
    return collect_output

########################################################################################################################
################          Run analysis          #############################################################
########################################################################################################################

# List of critical infrastructure systems to process
cis_dict = {
    "energy": {"power": ["transmission_line","distribution_line","cable","plant","substation",
                        "power_tower","power_pole"]},
    "transportation": {"road":  ["motorway", "trunk", "primary", "secondary", "tertiary", "other"], 
                        "air": ["airport", "runway", "terminal"],
                        "rail": ["railway"]},
    "water": {"water_supply": ["water_tower", "water_well", "reservoir_covered",
                                "water_treatment_plant", "water_storage_tank"]},
    "waste": {"waste_solid": ["waste_transfer_station"],
            "waste_water": ["wastewater_treatment_plant"]},
    "telecommunication": {"telecom": ["communication_tower", "mast"]},
    "healthcare": {"healthcare": ["clinic", "doctors", "hospital", "dentist", "pharmacy", 
                        "physiotherapist", "alternative", "laboratory", "optometrist", "rehabilitation", 
                        "blood_donation", "birthing_center"]},
    "education": {"education": ["college", "kindergarten", "library", "school", "university"]}
}

#cis_dict = {
#    "waste": {
#            "waste_water": ["wastewater_treatment_plant"]},
#}

hazard_type='fluvial'
country_codes=['GEO', 'KAZ', 'AZE'] #PNG, TAJ, PAK

pathway_dict = create_pathway_dict(data_path, flood_data_path, eq_data_path, landslide_data_path, cyclone_data_path)
for country_code in country_codes: 
    for ci_system in cis_dict: 
        for sub_system in cis_dict[ci_system]:
            infra_type_lst = cis_dict[ci_system][sub_system]
            test = country_infrastructure_hazard(pathway_dict, country_code, sub_system, infra_type_lst, hazard_type)
            create_damage_csv(test, hazard_type, pathway_dict, country_code, sub_system)