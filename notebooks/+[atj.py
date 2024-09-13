# coding: utf-8
import os
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np 
import shapely 
import csv
import ast
import h3

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
#define paths
p = Path('..')
data_path = Path(pathlib.Path.home().parts[0]) / 'Projects' / 'gmhcira' / 'data' #should contain folder 'Vulnerability' with vulnerability data
flood_data_path = Path(pathlib.Path('Z:') / 'eks510' / 'fathom-global') # Flood data
eq_data_path = Path(pathlib.Path('Z:') / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'earthquakes' / 'GEM') # Earthquake data
#eq_data_path = Path(pathlib.Path('Z:') / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'earthquakes' / 'GAR' / 'raw') #eq data GAR
#eq_data_path = Path(pathlib.Path.home().parts[0]) / 'Users' / 'snn490' / 'OneDrive - Vrije Universiteit Amsterdam' / 'ADB' / 'Data' / 'Earthquake_data' #eq data provided by ADB
landslide_data_path = Path(pathlib.Path('Z:') / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'landslides') # Landslide data
cyclone_data_path = Path(pathlib.Path('Z:') / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'tropical_cyclones') # Cyclone data
liquefaction_data_path = Path(pathlib.Path('Z:') / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'liquefaction') # Cyclone data
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
        'road_gmhcira' :  {
            'osm_keys' : ['highway','name','maxspeed','lanes','surface'],
            'osm_query' : """highway in ('motorway', 'motorway_link', 'motorway_junction', 'trunk', 'trunk_link',
                            'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 
                            'residential', 'road', 'unclassified', 'living_street', 'pedestrian', 'bus_guideway', 'escape', 'raceway', 
                            'cycleway', 'construction', 'bus_stop', 'crossing', 'mini_roundabout', 'passing_place', 'rest_area', 
                            'turning_circle', 'traffic_island', 'yes', 'emergency_bay', 'service', 'track')"""},
        'road' :  {
            'osm_keys' : ['highway','name','maxspeed','lanes','surface'],
            'osm_query' : """highway in ('motorway', 'motorway_link', 'trunk', 'trunk_link',
                            'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary', 'tertiary_link', 
                            'residential', 'road', 'unclassified', 'track')"""},
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
    if type(maxdam_asset) == str: maxdam_asset = float(maxdam_asset)

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

def get_damage_per_asset_and_overlay(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam_asset,unit_maxdam):
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
    if type(maxdam_asset) == str: maxdam_asset = float(maxdam_asset)

    # estimate damage
    if len(get_hazard_points) == 0: # no overlay of asset with hazard
        return 0
    else:
        if asset_geom.geom_type == 'LineString':
            overlay_meters = shapely.length(shapely.intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell
            return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_meters*maxdam_asset), np.sum(overlay_meters) #return asset number, total damage for asset number (damage factor * meters * max. damage)
        elif asset_geom.geom_type in ['MultiPolygon','Polygon']:
            overlay_m2 = shapely.area(shapely.intersection(get_hazard_points[:,1],asset_geom))
            if '/unit' in unit_maxdam:
                converted_maxdam = maxdam_asset / shapely.area(asset_geom) #convert to maxdam/m2
                return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_m2*converted_maxdam)
            else:
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
                    'landslide_rf': landslide_data_path,
                    'landslide_eq': landslide_data_path,}

    return pathway_dict

def read_hazard_data(hazard_data_path,data_path,hazard_type,ISO3):
    """
    Read hazard data files for a specific hazard type.
    Arguments:
        *hazard_data_path* (Path): Base directory path where hazard data is stored.
        *hazard_type* (str): Type of hazard for which data needs to be read ('fluvial', 'pluvial', 'windstorm', 'earthquake', 'landslide').
    
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
        if 'GAR' in str(hazard_data_path):
            data_lst = []
            hazard_data = hazard_data_path
            rp_lst = list(hazard_data.iterdir())
            for rp_folder in rp_lst:
                temp_lst = [file for file in rp_folder.iterdir() if file.suffix == '.tif']
                data_lst.extend(temp_lst)  # Use extend instead of append to flatten the list
            return data_lst
        
        elif 'ADB' in str(eq_data_path):
            hazard_data = eq_data_path 
            return [file for file in hazard_data.iterdir() if file.suffix == '.tif']
            
        elif 'GEM' in str(hazard_data_path):
            hazard_data = hazard_data_path
            data_lst = list(hazard_data.iterdir())
            data_lst = [file for file in data_lst if file.suffix == '.csv']
            return data_lst

    #elif hazard_type == 'earthquake':
    #    hazard_data = hazard_data_path
    #    return list(hazard_data.iterdir())

    elif hazard_type == 'landslide_rf':
        hazard_data = hazard_data_path / 'rainfall' / '{}_l24-norm-hist.tif'.format(ISO3)
        return [hazard_data]

    elif hazard_type == 'landslide_eq':
        hazard_data = hazard_data_path.parent / 'earthquakes' / 'GEM' / 'GEM-GSHM_PGA-475y-rock_v2023' / 'v2023_1_pga_475_rock_3min.tif' #use only one rp for the triggering conditions
        #hazard_data = hazard_data_path.parent / 'earthquakes' / 'GAR' / 'raw' / 'rp_475'/ 'gar17pga475.tif' #use only one rp for the triggering conditions
        return [hazard_data]

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

def read_vul_maxdam_old(data_path,hazard_type,infra_type):
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

def read_giri_earthquake_map(earthquake_map_path,bbox,diameter_distance=0.004999972912597225316/2):
     
    # load data from NetCDF file
    with xr.open_dataset(earthquake_map_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])

        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe

        #remove data that will not be used
        ds_vector['band_data'] = ds_vector['band_data']/980
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 10)]

        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector

def read_gar_earthquake_map(earthquake_map_path,bbox,diameter_distance=0.07201440288057610328/2): 
     
    # load data from NetCDF file
    with xr.open_dataset(earthquake_map_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])

        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe

        #remove data that will not be used
        ds_vector['band_data'] = ds_vector['band_data']/980
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 10)]

        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector

def read_earthquake_map(earthquake_map_path,bbox,diameter_distance=0.05000000000000000278/2):
     
    # load data from NetCDF file
    with xr.open_dataset(earthquake_map_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 10)]
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector

def h3_to_polygon(h3_index):
    # Get the boundary of the hexagon in (lat, lon) pairs
    boundary = h3.h3_to_geo_boundary(h3_index)
    # Convert to (lon, lat) pairs and create a Polygon
    return shapely.Polygon([(lon, lat) for lat, lon in boundary])

def overlay_hazard_bbox(df_ds,bbox_geometries):
    """
    Overlay hazard assets on a dataframe of spatial geometries.
    Arguments:
        *df_ds*: GeoDataFrame containing the spatial geometries of the hazard data. 
        *boundary*: GeoDataFrame containing the infrastructure assets.
    Returns:
        *geopandas.GeoSeries*: A GeoSeries containing the spatial geometries of df_ds that intersect with the administrative boundary.
    """
    bbox_polygon = shapely.box(*bbox) #create polygon using bbox coordinates
    
    #overlay 
    hazard_tree = shapely.STRtree(df_ds.geometry.values)
    intersect_index = hazard_tree.query(bbox_polygon,predicate='intersects')
    
    return df_ds.iloc[intersect_index].reset_index(drop=True)

def read_earthquake_map_csv(earthquake_map_path,bbox):
    #using h3 geometries: https://pypi.org/project/h3/
    #example Notebooks: https://github.com/uber/h3-py-notebooks
    #more info: https://h3geo.org/docs/quickstart
     
    ds_vector = pd.read_csv(earthquake_map_path)
    for col in ds_vector.columns: 
        if col not in ['lon', 'lat']: ds_vector = ds_vector.rename(columns={col:'band_data'})
    
    #remove data that will not be used
    ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 10)]

    #create geometry values and drop lat lon columns
    ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['lon'],ds_vector['lat']))]
    
    #overlay with bbox
    ds_vector = overlay_hazard_bbox(ds_vector,bbox)
    ds_vector = ds_vector.drop(['geometry'],axis=1)

    #transform to h3 hexagons
    ds_vector['h3_codes'] = ds_vector.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], 6), axis=1) #get h3 code
    ds_vector['geometry'] = ds_vector.apply(lambda row: h3_to_polygon(row['h3_codes']), axis=1) #get hexagon geometries

    #drop columns
    ds_vector = ds_vector.drop(['lon','lat','h3_codes'],axis=1)

    return ds_vector




def read_landslide_map(landslide_map_path,bbox,diameter_distance=0.008333333333325620637/2):
     
    # load data from NetCDF file
    with xr.open_dataset(landslide_map_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 1)]
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector

def read_susceptibility_map(landslide_map_path, hazard_type, bbox,diameter_distance=0.0008333333333333522519/2):

    if hazard_type in ['landslide_eq']:
         susc_footprint = pathlib.Path(landslide_map_path).parent.parent / 'susceptibility_giri' / 'susc_earthquake_trig_cdri.tif'
    elif hazard_type in ['landslide_rf']:
         susc_footprint = pathlib.Path(landslide_map_path).parent.parent / 'susceptibility_giri' / 'susc_prec_trig_cdri.tif'
     
    # load data from NetCDF file
    with xr.open_dataset(susc_footprint) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 5)]
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector

def read_susceptibility_map_cropped(susc_path, diameter_distance=0.0008333333333333522519/2):
     
    # load data from NetCDF file
    with xr.open_dataset(susc_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 1) & (ds_vector.band_data <= 5)] #also omit class 1 in this early phase, because won't be needed anyway following table in GIRI report
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        #ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

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
  
    hazard_output_path = pathway_dict['data_path'] / 'damage' / country_code / hazard_type
    hazard_output_path.mkdir(parents=True, exist_ok=True)
    
    ## Check if the directory exists
    #if not hazard_output_path.exists():
    #    # Create the directory
    #    hazard_output_path.mkdir(parents=True, exist_ok=True)
    
    csv_file_path = hazard_output_path / '{}_{}_{}.csv'.format(country_code, hazard_type, sub_system)
    
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write header
        csv_writer.writerow(['Country', 'Return period', 'Subsystem', 'Infrastructure type', 'Curve ID number', 'Damage ID number', 'Damage', 'Exposed assets'])
        
        # Write data
        for key, value in damage_output.items():
            csv_writer.writerow(list(key) + list(value))
    
    print(f"CSV file created at: {csv_file_path}")
def create_damage_csv_without_exposure(damage_output, hazard_type, pathway_dict, country_code, sub_system):
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

    if hazard_type in ['landslide_eq', 'landslide_rf']:

        hazard_output_path = pathway_dict['data_path'] / 'damage' / country_code
        hazard_output_path.mkdir(parents=True, exist_ok=True)
        
        ## Check if the directory exists
        #if not hazard_output_path.exists():
        #    # Create the directory
        #    hazard_output_path.mkdir(parents=True, exist_ok=True)
        
        csv_file_path = hazard_output_path / '{}_{}_{}.csv'.format(country_code, hazard_type, sub_system)
        
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write header
            csv_writer.writerow(['Country', 'Return period landslide', 'Return period hazard trigger', 'Subsystem', 'Infrastructure type', 'Curve ID number or assumption', 'Damage ID number', 'Damage'])
            
            # Write data
            for key, value in damage_output.items():
                # Extract values from key dictionary
                country = key[0]
                return_period = key[1]
                return_period_tr = key[2]
                subsystem = key[3]
                infrastructure_type = key[4]
                curve_id_number = key[5]
                damage_id_number = key[6]
                damage = value
    
                # Write row to CSV
                csv_writer.writerow([country, return_period, return_period_tr, subsystem, infrastructure_type, curve_id_number, damage_id_number, damage])
        
        print(f"CSV file created at: {csv_file_path}")

    else:
        hazard_output_path = pathway_dict['data_path'] / 'damage' / country_code
        hazard_output_path.mkdir(parents=True, exist_ok=True)
        
        ## Check if the directory exists
        #if not hazard_output_path.exists():
        #    # Create the directory
        #    hazard_output_path.mkdir(parents=True, exist_ok=True)
        
        csv_file_path = hazard_output_path / '{}_{}_{}.csv'.format(country_code, hazard_type, sub_system)
        
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Write header
            csv_writer.writerow(['Country', 'Return period', 'Subsystem', 'Infrastructure type', 'Curve ID number or assumption', 'Damage ID number', 'Damage'])
            
            # Write data
            for key, value in damage_output.items():
                # Extract values from key dictionary
                country = key[0]
                return_period = key[1]
                subsystem = key[2]
                infrastructure_type = key[3]
                curve_id_number = key[4]
                damage_id_number = key[5]
                damage = value
    
                # Write row to CSV
                csv_writer.writerow([country, return_period, subsystem, infrastructure_type, curve_id_number, damage_id_number, damage])
        
        print(f"CSV file created at: {csv_file_path}")

def get_damage_per_asset_rp(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam_asset,unit_maxdam):
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
    get_hazard_points = get_hazard_points[shapely.intersects(get_hazard_points[:,1],asset_geom)]
    return_periods = asset[1]['return_period'].values
    for i, (point, polygon) in enumerate(get_hazard_points):
        get_hazard_points[i][0] = return_periods[i]
    if type(maxdam_asset) == str: maxdam_asset = float(maxdam_asset)

    # estimate damage
    if len(get_hazard_points) == 0: # no overlay of asset with hazard
        return np.empty(0)
    else:
        if asset_geom.geom_type == 'LineString':
            overlay_meters = shapely.length(shapely.intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell
            damage = np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * overlay_meters * maxdam_asset
            return np.vstack([damage, get_hazard_points[:,0]])

        elif asset_geom.geom_type in ['MultiPolygon','Polygon']:
            overlay_m2 = shapely.area(shapely.intersection(get_hazard_points[:,1],asset_geom))
            if '/unit' in unit_maxdam:
                converted_maxdam = maxdam_asset / shapely.area(asset_geom) #convert to maxdam/m2
                damage = (np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * overlay_m2 * converted_maxdam)
                return np.vstack([damage, get_hazard_points[:,0]])
            else:
                damage = (np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * overlay_m2 * maxdam_asset)
                return np.vstack([damage, get_hazard_points[:,0]])
                    
        elif asset_geom.geom_type == 'Point':
            damage = (np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * maxdam_asset)
            return np.vstack([damage, get_hazard_points[:,0]])

def get_damage_per_asset_all_hazards(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam_asset,unit_maxdam):
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
    get_hazard_points = get_hazard_points[shapely.intersects(get_hazard_points[:,1],asset_geom)]

    if 'return_period' in asset[1].columns: #if there is no intensity value in the map, but the return periods are provided 
        return_periods = asset[1]['return_period'].values
        for i, (point, polygon) in enumerate(get_hazard_points):
            get_hazard_points[i][0] = return_periods[i]

    # estimate damage
    if len(get_hazard_points) == 0: # no overlay of asset with hazard
        print(0)
        
    else:
        if asset_geom.geom_type == 'LineString':
            overlay_meters = shapely.length(shapely.intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell
            if hazard_intensity[0] != 'Exposure to hazard':
                return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_meters*maxdam_asset) #return asset number, total damage for asset number (damage factor * meters * max. damage)
            elif hazard_intensity[0] == 'Exposure to hazard':
                damage = np.sum(np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * overlay_meters * maxdam_asset)
                return np.vstack([damage, get_hazard_points[:,0]])

        elif asset_geom.geom_type in ['MultiPolygon','Polygon']:
            overlay_m2 = shapely.area(shapely.intersection(get_hazard_points[:,1],asset_geom))
            if '/unit' in unit_maxdam:
                converted_maxdam = maxdam_asset / shapely.area(asset_geom) #convert to maxdam/m2
                if hazard_intensity[0] != 'Exposure to hazard':
                    return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_m2*converted_maxdam)
                elif hazard_intensity[0] == 'Exposure to hazard':
                    damage = np.sum(np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * overlay_m2 * converted_maxdam)
                    return np.vstack([damage, get_hazard_points[:,0]])
            else:
                if hazard_intensity[0] != 'Exposure to hazard':
                    return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_m2*maxdam_asset) #return asset number, total damage for asset number (damage factor * meters * max. damage)
                elif hazard_intensity[0] == 'Exposure to hazard':
                    damage = np.sum(np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * overlay_m2 * maxdam_asset)
                    return np.vstack([damage, get_hazard_points[:,0]])
                    
        elif asset_geom.geom_type == 'Point':
            if hazard_intensity[0] != 'Exposure to hazard':
                return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*maxdam_asset)
            elif hazard_intensity[0] == 'Exposure to hazard':
                damage = np.sum(np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * maxdam_asset)
                return np.vstack([damage, get_hazard_points[:,0]])

def read_vul_maxdam(data_path,hazard_type,infra_type,database_id_curves=False,database_maxdam=False):
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

    database_id_curves=False
    database_maxdam=False

    vul_data = data_path / 'Vulnerability'
    
    # Load assumptions file containing curve - maxdam combinations per infrastructure type
    if hazard_type in ['pluvial','fluvial']: 
        assumptions = pd.read_excel(vul_data / 'S1_Assumptions_Test.xlsx',sheet_name = 'Flooding assumptions',header=[1])
    elif hazard_type == 'windstorm':
        assumptions = pd.read_excel(vul_data / 'S1_Assumptions_Test.xlsx',sheet_name = 'Windstorm assumptions',header=[1])
    elif hazard_type == 'earthquake':
        assumptions = pd.read_excel(vul_data / 'S1_Assumptions_Test.xlsx',sheet_name = 'Earthquake assumptions',header=[1])
    elif hazard_type in ['landslide_eq', 'landslide_rf']:
        assumptions = pd.read_excel(vul_data / 'S1_Assumptions_Test.xlsx',sheet_name = 'Landslide assumptions',header=[1])

    if database_id_curves==False:
        #get assumptions from dictionary
        if hazard_type == 'earthquake':
            assump_curves = ['E7.1', 'E7.6', 'E7.7', 'E7.8', 'E7.9', 'E7.10', 'E7.11', 'E7.12', 'E7.13', 'E7.14' ]
        elif hazard_type in ['landslide_eq', 'landslide_rf']:
            assump_curves = [None]
    else:
        #get assumptions from database
        assumptions['Infrastructure type'] = assumptions['Infrastructure type'].str.lower()
        if "_" in infra_type: infra_type = infra_type.replace('_', ' ')
        assump_infra_type = assumptions[assumptions['Infrastructure type'] == infra_type]
        if assump_infra_type['Vulnerability ID number'].item() == 'No ID number, partial destruction is assumed':
            assump_curves = [None] #code evt uitbreiden, dat het onderscheid maakt tussen infrastructuur types waar wel/geen id nummer voor is gegeven
        else:
            assump_curves = ast.literal_eval(assump_infra_type['Vulnerability ID number'].item())

    if " " in infra_type: infra_type = infra_type.replace(' ', '_')

    # Get curves
    if hazard_type in ['pluvial','fluvial']:  
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_conversions.xlsx',sheet_name = 'F_Vuln_Depth',index_col=[0],header=[0,1,2,3,4])
        infra_curves =  curves[assump_curves]
    elif hazard_type == 'windstorm':
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_conversions.xlsx',sheet_name = 'W_Vuln_V10m',index_col=[0],header=[0,1,2,3,4])
        infra_curves =  curves[assump_curves]
    elif hazard_type == 'earthquake':
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_conversions.xlsx',sheet_name = 'E_Vuln_PGA',index_col=[0],header=[0,1,2,3,4])
        infra_curves =  curves[assump_curves]
    elif hazard_type in ['landslide_eq', 'landslide_rf']:
        if assump_curves == [None]:
            #infra_curves = pd.DataFrame([1], columns=['Complete destruction'])
            #infra_curves.columns=pd.MultiIndex.from_product([['Damage factor'],infra_curves.columns])
            infra_curves = pd.DataFrame([['Exposure to hazard', 0.5]], columns=['Intensity measure', 'Damage factor']).set_index('Intensity measure')
            infra_curves.columns=pd.MultiIndex.from_product([['Partial destruction (0.5)'],infra_curves.columns])
        else:
            curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_conversions.xlsx',sheet_name = 'L_Frag_PGD',index_col=[0],header=[0,1,2,3,4])
            infra_curves =  curves[assump_curves]

    if database_maxdam==False:
        maxdam_dict = {'unclassified':300, 
                        'primary':2000, 
                        'secondary':1300, 
                        'tertiary':700, 
                        'residential':500,
                        'trunk':2000, 
                        'trunk_link':2000, 
                        'motorway':2000, 
                        'motorway_link':2000, 
                        'primary_link':2000, 
                        'secondary_link':1300,
                        'tertiary_link':700,
                        'road':700,
                        'track':300, }
        infra_maxdam =  pd.Series([str(maxdam_dict[infra_type])], index=['default'])
        infra_maxdam.name = 'Amount'   
        infra_units =  pd.Series(['euro/m'], index=['default'])
        infra_units.name = 'unit'
    else:
        # get maxdam from database
        assump_maxdams = ast.literal_eval(assump_infra_type['Maximum damage ID number'].item())
        maxdam = pd.read_excel(vul_data / 'Table_D3_Costs_V1.1.0_converted.xlsx', sheet_name='Cost_Database',index_col=[0])
        infra_costs = maxdam[maxdam.index.isin(assump_maxdams)][['Amount', 'Unit']].dropna(subset=['Amount'])
        infra_maxdam = infra_costs['Amount'][pd.to_numeric(infra_costs['Amount'], errors='coerce').notnull()]
        infra_units = infra_costs['Unit'].filter(items=list(infra_maxdam.index), axis=0)

    return infra_curves,infra_maxdam,infra_units

def matrix_landslide_rf_susc(overlay_rf, get_susc_data, overlay_assets, susc_point):
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

    bool_series = (overlay_assets['hazard_point'] == susc_point[0])

    unique_classes = overlay_rf['cond_classes'].unique()
    if len(unique_classes) == 1:
        rf_class = unique_classes[0]
    else:
        rf_class = max(unique_classes)
    
    if rf_class == '1' and get_susc_data[0] == 1:
        #print('1x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '1' and get_susc_data[0] == 2:
        #print('1x2')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '1' and get_susc_data[0] == 3:
        #print('1x3')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '1' and get_susc_data[0] == 4:
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '1' and get_susc_data[0] == 5:
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '2' and get_susc_data[0] == 1:
        #print('2x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '2' and get_susc_data[0] == 2:
        #print('2x2', susc_point[0], len((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index))
        #print(len((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index))
        overlay_assets.loc[bool_series, 'return_period'] = 100
        overlay_assets.loc[bool_series, 'return_period_trig'] = 5
    elif rf_class == '2' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 50
        overlay_assets.loc[bool_series, 'return_period_trig'] = 5
    elif rf_class == '2' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 33
        overlay_assets.loc[bool_series, 'return_period_trig'] = 5
    elif rf_class == '2' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 20
        overlay_assets.loc[bool_series, 'return_period_trig'] = 5
    elif rf_class == '3' and get_susc_data[0] == 1:
        #print('3x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '3' and get_susc_data[0] == 2:
        overlay_assets.loc[bool_series, 'return_period'] = 50
        overlay_assets.loc[bool_series, 'return_period_trig'] = 25
    elif rf_class == '3' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 33
        overlay_assets.loc[bool_series, 'return_period_trig'] = 25
    elif rf_class == '3' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 20
        overlay_assets.loc[bool_series, 'return_period_trig'] = 25
    elif rf_class == '3' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 10
        overlay_assets.loc[bool_series, 'return_period_trig'] = 25
    elif rf_class == '4' and get_susc_data[0] == 1:
        #print('4x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '4' and get_susc_data[0] == 2:
        overlay_assets.loc[bool_series, 'return_period'] = 33
        overlay_assets.loc[bool_series, 'return_period_trig'] = 200
    elif rf_class == '4' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 20
        overlay_assets.loc[bool_series, 'return_period_trig'] = 200
    elif rf_class == '4' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 10
        overlay_assets.loc[bool_series, 'return_period_trig'] = 200
    elif rf_class == '4' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 7
        overlay_assets.loc[bool_series, 'return_period_trig'] = 200
    elif rf_class == '5' and get_susc_data[0] == 1:
        #print('5x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '5' and get_susc_data[0] == 2:
        overlay_assets.loc[bool_series, 'return_period'] = 20
        overlay_assets.loc[bool_series, 'return_period_trig'] = 1000
    elif rf_class == '5' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 10
        overlay_assets.loc[bool_series, 'return_period_trig'] = 1000
    elif rf_class == '5' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 7
        overlay_assets.loc[bool_series, 'return_period_trig'] = 1000
    elif rf_class == '5' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 5
        overlay_assets.loc[bool_series, 'return_period_trig'] = 1000

    return overlay_assets

def matrix_landslide_eq_susc(overlay_eq, get_susc_data, overlay_assets, susc_point):
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

    bool_series = (overlay_assets['hazard_point'] == susc_point[0])

    unique_classes = overlay_eq['cond_classes'].unique()
    if len(unique_classes) == 1:
        eq_class = unique_classes[0]
    else:
        eq_class = max(unique_classes)
    
    if eq_class == '1' and get_susc_data[0] == 1:
        #print('1x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif eq_class == '1' and get_susc_data[0] == 2:
        #print('1x2')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif eq_class == '1' and get_susc_data[0] == 3:
        #print('1x3')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif eq_class == '1' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 1000
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '1' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 200
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '2' and get_susc_data[0] == 1:
        #print('2x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif eq_class == '2' and get_susc_data[0] == 2:
        #print('2x2', susc_point[0], len((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index))
        #print(len((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index))
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif eq_class == '2' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 1000
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '2' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 200
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '2' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 100
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '3' and get_susc_data[0] == 1:
        #print('3x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif eq_class == '3' and get_susc_data[0] == 2:
        overlay_assets.loc[bool_series, 'return_period'] = 1000
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '3' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 200
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '3' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 100
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '3' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 20
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '4' and get_susc_data[0] == 1:
        #print('4x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif eq_class == '4' and get_susc_data[0] == 2:
        overlay_assets.loc[bool_series, 'return_period'] = 200
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '4' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 100
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '4' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 20
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '4' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 10
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '5' and get_susc_data[0] == 1:
        #print('5x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif eq_class == '5' and get_susc_data[0] == 2:
        overlay_assets.loc[bool_series, 'return_period'] = 100
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '5' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 20
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '5' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 10
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475
    elif eq_class == '5' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 2.5
        overlay_assets.loc[bool_series, 'return_period_trig'] = 475

    return overlay_assets

def matrix_landslide_rf_susc_old(overlay_rf, get_susc_data, overlay_assets, susc_point):
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

    bool_series = (overlay_assets['hazard_point'] == susc_point[0])

    unique_classes = overlay_rf['cond_classes'].unique()
    if len(unique_classes) == 1:
        rf_class = unique_classes[0]
    else:
        rf_class = max(unique_classes)
    
    if rf_class == '1' and get_susc_data[0] == 1:
        #print('1x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '1' and get_susc_data[0] == 2:
        #print('1x2')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '1' and get_susc_data[0] == 3:
        #print('1x3')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '1' and get_susc_data[0] == 4:
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '1' and get_susc_data[0] == 5:
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '2' and get_susc_data[0] == 1:
        #print('2x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '2' and get_susc_data[0] == 2:
        #print('2x2', susc_point[0], len((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index))
        #print(len((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index))
        overlay_assets.loc[bool_series, 'return_period'] = 100
    elif rf_class == '2' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 50
    elif rf_class == '2' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 33
    elif rf_class == '2' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 20
    elif rf_class == '3' and get_susc_data[0] == 1:
        #print('3x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '3' and get_susc_data[0] == 2:
        overlay_assets.loc[bool_series, 'return_period'] = 50
    elif rf_class == '3' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 33
    elif rf_class == '3' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 20
    elif rf_class == '3' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 10
    elif rf_class == '4' and get_susc_data[0] == 1:
        #print('4x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '4' and get_susc_data[0] == 2:
        overlay_assets.loc[bool_series, 'return_period'] = 33
    elif rf_class == '4' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 20
    elif rf_class == '4' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 10
    elif rf_class == '4' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 7
    elif rf_class == '5' and get_susc_data[0] == 1:
        #print('5x1')
        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    elif rf_class == '5' and get_susc_data[0] == 2:
        overlay_assets.loc[bool_series, 'return_period'] = 20
    elif rf_class == '5' and get_susc_data[0] == 3:
        overlay_assets.loc[bool_series, 'return_period'] = 10
    elif rf_class == '5' and get_susc_data[0] == 4:
        overlay_assets.loc[bool_series, 'return_period'] = 7
    elif rf_class == '5' and get_susc_data[0] == 5:
        overlay_assets.loc[bool_series, 'return_period'] = 5

    return overlay_assets

def filter_landslide_rf_rps(trig_rp, overlay_assets_ls_rp):
    """
    Reassign data by setting new landslide return period for certain rainfall-triggering event
    Arguments:
        **:
    Returns:
        **:
    """

    if trig_rp == 5:
        return overlay_assets_ls_rp

    return_period_trig = np.array(overlay_assets_ls_rp['return_period_trig'])
    return_period = np.array(overlay_assets_ls_rp['return_period'])
    
    if trig_rp == 25:
        corresponding_rps = np.where((return_period_trig == 5) & (return_period == 100), 50,
                                     np.where((return_period_trig == 5) & (return_period == 50), 33,
                                              np.where((return_period_trig == 5) & (return_period == 33), 20,
                                                       np.where((return_period_trig == 5) & (return_period == 20), 10, 
                                                                return_period))))

    elif trig_rp == 200:
        corresponding_rps = np.where((return_period_trig == 5) & (return_period == 100), 33,
                                     np.where((return_period_trig == 5) & (return_period == 50), 20,
                                              np.where((return_period_trig == 5) & (return_period == 33), 10,
                                                       np.where((return_period_trig == 5) & (return_period == 20), 7,
                                                                np.where((return_period_trig == 25) & (return_period == 50), 33, 
                                                                         np.where((return_period_trig == 25) & (return_period == 33), 20, 
                                                                                  np.where((return_period_trig == 25) & (return_period == 20), 10, 
                                                                                           np.where((return_period_trig == 25) & (return_period == 10), 7, 
                                                                                                    return_period))))))))

    elif trig_rp == 1000:
        corresponding_rps = np.where((return_period_trig == 5) & (return_period == 100), 20,
                                     np.where((return_period_trig == 5) & (return_period == 50), 10,
                                              np.where((return_period_trig == 5) & (return_period == 33), 7,
                                                       np.where((return_period_trig == 5) & (return_period == 20), 5,
                                                                np.where((return_period_trig == 25) & (return_period == 50), 20, 
                                                                         np.where((return_period_trig == 25) & (return_period == 33), 10, 
                                                                                  np.where((return_period_trig == 25) & (return_period == 20), 7, 
                                                                                           np.where((return_period_trig == 25) & (return_period == 10), 5, 
                                                                                                    np.where((return_period_trig == 200) & (return_period == 10), 20, 
                                                                                                             np.where((return_period_trig == 200) & (return_period == 10), 10, 
                                                                                                                      np.where((return_period_trig == 200) & (return_period == 10), 7, 
                                                                                                                               np.where((return_period_trig == 200) & (return_period == 10), 5, 
                                                                                                    return_period))))))))))))
    overlay_assets_ls_rp.loc[:,'return_period_trig'] = trig_rp # adjust column
    overlay_assets_ls_rp.loc[:,'return_period'] = corresponding_rps

    return overlay_assets_ls_rp

def accumulated_damage_rp_inverse(return_periods_dict_for_asset):
    """
    Adjusts the damage values in the return_periods_dict_for_asset dictionary.
    The damages of all return periods lower than a certain return period are added to the damages of that return period.
    
    Parameters:
    return_periods_dict_for_asset (dict): A dictionary with return periods as keys and damages as values.
    
    Returns:
    dict: Adjusted dictionary with accumulated damages.
    """
    
    # Sort the return periods in descending order
    sorted_return_periods = sorted(return_periods_dict_for_asset.keys(), reverse=True)
    
    # Initialize the cumulative damage
    cumulative_damage = 0
    
    # Iterate over the sorted return periods
    for current_period in sorted_return_periods:
        # Add the current period's damage to the cumulative damage
        cumulative_damage += return_periods_dict_for_asset[current_period]
        
        # Store the adjusted damage in the dictionary
        return_periods_dict_for_asset[current_period] = cumulative_damage
    
    return return_periods_dict_for_asset

def accumulated_damage_rp(return_periods_dict_for_asset):
    """
    Adjusts the damage values in the return_periods_dict_for_asset dictionary.
    The damages of all return periods lower than a certain return period are added to the damages of that return period.
    
    Parameters:
    return_periods_dict_for_asset (dict): A dictionary with return periods as keys and damages as values.
    
    Returns:
    dict: Adjusted dictionary with accumulated damages.
    """
    
    # Sort the return periods
    sorted_return_periods = sorted(return_periods_dict_for_asset.keys())
    
    # Initialize the adjusted dictionary
    adjusted_dict = {}
    
    # Iterate over the sorted return periods
    for i, current_period in enumerate(sorted_return_periods):
        # Initialize the cumulative damage
        cumulative_damage = return_periods_dict_for_asset[current_period]
        
        # Add the damages of all lower return periods
        for lower_period in sorted_return_periods[:i]:
            cumulative_damage += return_periods_dict_for_asset[lower_period]
        
        # Store the adjusted damage in the dictionary
        adjusted_dict[current_period] = cumulative_damage
    
    return adjusted_dict

def overlay_hazard_boundary(df_ds,country_border_geometries):
    """
    Overlay hazard assets on a dataframe of spatial geometries.
    Arguments:
        *df_ds*: GeoDataFrame containing the spatial geometries of the hazard data. 
        *boundary*: GeoDataFrame containing the infrastructure assets.
    Returns:
        *geopandas.GeoSeries*: A GeoSeries containing the spatial geometries of df_ds that intersect with the administrative boundary.
    """
    #overlay 
    hazard_tree = shapely.STRtree(df_ds.geometry.values)
    if (shapely.get_type_id(country_border_geometries.iloc[0]) == 3) | (shapely.get_type_id(country_border_geometries.iloc[0]) == 6): # id types 3 and 6 stand for polygon and multipolygon
        intersect_index = hazard_tree.query(country_border_geometries.geometry,predicate='intersects')
    intersect_index = np.unique(np.concatenate(intersect_index))
    
    return df_ds.iloc[intersect_index].reset_index(drop=True)

def overlay_hazard_boundary_temp(df_ds,country_border_geometries):
    """
    Overlay hazard assets on a dataframe of spatial geometries.
    Arguments:
        *df_ds*: GeoDataFrame containing the spatial geometries of the hazard data. 
        *boundary*: GeoDataFrame containing the infrastructure assets.
    Returns:
        *geopandas.GeoSeries*: A GeoSeries containing the spatial geometries of df_ds that intersect with the administrative boundary.
    """
    #overlay 
    hazard_tree = shapely.STRtree(df_ds.geometry.values)
    #if (shapely.get_type_id(country_border_geometries.iloc[0]) == 3) | (shapely.get_type_id(country_border_geometries.iloc[0]) == 6): # id types 3 and 6 stand for polygon and multipolygon
    intersect_index = hazard_tree.query(country_border_geometries.geometry,predicate='intersects')
    intersect_index = np.unique(np.concatenate(intersect_index))
    
    return df_ds.iloc[intersect_index].reset_index(drop=True)

def read_rainfall_map(rf_data_path,diameter_distance=0.25/2):
     
    # load data from NetCDF file
    with xr.open_dataset(rf_data_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        #ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 10)]
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector

def landslide_damage(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type):
    """
    Calculate and output exposure and damage for landslides.
    Arguments:

    Returns:
        *float*: The calculated damage for the specific asset.
    """  

    trig_rp_lst = sorted(overlay_assets['return_period_trig'].unique()) #get list of unique RPs for landslide trigger
    for trig_rp in trig_rp_lst:
        #overlay_assets_ls_rp = overlay_assets[overlay_assets['return_period_trig'] == trig_rp]
        overlay_assets_ls_rp = overlay_assets[overlay_assets['return_period_trig'] <= trig_rp] #get all locations where certain rp may occur
        overlay_assets_ls_rp = filter_landslide_rf_rps(trig_rp, overlay_assets_ls_rp)
        if hazard_type == 'landslide_eq': collect_asset_damages_per_curve_rp = {key: [] for key in [2.5, 10, 20, 100, 200, 1000]}
        if hazard_type == 'landslide_rf': collect_asset_damages_per_curve_rp = {key: [] for key in sorted(overlay_assets_ls_rp['return_period'].unique())}
        curve_ids_list = [] # for output at asset level            
        for infra_curve in infra_curves:
            # get curves
            curve = infra_curves[infra_curve[0]]
            hazard_intensity = curve.index.values
            fragility_values = (np.nan_to_num(curve.values,nan=(np.nanmax(curve.values)))).flatten()

            for maxdam in maxdams:
                if hazard_type == 'landslide_eq': return_periods_dict_for_infratype = {key: 0 for key in [2.5, 10, 20, 100, 200, 1000]}
                if hazard_type == 'landslide_rf': return_periods_dict_for_infratype = {key: 0 for key in sorted(overlay_assets_ls_rp['return_period'].unique())}
                collect_geom = []
                unit_maxdam = infra_units[maxdams[maxdams == maxdam].index[0]] #get unit maxdam
                
                collect_damage_asset = {}  # for output at asset level
                for asset in tqdm(overlay_assets_ls_rp.groupby('asset'),total=len(overlay_assets_ls_rp.asset.unique())): #group asset items for different hazard points per asset and get total number of unique assets
                    if hazard_type == 'landslide_eq': return_periods_dict_for_asset = {key: 0 for key in [2.5, 10, 20, 100, 200, 1000]}
                    if hazard_type == 'landslide_rf': return_periods_dict_for_asset = {key: 0 for key in sorted(overlay_assets_ls_rp['return_period'].unique())}
                    asset_geom = geom_dict[asset[0]]
                    collect_geom.append(asset_geom.wkt)
                    if np.max(fragility_values) == 0: #if exposure does not lead to damage
                        collect_inb.append(np.empty(0)) #can actually be removed?  
                    else:
                        collect_inb = (get_damage_per_asset_rp(asset,susc_numpified,asset_geom,hazard_intensity,fragility_values,maxdam,unit_maxdam)) #get list of damages for specific asset
                        if not len(collect_inb) == 0:
                            for i in range(len(collect_inb[1])):
                                return_periods_dict_for_infratype[collect_inb[1][i]] += collect_inb[0][i]
                                return_periods_dict_for_asset[collect_inb[1][i]] += collect_inb[0][i] #for output at asset level: get damage per RP asset
                            return_periods_dict_for_asset = accumulated_damage_rp(return_periods_dict_for_asset) #for output at asset leve: get accumulated RP damages for asset
                    for rp in collect_asset_damages_per_curve_rp:
                        asset_damage = pd.Series({asset[0]:return_periods_dict_for_asset[rp]})  # for output at asset level
                        asset_damage.columns = [infra_curve[0]]  # for output at asset level
                        collect_asset_damages_per_curve_rp[rp].append(asset_damage)  # for output at asset level
                curve_ids_list.append(infra_curve[0])  # for output at asset level

                #aggegated output
                return_periods_dict_for_infratype = accumulated_damage_rp(return_periods_dict_for_infratype) #accumulate damages
                for rp in list(return_periods_dict_for_infratype.keys()):
                    collect_output[country_code, rp, trig_rp, sub_system, infra_type, infra_curve[0], ((maxdams[maxdams == maxdam]).index)[0]] = return_periods_dict_for_infratype[rp] #collect output for asset, infra_curve and maxdam combination                              
    
        #asset level output
        for rp in collect_asset_damages_per_curve_rp:
            if len(collect_asset_damages_per_curve_rp[rp]) != 0: 
                asset_damages_per_curve_rp = pd.concat(collect_asset_damages_per_curve_rp[rp], ignore_index=False).to_frame(name='Partial destruction (0.5)')
                #asset_damages_per_curve_rp.columns = curve_ids_list
                damaged_assets = assets_infra_type.merge(asset_damages_per_curve_rp,left_index=True,right_index=True,how='outer')
                damaged_assets = damaged_assets.drop(['buffered'],axis=1)
                damaged_assets.crs = 3857
                damaged_assets = damaged_assets.to_crs(4326)
                damaged_assets.damage = damaged_assets[curve_ids_list].fillna(0)
                damaged_assets['return_period_trig'] = trig_rp
                damaged_assets['return_period_landslide'] = rp
                save_path = pathway_dict['data_path'] / 'damage' / country_code / hazard_type / f'{country_code}_{hazard_type}_ls{rp}_trig{trig_rp}_{sub_system}_{infra_type}.parquet'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                damaged_assets.to_parquet(save_path)


def landslide_damage_and_overlay(overlay_assets,infra_curves,susc_numpified,assets_infra_type, hazard_type):
    """
    Calculate and output exposure and damage for landslides.
    Arguments:

    Returns:
        *float*: The calculated damage for the specific asset.
    """      

    trig_rp_lst = sorted(overlay_assets['return_period_trig'].unique()) #get list of unique RPs for landslide trigger
    for trig_rp in trig_rp_lst:
        #overlay_assets_ls_rp = overlay_assets[overlay_assets['return_period_trig'] == trig_rp]
        overlay_assets_ls_rp = overlay_assets[overlay_assets['return_period_trig'] <= trig_rp] #get all locations where certain rp may occur
        if hazard_type == 'landslide_eq': 
            collect_asset_damages_per_curve_rp = {key: [] for key in [2.5, 10, 20, 100, 200, 1000]}
            collect_asset_exposure_per_curve_rp = {key: [] for key in [2.5, 10, 20, 100, 200, 1000]}
            collect_asset_landslides_per_curve_rp = {key: [] for key in [2.5, 10, 20, 100, 200, 1000]}
        elif hazard_type == 'landslide_rf': 
            overlay_assets_ls_rp = filter_landslide_rf_rps(trig_rp, overlay_assets_ls_rp)
            collect_asset_damages_per_curve_rp = {key: [] for key in sorted(overlay_assets_ls_rp['return_period'].unique())}
            collect_asset_exposure_per_curve_rp = {key: [] for key in sorted(overlay_assets_ls_rp['return_period'].unique())}
            collect_asset_landslides_per_curve_rp = {key: [] for key in sorted(overlay_assets_ls_rp['return_period'].unique())}
        curve_ids_list = [] # for output at asset level            
        for infra_curve in infra_curves:
            # get curves
            curve = infra_curves[infra_curve[0]]
            hazard_intensity = curve.index.values
            fragility_values = (np.nan_to_num(curve.values,nan=(np.nanmax(curve.values)))).flatten()

            for maxdam in maxdams:
                if hazard_type == 'landslide_eq': return_periods_dict_for_infratype = {key: 0 for key in [2.5, 10, 20, 100, 200, 1000]}
                if hazard_type == 'landslide_rf': return_periods_dict_for_infratype = {key: 0 for key in sorted(overlay_assets_ls_rp['return_period'].unique())}
                collect_geom = []
                unit_maxdam = infra_units[maxdams[maxdams == maxdam].index[0]] #get unit maxdam
                
                collect_damage_asset = {}  # for output at asset level
                for asset in tqdm(overlay_assets_ls_rp.groupby('asset'),total=len(overlay_assets_ls_rp.asset.unique())): #group asset items for different hazard points per asset and get total number of unique assets
                    if hazard_type == 'landslide_eq': 
                        return_periods_dict_for_asset = {key: 0 for key in [2.5, 10, 20, 100, 200, 1000]}
                        return_periods_dict_for_asset_exposure = {key: 0 for key in [2.5, 10, 20, 100, 200, 1000]}
                        rp_dict_for_asset_landslide_occur = {key: 0 for key in [2.5, 10, 20, 100, 200, 1000]}
                    elif hazard_type == 'landslide_rf': 
                        return_periods_dict_for_asset = {key: 0 for key in sorted(overlay_assets_ls_rp['return_period'].unique())}
                        return_periods_dict_for_asset_exposure = {key: 0 for key in sorted(overlay_assets_ls_rp['return_period'].unique())}
                        rp_dict_for_asset_landslide_occur = {key: 0 for key in sorted(overlay_assets_ls_rp['return_period'].unique())}
                    asset_geom = geom_dict[asset[0]]
                    collect_geom.append(asset_geom.wkt)
                    if np.max(fragility_values) == 0: #if exposure does not lead to damage
                        collect_inb.append(np.empty(0)) #can actually be removed? 
                        overlay_inb.append(np.empty(0)) #can actually be removed? 
                    else:
                        #collect_inb = (get_damage_per_asset_rp(asset,susc_numpified,asset_geom,hazard_intensity,fragility_values,maxdam,unit_maxdam)) #get list of damages for specific asset
                        collect_inb, overlay_inb = get_damage_and_overlay_per_asset_rp(asset,susc_numpified,asset_geom,hazard_intensity,fragility_values,maxdam,unit_maxdam)
                        if not len(collect_inb) == 0:
                            for i in range(len(collect_inb[1])):
                                return_periods_dict_for_infratype[collect_inb[1][i]] += collect_inb[0][i]
                                return_periods_dict_for_asset[collect_inb[1][i]] += collect_inb[0][i] #for output at asset level: get damage per RP asset
                                return_periods_dict_for_asset_exposure[overlay_inb[1][i]] += overlay_inb[0][i] #for exposure output at asset level: get exposure per RP asset
                                rp_dict_for_asset_landslide_occur[overlay_inb[1][i]] += 1 #for exposure output at asset level: get landslide occurrence per RP asset
                            return_periods_dict_for_asset = accumulated_damage_rp(return_periods_dict_for_asset) #for output at asset level: get accumulated RP damages for asset
                            return_periods_dict_for_asset_exposure = accumulated_damage_rp(return_periods_dict_for_asset_exposure) #for output at asset level: get accumulated RP exposure for asset
                            rp_dict_for_asset_landslide_occur = accumulated_damage_rp(rp_dict_for_asset_landslide_occur) #for output at asset level: get accumulated landslide occurrence for asset
                    for rp in collect_asset_damages_per_curve_rp:
                        asset_damage = pd.Series({asset[0]:return_periods_dict_for_asset[rp]})  # for output at asset level
                        asset_exposure = pd.Series({asset[0]:return_periods_dict_for_asset_exposure[rp]})  # for exposure output at asset level
                        asset_damage.columns = [infra_curve[0]]  # for output at asset level
                        asset_exposure.columns = 'overlay'  # for exposure output at asset level
                        collect_asset_damages_per_curve_rp[rp].append(asset_damage)  # for output at asset level
                        collect_asset_exposure_per_curve_rp[rp].append(asset_exposure)  # for exposure output at asset level
                        asset_landslide_occ = pd.Series({asset[0]:rp_dict_for_asset_landslide_occur[rp]})  # for # of landslides output at asset level
                        asset_landslide_occ.columns = 'number of landslides'  # for # of landslides output at asset level
                        collect_asset_landslides_per_curve_rp[rp].append(asset_landslide_occ)  # for # of landslides output at asset level
                curve_ids_list.append(infra_curve[0])  # for output at asset level

                #aggegated output
                return_periods_dict_for_infratype = accumulated_damage_rp(return_periods_dict_for_infratype) #accumulate damages
                for rp in list(return_periods_dict_for_infratype.keys()):
                    collect_output[country_code, rp, trig_rp, sub_system, infra_type, infra_curve[0], ((maxdams[maxdams == maxdam]).index)[0]] = return_periods_dict_for_infratype[rp] #collect output for asset, infra_curve and maxdam combination                              
    
        #asset level output
        for rp in collect_asset_damages_per_curve_rp:
            if len(collect_asset_damages_per_curve_rp[rp]) != 0: 
                asset_damages_per_curve_rp = pd.concat(collect_asset_damages_per_curve_rp[rp], ignore_index=False).to_frame(name='Partial destruction (0.5)')
                asset_exposure_per_curve_rp = pd.concat(collect_asset_exposure_per_curve_rp[rp], ignore_index=False).to_frame(name='Overlay')
                asset_damages_per_curve_rp = asset_damages_per_curve_rp.merge(asset_exposure_per_curve_rp, left_index=True, right_index=True) #merge exposure with damages dataframe
                asset_landslides_per_curve_rp = pd.concat(collect_asset_landslides_per_curve_rp[rp], ignore_index=False).to_frame(name='number of landslides')
                asset_damages_per_curve_rp = asset_damages_per_curve_rp.merge(asset_landslides_per_curve_rp, left_index=True, right_index=True) #merge landslides with damages dataframe
                #asset_damages_per_curve_rp.columns = curve_ids_list
                damaged_assets = assets_infra_type.merge(asset_damages_per_curve_rp,left_index=True,right_index=True,how='outer')
                damaged_assets['Overlay'] = damaged_assets['Overlay'].fillna(0)
                damaged_assets['number of landslides'] = damaged_assets['number of landslides'].fillna(0)
                damaged_assets = damaged_assets.drop(['buffered'],axis=1)
                damaged_assets.crs = 3857
                damaged_assets = damaged_assets.to_crs(4326)
                damaged_assets[curve_ids_list] = damaged_assets[curve_ids_list].fillna(0)
                damaged_assets['return_period_trig'] = trig_rp
                damaged_assets['return_period_landslide'] = rp
                save_path = pathway_dict['data_path'] / 'damage' / country_code / hazard_type / f'{country_code}_{hazard_type}_ls{rp}_trig{trig_rp}_{sub_system}_{infra_type}.parquet'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                damaged_assets.to_parquet(save_path)

    return collect_output

def get_damage_and_overlay_per_asset_rp(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam_asset,unit_maxdam):
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
    get_hazard_points = get_hazard_points[shapely.intersects(get_hazard_points[:,1],asset_geom)]
    return_periods = asset[1]['return_period'].values
    for i, (point, polygon) in enumerate(get_hazard_points):
        get_hazard_points[i][0] = return_periods[i]
    if type(maxdam_asset) == str: maxdam_asset = float(maxdam_asset)

    # estimate damage
    if len(get_hazard_points) == 0: # no overlay of asset with hazard
        return np.empty(0), np.empty(0)
    else:
        if asset_geom.geom_type == 'LineString':
            overlay_meters = shapely.length(shapely.intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell
            damage = np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * overlay_meters * maxdam_asset
            return np.vstack([damage, get_hazard_points[:,0]]), np.vstack([overlay_meters, get_hazard_points[:,0]])

        elif asset_geom.geom_type in ['MultiPolygon','Polygon']:
            overlay_m2 = shapely.area(shapely.intersection(get_hazard_points[:,1],asset_geom))
            if '/unit' in unit_maxdam:
                converted_maxdam = maxdam_asset / shapely.area(asset_geom) #convert to maxdam/m2
                damage = (np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * overlay_m2 * converted_maxdam)
                return np.vstack([damage, get_hazard_points[:,0]]), np.vstack([overlay_m2, get_hazard_points[:,0]])
            else:
                damage = (np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * overlay_m2 * maxdam_asset)
                return np.vstack([damage, get_hazard_points[:,0]]), np.vstack([overlay_m2, get_hazard_points[:,0]])

def read_liquefaction_map(liquefaction_map_path,bbox,diameter_distance=0.01051720562427702239/2): #0.01083941445811754771/2):
     
    # load data from NetCDF file
    with xr.open_dataset(liquefaction_map_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 1)] #all pga's falling in very low category result in no damages
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values
        
        return ds_vector.reset_index(drop=True)

def overlay_dataframes(df1,df2):
    """
    Overlay a dataframe on another dataframe of spatial geometries.
    Arguments:
        *df1*: GeoDataFrame containing the spatial geometries. 
        *df2*: GeoDataFrame containing the spatial geometries.
    Returns:
        *geopandas.GeoSeries*: A GeoSeries containing the spatial geometries of df1 that intersect with df2.
    """
    
    #overlay 
    hazard_tree = shapely.STRtree(df1.geometry.values)
    intersect_index = hazard_tree.query(df2.geometry.values, predicate='intersects')
    return df1.iloc[intersect_index[1]].reset_index(drop=True)

def eq_liquefaction_matrix(hazard_map,cond_map):
    """
    apply earthquake and liquefaction matrix and drop irrelevant hazard cells
    Arguments:
        *hazard_map*: GeoDataFrame containing earthquake data. 
        *cond_map*: GeoDataFrame containing liquefaction data.
    Returns:
        *geopandas.DataFrame*: A dataframe containing relevant earthquake data.
    """
    
    bins = [0, 0.092, 0.18, 0.34, 0.65, float('inf')]  # Adjust the thresholds as needed
    labels = ['1', '2', '3', '4', '5']
    
    # Create a new column 'classes' based on the thresholds
    hazard_map['classes'] = pd.cut(hazard_map['band_data'], bins=bins, labels=labels, right=False, include_lowest=True)
    
    overlay_hazardpoints = pd.DataFrame(overlay_hazard_assets(cond_map, hazard_map).T, 
                                        columns=['hazard_point', 'cond_point']) #get df with overlays of liquefaction cells with hazards cells
    
    # Convert DataFrame to numpy array
    hazard_numpified = hazard_map.to_numpy()
    cond_numpified = cond_map.to_numpy()
    drop_hazard_points = []

    hazard_classes = hazard_numpified[:, 2]  # get haz class
    cond_classes = cond_numpified[:, 0]      # get con class
    
    # Get the hazard and condition class pairs
    for i, haz_point in tqdm(enumerate(overlay_hazardpoints['hazard_point'].unique())):
        # Get earthquake category for hazard point
        eq_class = hazard_classes[haz_point]
    
        # Get all cond_points associated with this hazard point
        cond_points_for_hazpoint = overlay_hazardpoints[overlay_hazardpoints['hazard_point'] == haz_point]['cond_point']
        
        # Get the lowest cond point value for this hazard point
        cond_point_value = cond_classes[cond_points_for_hazpoint].min()
    
        # Build condition to decide whether to drop the hazard point
        if (
            (eq_class == '1' and cond_point_value in [2, 3, 4, 5]) or
            (eq_class == '2' and cond_point_value in [2, 3, 4]) or
            (eq_class == '3' and cond_point_value in [2, 3]) or
            (eq_class == '4' and cond_point_value == 2)
        ):
            drop_hazard_points.append(haz_point)
    
    return hazard_map.drop(index=drop_hazard_points) 
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

cis_dict = {
    "transportation": {"road": ['unclassified', 'primary', 'secondary', 'tertiary', 'residential', 
                                'trunk', 'trunk_link',  'motorway','motorway_link',  'primary_link','secondary_link', 'tertiary_link','road', 'track' ]
}}

#cis_dict = {
#    "transportation": {"road": ['primary']
#}}

sub_system = 'road'

for ci_system in cis_dict: 
    for sub_system in cis_dict[ci_system]:
        infra_type_lst = cis_dict[ci_system][sub_system]
pathway_dict = create_pathway_dict(data_path, flood_data_path, eq_data_path, landslide_data_path, cyclone_data_path)
country_code= 'GEO' #PNG, TJK, PAK, GEO
hazard_types = ['earthquake']#['fluvial'] #['landslide_rf','landslide_eq'] 


#database_id_curves=True,database_maxdam=False #should be added as parameters for read_vul_maxdam 
eq_data = 'GEM' #GIRI, GEM or GAR data

# get country osm data
data_loc = country_download(country_code)

# get infrastructure data:
print(f'Time to extract OSM data for {sub_system}')
assets = extract_cis(data_loc, sub_system)

# convert assets to epsg3857 (system in meters)
assets = gpd.GeoDataFrame(assets).set_crs(4326).to_crs(3857)

if sub_system == 'road':
    assets = assets.rename(columns={'highway' : 'asset'})

for hazard_type in hazard_types:
    # read hazard data
    hazard_data_path = pathway_dict[hazard_type]
    data_path = pathway_dict['data_path']
    hazard_data_list = read_hazard_data(hazard_data_path,data_path,hazard_type,country_code)
    if hazard_type in ['pluvial','fluvial','windstorm','landslide_eq','landslide_rf']: hazard_data_list = [file for file in hazard_data_list if file.suffix == '.tif'] #put this code in read hazard data
    
    if hazard_type in ['windstorm','earthquake','landslide_eq','landslide_rf']:
        # load country geometry file and create geometry to clip
        ne_countries = gpd.read_file(data_path / "natural_earth" / "ne_10m_admin_0_countries.shp") #https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
        bbox = ne_countries.loc[ne_countries['ISO_A3']==country_code].geometry.envelope.values[0].bounds
        country_border_geometries = ne_countries.loc[ne_countries['ISO_A3']==country_code].geometry
        
    collect_output = {}
    for single_footprint in hazard_data_list: #tqdm(hazard_data_list,total=len(hazard_data_list)):
    
        hazard_name = single_footprint.parts[-1].split('.')[0]
        
        # load hazard map
        if hazard_type in ['pluvial','fluvial']:
            hazard_map = read_flood_map(single_footprint)
        elif hazard_type in ['windstorm']:
             hazard_map = read_windstorm_map(single_footprint,bbox)
        elif hazard_type == 'earthquake': 
             if eq_data == 'GAR': 
                 hazard_map = read_gar_earthquake_map(single_footprint, bbox) #GAR
             elif eq_data == 'GIRI': 
                 hazard_map = read_giri_earthquake_map(single_footprint, bbox) #GIRI
             elif eq_data == 'GEM':
                 hazard_map = read_earthquake_map_csv(single_footprint, bbox) #GEM
             
             liquefaction_map_path = liquefaction_data_path / 'liquefaction_v1_deg.tif'
             cond_map = read_liquefaction_map(liquefaction_map_path, bbox) 
             hazard_map = overlay_dataframes(hazard_map,cond_map) #get hazard polygons that overlay with cond_map
             hazard_map = eq_liquefaction_matrix(hazard_map,cond_map) #apply liquefaction earthquake matrix and drop hazard points that are irrelevant
        elif hazard_type in ['landslide_eq', 'landslide_rf']:
             if hazard_type == 'landslide_eq':
                 if eq_data == 'GAR': 
                     cond_map = read_gar_earthquake_map(single_footprint, bbox) #GAR
                 elif eq_data == 'GIRI': 
                     cond_map = read_giri_earthquake_map(single_footprint, bbox) #GIRI
                 elif eq_data == 'GEM':
                     cond_map = read_earthquake_map_csv(single_footprint, bbox) #GEM
                 
                 # Define the thresholds for the classes
                 bins = [0, 0.05, 0.15, 0.25, 0.35, 0.45, float('inf')]  # Adjust the thresholds as needed
                 labels = ['NaN', '1', '2', '3', '4', '5']
                
                 # Create a new column 'classes' based on the thresholds
                 cond_map['cond_classes'] = pd.cut(cond_map['band_data'], bins=bins, labels=labels, right=False, include_lowest=True)
                 
                 #susc_map = read_susceptibility_map(single_footprint, hazard_type, bbox)
                 susc_map = read_susceptibility_map_cropped((pathway_dict['landslide_eq'] / 'susceptibility_giri' / '{}_EQ_triggered_LS.tif'.format(country_code)))
                 #susc_map = overlay_hazard_boundary(susc_map,country_border_geometries) #overlay with exact administrative border
                 susc_map = overlay_hazard_boundary_temp(susc_map,country_border_geometries) #overlay with exact administrative border
                 susc_map['geometry'] = shapely.buffer(susc_map.geometry, distance=0.0008333333333333522519/2, cap_style='square').values
             elif hazard_type == 'landslide_rf':
                 cond_map = read_rainfall_map(single_footprint)
                 
                 # Define the thresholds for the classes
                 bins = [0, 0.3, 2.0, 3.7, 5.0, float('inf')]  # Adjust the thresholds as needed
                 labels = ['1', '2', '3', '4', '5']
                
                 # Create a new column 'classes' based on the thresholds
                 cond_map['cond_classes'] = pd.cut(cond_map['band_data'], bins=bins, labels=labels, right=False, include_lowest=True)
                 
                 #susc_map = read_susceptibility_map(single_footprint, hazard_type, bbox)
                 susc_map = read_susceptibility_map_cropped((pathway_dict['landslide_rf'] / 'susceptibility_giri' / '{}_RF_triggered_LS_SSP126.tif'.format(country_code)))
                 #susc_map = overlay_hazard_boundary(susc_map,country_border_geometries) #overlay with exact administrative border
                 susc_map = overlay_hazard_boundary_temp(susc_map,country_border_geometries) #overlay with exact administrative border 
                 susc_map['geometry'] = shapely.buffer(susc_map.geometry, distance=0.0008333333333333522519/2, cap_style='square').values
    
        # convert hazard data to epsg 3857
        if hazard_type in ['landslide_eq', 'landslide_rf']:
            cond_map = gpd.GeoDataFrame(cond_map).set_crs(4326).to_crs(3857)
            susc_map = gpd.GeoDataFrame(susc_map).set_crs(4326).to_crs(3857)
        else:
            hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
    
        # Loop through unique infrastructure types within the subsystem
        for infra_type in infra_type_lst: 
            assets_infra_type = assets[assets['asset'] == infra_type].copy().reset_index(drop=True)
        
            # create dicts for quicker lookup
            geom_dict = assets_infra_type['geometry'].to_dict()
            type_dict = assets_infra_type['asset'].to_dict()
    
            ## read vulnerability and maxdam data:
            infra_curves,maxdams,infra_units = read_vul_maxdam(data_path,hazard_type, infra_type)
    
            # start analysis 
            print(f'{country_code} runs for {infra_type} for {hazard_type} using the {hazard_name} map')# for {len(infra_curves.T)*len(maxdams)} combinations')
    
            if hazard_type in ['landslide_eq', 'landslide_rf']:
                if not assets_infra_type.empty:
                    # overlay assets
                    overlay_assets = pd.DataFrame(overlay_hazard_assets(susc_map,buffer_assets(assets_infra_type)).T,columns=['asset','hazard_point'])
                else: 
                    overlay_assets = pd.DataFrame(columns=['asset','hazard_point']) #empty dataframe
                
                # convert dataframe to numpy array
                susc_numpified = susc_map.to_numpy()
    
                #apply hazard x susceptibility matrix
                overlay_assets['return_period'] = pd.Series(dtype='int')
                overlay_assets['return_period_trig'] = pd.Series(dtype='int')
                for susc_point in tqdm(overlay_assets.groupby('hazard_point'),total=len(overlay_assets.hazard_point.unique())):
                    get_susc_data = susc_numpified[susc_point[0]] # get susc classes and coordinates
                    overlay_cond = cond_map[shapely.intersects(cond_map['geometry'],get_susc_data[1])] #overlay earthquake map with single susc geom 
                    #put return period in overlay_assets
                    if not overlay_cond.empty:
                        if hazard_type == 'landslide_eq':
                            overlay_assets = matrix_landslide_eq_susc(overlay_cond, get_susc_data, overlay_assets, susc_point) 
                        elif hazard_type == 'landslide_rf':
                            overlay_assets = matrix_landslide_rf_susc(overlay_cond, get_susc_data, overlay_assets, susc_point)
                    else:
                        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
    
                #run and output damage calculations for landslides
                if not assets_infra_type.empty:
                    if assets_infra_type['geometry'][0].geom_type == 'LineString':
                        collect_output = landslide_damage_and_overlay(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type)
                    else:
                        collect_output = landslide_damage(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type)
            
            elif hazard_type in ['earthquake', 'pluvial', 'fluvial']: #other hazard
                if not assets_infra_type.empty:
                    # overlay assets
                    overlay_assets = pd.DataFrame(overlay_hazard_assets(hazard_map,buffer_assets(assets_infra_type)).T,columns=['asset','hazard_point'])
                else: 
                    overlay_assets = pd.DataFrame(columns=['asset','hazard_point']) #empty dataframe
        
                # convert dataframe to numpy array
                hazard_numpified = hazard_map.to_numpy()
    
                collect_asset_damages_per_curve = [] # for output at asset level
                #collect_asset_exposure_per_curve = [] # for exposure output at asset level
                curve_ids_list = [] # for output at asset level
                for infra_curve in infra_curves:
                    # get curves
                    curve = infra_curves[infra_curve[0]]
                    hazard_intensity = curve.index.values
                    fragility_values = (np.nan_to_num(curve.values,nan=(np.nanmax(curve.values)))).flatten()
    
                    for maxdam in maxdams:
                        collect_inb = []
                        collect_geom = []
                        unit_maxdam = infra_units[maxdams[maxdams == maxdam].index[0]] #get unit maxdam
    
                        collect_damage_asset = {}  # for output at asset level
                        collect_overlay_asset = {}  # for exposure output at asset level
                        for asset in tqdm(overlay_assets.groupby('asset'),total=len(overlay_assets.asset.unique())): #group asset items for different hazard points per asset and get total number of unique assets
                            asset_geom = geom_dict[asset[0]]
                            collect_geom.append(asset_geom.wkt)
                            if np.max(fragility_values) == 0: #if exposure does not lead to damage
                                collect_inb.append(0)  
                            else:
                                #collect_inb.append(get_damage_per_asset_og(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam))
                                collect_inb.append(get_damage_per_asset(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam,unit_maxdam)) #get list of damages for specific asset
                                #collect_damage_asset[asset[0]] = get_damage_per_asset(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam,unit_maxdam) #for output at asset level
                                damage_asset, overlay_asset = get_damage_per_asset_and_overlay(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam,unit_maxdam) #for output at asset level
                                collect_damage_asset[asset[0]] = damage_asset # for output at asset level
                                collect_overlay_asset[asset[0]] = overlay_asset # for exposure output at asset level
                        
                        collect_output[country_code, hazard_name, sub_system, infra_type, infra_curve[0], ((maxdams[maxdams == maxdam]).index)[0]] = np.sum(collect_inb) #, collect_geom # dictionary to store results for various combinations of hazard maps, infrastructure curves, and maximum damage values.
                        asset_damage = pd.Series(collect_damage_asset)  # for output at asset level
                        asset_damage.columns = [infra_curve[0]]  # for output at asset level
                        collect_asset_damages_per_curve.append(asset_damage)  # for output at asset level
                        asset_exposure = pd.Series(collect_overlay_asset)  # for exposure output at asset level
                        asset_exposure.columns = 'overlay'  # for exposure output at asset level
                        #collect_asset_exposure_per_curve.append(asset_exposure)  # for exposure output at asset level
                    curve_ids_list.append(infra_curve[0])  # for output at asset level
    
                if collect_asset_damages_per_curve[0].empty == False: #collect_asset_damages_per_curve.empty == False
                    asset_damages_per_curve = pd.concat(collect_asset_damages_per_curve,axis=1)
                    asset_damages_per_curve.columns = curve_ids_list
                    asset_damages_per_curve = asset_damages_per_curve.merge(asset_exposure.rename('overlay'), left_index=True, right_index=True) #merge exposure with damages dataframe
                    damaged_assets = assets_infra_type.merge(asset_damages_per_curve,left_index=True,right_index=True,how='outer')
                    damaged_assets = damaged_assets.drop(['buffered'],axis=1)
                    damaged_assets.crs = 3857
                    damaged_assets = damaged_assets.to_crs(4326)
                    damaged_assets[curve_ids_list] = damaged_assets[curve_ids_list].fillna(0)
                    save_path = pathway_dict['data_path'] / 'damage' / country_code / hazard_type /f'{country_code}_{hazard_type}_{hazard_name}_{sub_system}_{infra_type}.parquet'
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    damaged_assets.to_parquet(save_path)

            #break #delete after testing, otherwise damage will only be assessed for first hazard map
    
        #create_damage_csv(collect_output, hazard_type, pathway_dict, country_code, sub_system) #with exposure
        if hazard_type in ['landslide_eq', 'landslide_rf']:
            create_damage_csv_without_exposure(collect_output, hazard_type, pathway_dict, country_code, sub_system) 
        else:
            create_damage_csv_without_exposure(collect_output, hazard_type, pathway_dict, country_code, sub_system) #check whether this line should be moved to the left (i.e., Excel overwriting is the case now??)
(save_path.parent).mkdir(parents=True, exist_ok=True)
save_path.parent.mkdir(parents=True, exist_ok=True)
