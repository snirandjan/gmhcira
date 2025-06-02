"""
Code to extract and organize OSM data for multiple infrastructure types for any country in the world
  
@Author: Sadhana Nirandjan & Elco Koks  - Institute for Environmental studies, VU University Amsterdam
"""

import logging
import warnings
from pathlib import Path
import urllib.request
from urllib.parse import urljoin
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from osgeo import gdal, ogr
from shapely.geometry import MultiPolygon
from shapely.ops import transform
from shapely.validation import make_valid
from tqdm import tqdm

import damage_functions
import osm_flex.download as dl
from osm_flex.config import DICT_CIS_OSM, DICT_GEOFABRIK, OSM_CONFIG_FILE, OSM_DATA_DIR, OSM_DIR
from osm_flex.simplify import remove_contained_points, remove_contained_polys

#########################################################################################################################
################          define paths and variables        #############################################################
#########################################################################################################################
# Logging setup
LOGGER = logging.getLogger(__name__)
gdal.SetConfigOption("OSM_CONFIG_FILE", str(OSM_CONFIG_FILE))


base_path = Path('/scistor/ivm/') 
OSM_DIR = base_path / 'data_catalogue' / 'open_street_map' / 'osm' 
OSM_DATA_DIR = OSM_DIR / "osm_bpf"
#OSMCONVERT_PATH = OSM_DIR / 'osmconvert'
#POLY_DIR = OSM_DIR.joinpath("poly")
#EXTRACT_DIR = OSM_DIR.joinpath("extracts")

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

#########################################################################################################################
################          functions        #############################################################
#########################################################################################################################

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

def _download_file(download_url: str, filepath: Path, overwrite: bool = True):
    """Download a file located at an URL to a local file path

    Parameters
    ----------
    download_url : str
        URL of the file to download
    filepath : str or Path
        Local file path to store the file
    overwrite : bool, optional
        Overwrite existing files. If ``False``, the download will be skipped for
        existing files. Defaults to ``True``.
    """
    if not Path(filepath).is_file() or overwrite:
        LOGGER.info(f"Download file: {filepath}")
        urllib.request.urlretrieve(download_url, filepath)
    else:
        LOGGER.info(f"Skip existing file: {filepath}")

def get_country_geofabrik(iso3, file_format='pbf', save_path=OSM_DATA_DIR,
                          overwrite=False):
    """
    Download country files with all OSM map info from the provider
    Geofabrik.de.

    Parameters
    ----------
    iso3 : str
        ISO3 code of country to download
        Exceptions: Russia is divided into European and Asian part
        ('RUS-E', 'RUS-A'), Canary Islands are 'IC'.
    file_format : str
        Format in which file should be downloaded; options are
        ESRI Shapefiles (shp), which can easily be loaded into gdfs,
        or osm-Protocolbuffer Binary Format (pbf), which is smaller in
        size, but has a more complicated query syntax to load (functions
        are provided in the OSMFileQuery class).
    save_path : str or pathlib.Path
        Folder in which to save the file

    Returns
    -------
    filepath : Path
        The path to the downloaded file (``save_path`` + the Geofabrik filename)

    See also
    --------
    DICT_GEOFABRIK for exceptions / special regions.
    """

    download_url = dl._create_gf_download_url(iso3, file_format)
    filepath = Path(save_path, Path(download_url).name)
    _download_file(download_url, filepath, overwrite)

    return filepath

def country_download(iso3):
    """
    Download OpenStreetMap data for a specific country.
    Arguments:
        *iso3* (str): ISO 3166-1 alpha-3 country code.
    Returns:
        *Path*: The file path of the downloaded OpenStreetMap data file.
    """
    
    get_country_geofabrik(iso3) # Use the download library to get the geofabrik data for the specified country
    data_loc = OSM_DATA_DIR.joinpath(f'{DICT_GEOFABRIK[iso3][1]}-latest.osm.pbf') # Specify the location of the OpenStreetMap (OSM) data file
    return data_loc

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

def geometry_collection_to_polygon(geometry):
    """
    Extracts polygonal parts from a GeometryCollection.
    
    Returns a Polygon, MultiPolygon, or original geometry.
    """
    if geometry.geom_type == 'GeometryCollection':
        # Extract polygons and multipolygons from the collection
        polygons = [geom for geom in geometry.geoms if geom.geom_type in ['Polygon', 'MultiPolygon']]
        
        if not polygons:
            # No polygonal geometries found, return None or handle as appropriate
            return None

        # Combine polygons if multiple are found
        return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]

    return geometry  # Return the original geometry if it's not a GeometryCollection

def check_and_convert_geometry_collection(gdf):
    """
    Check if GeometryCollections exist in the GeoDataFrame and convert them to Polygons if possible.
    
    Args:
        gdf (geopandas.GeoDataFrame): The input GeoDataFrame.

    Returns:
        geopandas.GeoDataFrame: Updated GeoDataFrame with GeometryCollections converted.
    """
    # Check for GeometryCollections
    has_geometry_collections = gdf['geometry'].apply(lambda geom: geom.geom_type == 'GeometryCollection').any()

    if has_geometry_collections:
        print("GeometryCollection detected. Converting...")
        # Apply conversion function to GeometryCollections
        gdf['geometry'] = gdf['geometry'].apply(geometry_collection_to_polygon)
    else:
        print("No GeometryCollections found.")

    return gdf

def make_geometry_valid(geometry):
    """
    Ensures polygon geometries are valid using buffering or repair.
    
    Args:
        geometry (shapely geometry): Input geometry.

    Returns:
        shapely geometry: Valid geometry.
    """
    if geometry.geom_type in ['Polygon', 'MultiPolygon']:
        return geometry.buffer(0) if geometry.is_valid else make_valid(geometry)
    return geometry  # Return geometry unchanged if not polygonal

def remove_contained_assets_and_convert(assets):
    """
    Process the geometry of assets, removing contained points and polygons, 
    and converting points to polygons with appropriate buffering per infrastructure type.
    
    Args:
        assets (geopandas.GeoDataFrame): Input GeoDataFrame containing asset geometries.
    
    Returns:
        geopandas.GeoDataFrame: Processed GeoDataFrame with updated asset geometries.
    """
    warnings.filterwarnings("ignore", message=".*Results from 'area' are likely incorrect.*")
    warnings.filterwarnings("ignore", message=".*Results from 'buffer' are likely incorrect.*")
    
    #assets['geometry'] = assets['geometry'].apply(lambda geom: geom.buffer(0) if geom.is_valid else make_valid(geom)) # Ensure all geometries are valid
    assets['geometry'] = assets['geometry'].apply(make_geometry_valid)
    assets =  remove_contained_polys(remove_contained_points(assets)) #remove points and polygons within a (larger) polygon
    
    if assets.empty:
        return assets  # Early return if no assets remain

    # Buffer distances by infrastructure type
    buffer_distances = {
        'plant': 0.0007016138507930242,
        'substation': 0.0003004193298545618,  
        'reservoir_covered': 6.039122452778908e-05,
        'water_treatment_plant': 3.44507256807309e-05,        
        'wastewater_plant': 0.000574197765365838,
        'waste_transfer_station': 0.000574197765365838,
        'clinic':8.258764284137207e-05,
        'doctors':7.075323314034266e-05,
        'hospital':0.0005037184059573436,
        'dentist':7.439664810916154e-05,
        'pharmacy':6.060785839933688e-05,
        'physiotherapist':5.5876605122779804e-05,
        'alternative':5.5876605122779804e-05,
        'laboratory':5.5876605122779804e-05,
        'optometrist':5.5876605122779804e-05,
        'rehabilitation':5.5876605122779804e-05,
        'blood_donation':5.5876605122779804e-05,
        'birthing_center':5.5876605122779804e-05,
        'college':0.0010528286903641435,
        'kindergarten':0.00012760493965901262,
        'library':0.00010004286581216119,
        'school':0.0001492946457508577,
        'university':8.958092361101363e-05,
    }
    
    # Process each infrastructure type group separately
    updated_assets_list = []
    
    for infra_type, group in assets.groupby('asset'):
        if group.loc[group.geom_type.isin(['Polygon', 'MultiPolygon'])].empty:
            # Use predefined buffer if available, else set a default buffer
            buffer_distance = buffer_distances.get(infra_type, 5.5876605122779804e-05)  # Default buffer for unknown types
        else:
            # Calculate buffer from existing polygons if present
            buffer_distance = np.sqrt(group.loc[group.geom_type.isin(['Polygon', 'MultiPolygon'])].area.median()) / 2
        
        # Buffer points to polygons
        group.loc[group.geom_type == 'Point', 'geometry'] = group.loc[group.geom_type == 'Point'].buffer(
            distance=buffer_distance, cap_style='square'
        )
        
        updated_assets_list.append(group)
    
    # Concatenate all processed groups
    updated_assets = pd.concat(updated_assets_list, ignore_index=True)
    
    return updated_assets

def create_point_from_polygon(gdf):
    """
    Transforms polygons into points
    Arguments:
        gdf: A geodataframe containing a column geometry
    Returns:
    - geopandas.GeoDataFrame: The updated GeoDataFrame without polygons but with only point geometries
    """
    warnings.filterwarnings("ignore", message=".*Results from 'centroid' are likely incorrect.*")

    gdf['geometry'] = gdf['geometry'].apply(make_geometry_valid)
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: MultiPolygon([geom]) if geom.geom_type == 'Polygon' else geom) #convert to multipolygons in case polygons are in the df
    #gdf.loc[gdf.geom_type == 'MultiPolygon','geometry'] = gdf.loc[assets.geom_type == 'MultiPolygon'].centroid #convert polygon to point
    gdf.loc[gdf.geom_type == 'MultiPolygon','geometry'] = gdf.loc[gdf.geom_type == 'MultiPolygon'].centroid #convert polygon to point
    return gdf
    
def process_selected_assets(gdf, polygon_types, point_types, line_types=None):
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
    if polygon_types == ['plant', 'substation']:
        polygon_gdf = remove_contained_assets_and_convert(filtered_assets).drop(columns=['asset_temp'])
    else:
        polygon_gdf = (filtered_assets.groupby('asset_temp').apply(remove_contained_assets_and_convert, include_groups=False)).reset_index(drop=True)

    # For assets that we need as (multi-)points: group by asset type and apply the processing function
    filtered_assets = gdf[gdf['asset'].isin(point_types)] # Filter only selected asset types
    #point_gdf = (filtered_assets.groupby('asset').apply(create_point_from_polygon)).reset_index(drop=True)
    point_gdf = (filtered_assets.groupby('asset_temp').apply(lambda group: create_point_from_polygon(remove_polygons_with_contained_points(group)), include_groups=False)).reset_index(drop=True)

    # Concatenate the two dataframes along rows
    if line_types == None:
        merged_gdf = pd.concat([polygon_gdf, point_gdf], ignore_index=True)
    else:
        line_gdf = gdf[gdf['asset'].isin(line_types)].drop(columns=['asset_temp'])
        merged_gdf = pd.concat([polygon_gdf, point_gdf, line_gdf], ignore_index=True)
        
    return merged_gdf

def filter_air(assets):
    """
    Filters the assets DataFrame to retain:
    - Runways as LineString or MultiLineString geometries
    - Terminals and airports as Polygon or MultiPolygon geometries
    
    Args:
        assets (gpd.GeoDataFrame): GeoDataFrame containing asset geometries and types.
    
    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame with relevant assets.
    """
    assets['geometry'] = assets['geometry'].apply(make_geometry_valid)
    
    # Define asset type and geometry mappings
    runway_types = ['LineString', 'MultiLineString']
    airport_types = ['Polygon', 'MultiPolygon']
    
    # Create filters for each condition
    runway_filter = (
        (assets['asset'] == 'runway') & 
        (assets['geometry'].geom_type.isin(runway_types))
    )
    
    terminal_airport_filter = (
        (assets['asset'].isin(['terminal', 'airport'])) & 
        (assets['geometry'].geom_type.isin(airport_types))
    )
    
    # Apply filters and return the filtered DataFrame
    filtered_assets = assets[runway_filter | terminal_airport_filter]
    
    return filtered_assets

def filter_water_treatment_plants(df, asset_type='water_treatment_plant', keywords=None):
    """
    Filters out specified asset types containing unwanted keywords in the name column.
    
    Args:
        df (pd.DataFrame): DataFrame containing asset information.
        asset_type (str): The asset type to filter, default is 'water_treatment_plant'.
        keywords (list): List of keywords to check in the name column.
    
    Returns:
        pd.DataFrame: Filtered DataFrame without unwanted asset entries.
    """

    if keywords is None:
        keywords = ['hydrant', 'pump', 'waterwheel']
    
    # Create a mask to identify rows to exclude
    mask = (
        (df['asset'] == asset_type) &
        df['name'].str.contains('|'.join(keywords), case=False, na=False)
    )
    
    # Return filtered DataFrame
    return df[~mask]

def osm_extraction(country_code, sub_system, country_border_geometries, asset_loc):
    """
    Extracts, reclassifies, filters, and processes OpenStreetMap (OSM) infrastructure data 
    for a given country and infrastructure subsystem, then saves the output as a GeoDataFrame.

    This function performs multiple data handling steps:
    - Reads raw OSM data for a specific country.
    - Extracts features relevant to the given subsystem (e.g., roads, power lines, healthcare).
    - Reclassifies raw OSM tags to standardized asset types.
    - Cleans and filters geometries (removes irrelevant or malformed ones).
    - Clips assets to the country's boundary.
    - Ensures each asset has a unique `osm_id` and saves the result as a Parquet file.

    Args:
        country_code (str): ISO3 country code (e.g., 'KEN', 'IDN') to identify the country for which assets are extracted.
        sub_system (str): The name of the infrastructure subsystem to extract (e.g., 'power', 'road', 'healthcare').
        country_border_geometries (GeoDataFrame): A GeoDataFrame containing the boundary geometry of the country, 
            used for clipping extracted features.
        asset_loc (Path or str): File path where the processed assets will be saved in Parquet format.

    Returns:
        geopandas.GeoDataFrame: A cleaned, clipped, and reclassified GeoDataFrame of infrastructure assets 
        for the given subsystem and country.

    Notes:
        - Relies on a number of helper functions including `extract_cis`, `filter_dataframe`, 
          `process_selected_assets`, `clip_shapely`, `delete_linestring_data`, etc.
        - Applies custom classification logic per subsystem (e.g., mapping OSM `power=line` to `transmission_line`).
        - Handles geometry conversions (point, polygon, line) to ensure consistent structure.
        - Auto-generates unique `osm_id` values where missing and resolves duplicates.
        - Output is saved using `.to_parquet()` at the path specified by `asset_loc`.
        - Prints progress and warning messages for user feedback and diagnostics.

    Raises:
        KeyError: If expected OSM keys for reclassification are missing from the input data.
        ValueError: If input geometries are invalid or inconsistent.
    """
    #data_loc = country_download(country_code)
    data_loc = OSM_DIR.parent / 'country_osm' / f'{country_code}.osm.pbf' 

    # get infrastructure data
    print(f'Time to extract {country_code} OSM data for {sub_system}')
    assets = extract_cis(data_loc, sub_system)

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
        line_types = ['transmission_line', 'distribution_line', 'cable']
        polygon_types = ['plant', 'substation']
        point_types = ['power_tower', 'power_pole']
        assets = process_selected_assets(assets, polygon_types, point_types, line_types)
        assets = check_and_convert_geometry_collection(assets)

    elif sub_system == 'road':
        assets = assets.rename(columns={'highway' : 'asset'})
        assets['asset'] = assets['asset'].str.lower() 

        #reclassify assets 
        mapping_dict = {
            "motorway" : "motorway", 
            "motorway_link" : "motorway", 
            "trunk" : "trunk",
            "trunk_link" : "trunk",
            "primary" : "primary", 
            "primary_link" : "primary", 
            "secondary" : "secondary", 
            "secondary_link" : "secondary", 
            "tertiary" : "tertiary", 
            "tertiary_link" : "tertiary", 
            "residential" : "residential",           
            "road" : "road", 
            "unclassified" : "track",
            "track" : "track",
        }
        assets['asset'] = assets.asset.apply(lambda x : mapping_dict[x])  #reclassification

    elif sub_system == 'rail':
        assets = assets.rename(columns={'railway' : 'asset'})
        assets['asset'] = assets['asset'].str.lower() 
        
        #reclassify assets 
        mapping_dict = {
            "rail" : "railway", 
            "narrow_gauge" : "railway"}

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
        assets = filter_air(assets)
        assets = check_and_convert_geometry_collection(assets)
    
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
        assets = filter_water_treatment_plants(assets)
        
        # process geometries according to infra type
        polygon_types = ['reservoir_covered', 'water_treatment_plant']
        point_types = ['water_tower', 'water_well', 'storage_tank']
        assets = process_selected_assets(assets, polygon_types, point_types)
        assets = check_and_convert_geometry_collection(assets)

    elif sub_system == 'waste_solid':
        assets = assets.rename(columns={'amenity' : 'asset'})
        assets = remove_contained_assets_and_convert(assets)
        assets = check_and_convert_geometry_collection(assets)

    elif sub_system == 'waste_water':
        assets = assets.rename(columns={'man_made' : 'asset'})
        mapping_dict = {
            "wastewater_plant" : "wastewater_treatment_plant", 
        }
        assets['asset'] = assets.asset.apply(lambda x : mapping_dict[x])  #reclassification
        assets = remove_contained_assets_and_convert(assets)
        assets = check_and_convert_geometry_collection(assets)

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

        # process geometries according to infra type
        polygon_types = []
        point_types = ['communication_tower', 'mast']
        assets = process_selected_assets(assets, polygon_types, point_types) #remove duplicates and transform polygons into points per asset type
        #assets = create_point_from_polygon(remove_polygons_with_contained_points(assets)) #remove duplicates and transform polygons into points

    elif sub_system == 'healthcare':
        column_names_lst = ['amenity' , 'building', 'healthcare']
        assets = filter_dataframe(assets, column_names_lst)
        list_of_assets_to_keep = ["clinic", "doctors", "hospital", "dentist", "pharmacy", 
                        "physiotherapist", "alternative", "laboratory", "optometrist", "rehabilitation", 
                        "blood_donation", "birthing_center"]
        assets = assets.loc[assets.asset.isin(list_of_assets_to_keep)].reset_index(drop=True)
        assets = check_and_convert_geometry_collection(assets)
    
    elif sub_system == 'education':
        column_names_lst = ['amenity' , 'building']
        assets = filter_dataframe(assets, column_names_lst)
        list_of_assets_to_keep =["college", "kindergarten", "library", "school", "university"]
        assets = assets.loc[assets.asset.isin(list_of_assets_to_keep)].reset_index(drop=True)
        assets = check_and_convert_geometry_collection(assets)

    assets = gpd.GeoDataFrame(assets, geometry='geometry')
    assets = assets.set_crs(4326, inplace=True)

    # clip assets
    if not country_border_geometries.empty:  
        if country_code not in ['...missing iso3']: 
            print(f'Time to clip {sub_system} extraction for {country_code}')    
            assets = damage_functions.clip_shapely(assets, country_border_geometries) # Clip using the rewritten function
        else: #only clean up dataframe, but don't perform clip
            print(f'Time to clean-up {sub_system} extraction for {country_code}') 
            assets = damage_functions.overlay_shapely(assets, country_border_geometries)
    else:
        print(
            f"ISO_3digit code not specified in file containing shapefiles of country boundaries. "
            f"Floating data will not be removed for area '{country_code}'.")
        
    # check for double or missing osm_ids 
    if assets['osm_id'].isna().any() == True:
        print(f'NOTIFICATION: there are assets without an osm_id in this dataframe {country_code}, {sub_system}!')
        subset = assets[assets['osm_id'].isna()]
        unique_ids = [str(i) + '_x' for i in range(1, len(subset) + 1)] # Assign a unique osm_id to these rows
        assets.loc[subset.index, 'osm_id'] = unique_ids # Update the original DataFrame with the new osm_id values
        print('osm_ids has been created for assets missing an osm_id')
    if assets['osm_id'].is_unique == False:
        print(f'NOTIFICATION: there are assets with a duplicate osm_id in this dataframe {country_code}, {sub_system}!')
        subset = assets[assets['osm_id'].duplicated(keep=False)]
        unique_suffixes = [f"_{chr(ord('a') + i)}" for i in subset.groupby('osm_id').cumcount()]
        assets.loc[subset.index, 'osm_id'] = assets.loc[subset.index, 'osm_id'].astype(str) + pd.Series(unique_suffixes, index=subset.index)

    # Export the GeoDataFrame to a shapefile
    (asset_loc.parent).mkdir(parents=True, exist_ok=True)
    assets.to_parquet(asset_loc)

    return assets

def asset_extraction(country_code, pathway_dict, cis_dict, overwrite):
    """
    Extracts OpenStreetMap (OSM) infrastructure asset data for a given country and 
    saves it to the specified output path if it doesn't already exist or if overwriting is enabled.

    This function loops through the specified infrastructure systems and subsystems,
    determines the output path for each, and performs spatial extraction of relevant OSM data 
    within the country's administrative boundary.

    Args:
        country_code (str): ISO3 country code (e.g., 'ETH', 'IDN').
        pathway_dict (dict): Dictionary of required paths for input and output data, 
            including 'data_path' and 'output_path'.
        cis_dict (dict): Dictionary defining the critical infrastructure systems and their 
            associated subsystems (e.g., energy → power → transmission_line).
        overwrite (bool): If True, existing extracted data will be overwritten; 
            otherwise, only missing files are generated.

    Returns:
        None

    Notes:
        - Uses a simplified GADM admin0 boundary to clip to filter infrastructure data.
        - The actual OSM extraction is delegated to the `osm_extraction()` function.
        - A warning is printed if the country code is invalid (i.e., equals '-99').
    """
    if country_code == '-99':
        print('Please check country or file, ISO3 equals -99')
    else:
        # load country geometry file
        data_path = pathway_dict['data_path']
        gadm_countries = gpd.read_parquet(data_path / "gadm" / "gadm_410_simplified_admin0_income") 
        #ne_countries = gpd.read_file(data_path / "natural_earth" / "ne_10m_admin_0_countries.shp") #https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/        
        country_border_geometries = gadm_countries.loc[gadm_countries['GID_0']==country_code].geometry

        for ci_system in cis_dict: 
            for sub_system in cis_dict[ci_system]:
                asset_loc = pathway_dict['output_path'] / 'extracts' / country_code / sub_system
                if not asset_loc.exists() or overwrite == True:
                    ## extract and output osm data
                    osm_extraction(country_code, sub_system, country_border_geometries, asset_loc)
                    

