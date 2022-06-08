"""
  
@Author: Sadhana Nirandjan & Elco Koks  - Institute for Environmental studies, VU University Amsterdam
"""
import numpy as np
import geopandas as gpd
from tqdm import tqdm

#import functions for line_length
from boltons.iterutils import pairwise 
#from geopy.distance import geodesic

#import functions for polygon_area
import pyproj  #and for convert_crs
import shapely.ops as ops
from shapely.geometry.polygon import Polygon
from functools import partial

#import functions for clip_pygeos
import pygeos


def check_dfs_empty(fetched_data_dict):
    """Check whether dataframes saved in dictionary are all empty
    Argumentss:
        *fetched_data_dict*: dictionary with df saved as valyes
    Returns:
        True if all dataframes are empty, or false when at least one dataframe is not empty 
    """    
    fetched_data_empty = []
    for group in fetched_data_dict:
        fetched_data_empty.append(fetched_data_dict[group].empty)
    
    return all(elem == True for elem in fetched_data_empty)

def transform_to_gpd(df1):
    """function to transform pygeos format to geopandas
    Arguments:
        *df1* : dataframe with pygeos coordinates
    Returns:
         df with shapely coordinates
    """
    from shapely.wkb import loads
    temp_df = df1.copy()
    temp_df['geometry'] = temp_df.geometry.apply(lambda x : loads(pygeos.to_wkb(x))) #transform geometry back to shapely geometry
    temp_df = gpd.GeoDataFrame(temp_df, crs="EPSG:4326", geometry='geometry')
    
    return temp_df

def estimate_infra_in_aerodrome(lst_infrastructure_types, df, df_adjust):
    """
    Function to extract airport multipolygons from OpenStreetMap    
    Arguments:
        *osm_path* : file path to the .osm.pbf file of the region 
        for which we want to do the analysis.        
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with all unique airport multipolygons.   
    """
    for infrastructure_type in lst_infrastructure_types:
            df_subset = df[df['asset'] == infrastructure_type].reset_index(drop=True)
            count=0
            if df_subset.empty == False:
                geometry_type = pygeos.geometry.get_type_id(df_subset.iloc[0]["geometry"]) #get geometry id
                #check for each aerodrome which terminals are overlaying with it 
                spat_tree = pygeos.STRtree(df_subset.geometry) # https://pygeos.readthedocs.io/en/latest/strtree.htm
                for aerodrome_row in df_adjust.itertuples():
                    df_overlapping_assets = (df_subset.loc[spat_tree.query(aerodrome_row.aerodrome_geometry,predicate='intersects').tolist()]).sort_index(ascending=True) #get assets that overlaps with polygon
                    #calculate amount of infrastructure associated to infrastructure type that falls within aeredrome
                    df_overlapping_assets.insert(1, "amount", "") #add assettype as column after first column for length calculations
                    if not df_overlapping_assets.empty:
                        count+=1
                        geom_series = list(df_overlapping_assets.geometry)
                        #calculate amount of infrastructure depending on datatype
                        if  geometry_type == 3 or geometry_type == 6: #"polygon" or "multipolygon" in geometry_type
                            df_overlapping_assets["amount"] = polygon_area_pygeos(geom_series) #calculate area per object and put in dataframe
                            df_adjust.loc[aerodrome_row.Index, '{}_m2'.format(infrastructure_type)] = df_overlapping_assets['amount'].sum()
                        elif  geometry_type == 1 or geometry_type == 5: #"linestring" or "multilinestring" in geometry_type:
                            df_overlapping_assets["amount"] = line_length_pygeos(geom_series) #calculate area per object and put in dataframe
                            df_adjust.loc[aerodrome_row.Index, '{}_m'.format(infrastructure_type)] = df_overlapping_assets['amount'].sum()
            else: #specified infrastructure type not used in pbf file
                df_adjust['{}_m'.format(infrastructure_type)] = np.nan #insert nans for whole column


            print('A total of {} aerodromes contain information about the infrastructure type: {}'.format(count, infrastructure_type))
            
    return df_adjust


########################################################################################################################
################          Fast codes using pygeos          #############################################################
########################################################################################################################
    
def clip_pygeos(df1,df2,spat_tree,reset_index=False):
    """fast clipping using pygeos
    Arguments:
        *df1: dataframe with spatial data to be clipped
        *df2: dataframe with spatial data for mask
        *spat_tree: spatial tree with coordinates in pygeos format
        
    Returns:
        dataframe with coordinates in pygeos geometry
    """
    df1 = df1.loc[spat_tree.query(df2.geometry,predicate='intersects').tolist()]
    df1['geometry'] = pygeos.intersection(df1.geometry,df2.geometry)
    df1 = df1.loc[~pygeos.is_empty(df1.geometry)]
    
    if reset_index==True:
        df1.reset_index(drop=True,inplace=True)

    return df1

def clip_pygeos2(gdf1,gdf2,spat_tree,reset_index=False):
    """fast clipping using pygeos, avoiding errors due to self-intersection (while clipping multipolygons)
    Arguments:
        *gdf1: geodataframe with spatial data to be clipped
        *gdf2: geodataframe with spatial data for mask 
        
    Returns:
        dataframe with coordinates in pygeos geometry
    """
    from shapely.wkb import loads
    geom1 = pygeos.from_shapely(gdf1.geometry.buffer(0)) #.buffer avoids self-intersection error
    geom2 = pygeos.from_shapely(gdf2.geometry)
    geom1 = pygeos.intersection(geom1,geom2)
    gdf1['pygeos_geom'] = geom1
    gdf1 = gdf1.loc[~pygeos.is_empty(gdf1.pygeos_geom)]
    gdf1['geometry'] = gdf1.pygeos_geom.apply(lambda x : loads(pygeos.to_wkb(x))) #transform intersecting geometry back to shapely geometry 
    
    if reset_index==True:
        gdf1.reset_index(drop=True,inplace=True)
    
    return gdf1.drop(columns=['pygeos_geom'])

def convert_crs(geom_series):
    """convert crs to other projection to enable spatial calculations with length and areas
    Arguments:
        *geom_series: series with geographic coordinates in Pygeos format
        
    Returns:
        Serie with coordinates in other projection
    """
    # translate geopandas geometries into pygeos geometries
        
    current_crs="epsg:4326"
    #The commented out crs does not work in all cases
    #current_crs = [*network.edges.crs.values()]
    #current_crs = str(current_crs[0])
    lat = pygeos.geometry.get_y(pygeos.centroid(geom_series[0]))
    lon = pygeos.geometry.get_x(pygeos.centroid(geom_series[0]))
    # formula below based on :https://gis.stackexchange.com/a/190209/80697 
    approximate_crs = "epsg:" + str(int(32700-np.round((45+lat)/90,0)*100+np.round((183+lon)/6,0)))
    #from pygeos/issues/95
    geometries = list(geom_series)
    coords = pygeos.get_coordinates(geometries)
    transformer=pyproj.Transformer.from_crs(current_crs, approximate_crs,always_xy=True)
    new_coords = transformer.transform(coords[:, 0], coords[:, 1])
    
    return pygeos.set_coordinates(geometries.copy(), np.array(new_coords).T)

def line_length_pygeos(geom_series):
    """length per asset in meters
    Arguments:
        *geom_series: series with geographic coordinates in Pygeos format
        
    Returns:
        Serie with length in meters
    """
#    return pygeos.length((geom_series))
    return pygeos.length(convert_crs(geom_series))

def polygon_area_pygeos(geom_series):
    """area per asset in m2
    Arguments:
        *geom_series: series with geographic coordinates in Pygeos format
        
    Returns:
        Serie with area in m2
    """  
    return pygeos.area(convert_crs(geom_series))

def count_per_grid_pygeos(infra_dataset, df_store):
    """count of assets per grid
    Arguments:
        *infra_dataset* : a pd with WGS-84 coordinates in Pygeos 
        *df_store* : pd containing WGS-84 (in Pygeos) coordinates per grid on each row
        
    Returns:
        Count per assets per grid in dataframe with the following format: column => {asset}_count and row => the grid
    """
    asset_list = []

    for asset in infra_dataset.asset.unique():
        if not "{}_count".format(asset) in df_store.columns: df_store.insert(0, "{}_count".format(asset), "") #add assettype as column after first column
        asset_list.append(asset)

    spat_tree = pygeos.STRtree(infra_dataset.geometry) # https://pygeos.readthedocs.io/en/latest/strtree.html
    for grid_cell in tqdm(df_store.itertuples(),total=len(df_store)):
        asset_clip = clip_pygeos(infra_dataset,grid_cell,spat_tree) #clip infra data using GeoPandas clip
        count = asset_clip.asset.value_counts() #count number of assets per asset type

        for asset in asset_list:
            if asset in count.index:
                df_store.loc[grid_cell.Index, "{}_count".format(asset)] = count.get(key = asset)
            else:
                df_store.loc[grid_cell.Index, "{}_count".format(asset)] = 0
    
    return df_store

def length_km_per_grid_pygeos(infra_dataset, df_store):
    """Total length in kilometers per assettype per grid (using Pygeos functions to improve speed)
    Arguments:
        *infra_dataset* : a pd with WGS-84 coordinates in Pygeos 
        *df_store* : pd containing WGS-84 (in Pygeos) coordinates per grid on each row
        
    Returns:
        Length in km per assettype per grid in dataframe with the following format: columns => {asset}_km and rows => the gridcell
    """
    asset_list = []

    for asset in infra_dataset.asset.unique():
        if not "{}_count".format(asset) in df_store.columns: df_store.insert(0, "{}_count".format(asset), "") #add assettype as column after first column for count calculations
        if not "{}_km".format(asset) in df_store.columns: df_store.insert(0, "{}_km".format(asset), "") #add assettype as column after first column for length calculations
        asset_list.append(asset)

    spat_tree = pygeos.STRtree(infra_dataset.geometry) # https://pygeos.readthedocs.io/en/latest/strtree.html
    
    for grid_cell in tqdm(df_store.itertuples(),total=len(df_store)):
        asset_clip = clip_pygeos(infra_dataset,grid_cell,spat_tree) #clip infra data using GeoPandas clip
        
        #count per asset type
        count = asset_clip.asset.value_counts() #count number of assets per asset type
        for asset_type in asset_list:
            if asset_type in count.index:
                df_store.loc[grid_cell.Index, "{}_count".format(asset_type)] = count.get(key = asset_type)
            else:
                df_store.loc[grid_cell.Index, "{}_count".format(asset_type)] = 0

        #calculate length for each asset in clipped infrastructure grid
        asset_clip.insert(1, "length_km", "") #add assettype as column after first column for length calculations
        if not asset_clip.empty:
            geom_series = list(asset_clip.geometry)
            asset_clip["length_km"]=line_length_pygeos(geom_series)/1000 #calculate length per object, transform to km and put in dataframe
        
        length_per_type = asset_clip.groupby(['asset'])['length_km'].sum() #get total length per asset_type in grid

        for asset_type in asset_list:
            if asset_type in length_per_type.index:
                df_store.loc[grid_cell.Index, "{}_km".format(asset_type)] = length_per_type.get(key = asset_type)
            else:
                df_store.loc[grid_cell.Index, "{}_km".format(asset_type)] = 0  
                       
    # print(df_store["{}_km".format(asset_type)])
    # df_store["{}_km".format(asset_type)] = df_store["{}_km".format(asset_type)].astype(float)
    # print(df_store["{}_km".format(asset_type)])
    
    return df_store


def area_km2_per_grid_pygeos(infra_dataset, df_store):
    """Total area in km2 per assettype per grid
    Arguments:
        *infra_dataset* : a pd with WGS-84 coordinates in Pygeos 
        *df_store* : pd containing WGS-84 (in Pygeos) coordinates per grid on each row
        
    Returns:
        Area in km2 per assettype per grid in dataframe with the following format: column => {asset}_km2 and row => the gridcell
    """
    asset_list = []

    for asset in infra_dataset.asset.unique():
        if not "{}_count".format(asset) in df_store.columns: df_store.insert(0, "{}_count".format(asset), "") #add assettype as column after first column for count calculations
        if not "{}_km2".format(asset) in df_store.columns: df_store.insert(0, "{}_km2".format(asset), "") #add assettype as column after first column for area calculations
        asset_list.append(asset)

    spat_tree = pygeos.STRtree(infra_dataset.geometry) # https://pygeos.readthedocs.io/en/latest/strtree.html

    for grid_cell in tqdm(df_store.itertuples(),total=len(df_store)):
        asset_clip = clip_pygeos(infra_dataset, grid_cell,spat_tree) #clip infra data using GeoPandas clip

        #count per asset type
        count = asset_clip.asset.value_counts() #count number of assets per asset type
        for asset_type in asset_list:
            if asset_type in count.index:
                df_store.loc[grid_cell.Index, "{}_count".format(asset_type)] = count.get(key = asset_type)
            else:
                df_store.loc[grid_cell.Index, "{}_count".format(asset_type)] = 0

        #calculate area for each asset in clipped infrastructure grid
        asset_clip.insert(1, "area_km2", "") #add assettype as column after first column for length calculations
        if not asset_clip.empty:
            geom_series = list(asset_clip.geometry)
            asset_clip["area_km2"] = polygon_area_pygeos(geom_series)/1000000 #calculate area per object and put in dataframe

        area_per_type = asset_clip.groupby(['asset'])['area_km2'].sum() #get total length per asset_type in grid
        for asset_type in asset_list:
            if asset_type in area_per_type.index:
                df_store.loc[grid_cell.Index, "{}_km2".format(asset_type)] = area_per_type.get(key = asset_type)
            else:
                df_store.loc[grid_cell.Index, "{}_km2".format(asset_type)] = 0        
        
    return df_store

def get_all_values(nested_dictionary):
    """Get all values in a nested dictionary
    Arguments:
        *nested_dictionary*: dictionary containing keys and values
    """    
    for key, value in nested_dictionary.items():
        if type(value) is dict:
            get_all_values(value)
        else:
            print("{:<28}: {:>30}".format(key, value))

def check_dfs_empty(fetched_data_dict):
    """Check whether dataframes saved in dictionary are all empty
    Argumentss:
        *fetched_data_dict*: dictionary with df saved as valyes
    Returns:
        True if all dataframes are empty, or false when at least one dataframe is not empty 
    """    
    fetched_data_empty = []
    for group in fetched_data_dict:
        fetched_data_empty.append(fetched_data_dict[group].empty)
    
    return all(elem == True for elem in fetched_data_empty)
            
