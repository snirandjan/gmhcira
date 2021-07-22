import os,sys
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import pygeos
from tqdm import tqdm
import dask.dataframe as dd
from dask.multiprocessing import get
from dask.diagnostics import ProgressBar
import pyproj

def get_fragility(v,vthreshold,vhalf):
    """[summary]

    Args:
        v ([type]): [description]
        vthreshold ([type]): [description]
        vhalf ([type]): [description]

    Returns:
        [type]: [description]
    """    
    vn = max(v-vthreshold,0)/(vhalf-vthreshold)
    return np.power(vn,3)/(1+np.power(vn,3))

def get_geoms(row):
    """[summary]

    Args:
        row ([type]): [description]

    Returns:
        [type]: [description]
    """
    return pygeos.points(row.x,row.y)

def reproject(df_ds,current_crs="epsg:3857",approximate_crs = "epsg:4326"):
    """[summary]

    Args:
        df_ds ([type]): [description]
        current_crs (str, optional): [description]. Defaults to "epsg:3857".
        approximate_crs (str, optional): [description]. Defaults to "epsg:4326".

    Returns:
        [type]: [description]
    """    

    geometries = df_ds['geometry']
    coords = pygeos.get_coordinates(geometries)
    transformer=pyproj.Transformer.from_crs(current_crs, approximate_crs,always_xy=True)
    new_coords = transformer.transform(coords[:, 0], coords[:, 1])
    
    return pygeos.set_coordinates(geometries.copy(), np.array(new_coords).T) 


def read_flood_as_dataframe(hazard_file):
    """[summary]

    Args:
        hazard_file ([type]): [description]

    Returns:
        [type]: [description]
    """    
    with xr.open_rasterio(hazard_file) as ds:
        df_ds = ds.to_dataframe(name='Xaver').reset_index()
        df_ds = df_ds.loc[(df_ds.Xaver != -9999) & (df_ds.Xaver != 0)]
        df_ds.reset_index(inplace=True,drop=True)

        ddata = dd.from_pandas(df_ds, npartitions=32)

        with ProgressBar():
            df_ds['geometry'] = ddata.map_partitions(lambda df_ds: df_ds.apply((lambda row: get_geoms(row)),axis=1),meta=('object')).compute(scheduler='processes') 

        #df_ds['geometry'] = reproject(df_ds,current_crs="epsg:3857",approximate_crs = "epsg:4326")
        
    return df_ds

def rough_flood_extent(df_ds):
    """[summary]

    Args:
        df_ds ([type]): [description]

    Returns:
        [type]: [description]
    """
    return pygeos.convex_hull(pygeos.multipoints(df_ds['geometry'].values))


def read_line_assets(osm_data_path,asset_type='railways',country='DEU'):
    """[summary]

    Args:
        asset_type (str, optional): [description]. Defaults to 'railways'.
        country (str, optional): [description]. Defaults to 'DEU'.

    Returns:
        [type]: [description]
    """    
    # load data
    assets = pd.read_feather(os.path.join(osm_data_path,'{}_{}.feather'.format(country,asset_type)))
    assets.geometry = assets.geometry.progress_apply(pygeos.from_wkb)
    
    # reproject
    assets['geometry'] = reproject(assets,current_crs="epsg:4326",approximate_crs ="epsg:3857")
    
    # make STRtree
    tree = pygeos.STRtree(assets.geometry.values)
    
    return assets,tree

def read_point_assets(osm_data_path,asset_type='telecom',country='DEU'):
    """[summary]

    Args:
        asset_type (str, optional): [description]. Defaults to 'telecom'.
        country (str, optional): [description]. Defaults to 'DEU'.

    Returns:
        [type]: [description]
    """    
    # load data
    assets = pd.read_feather(os.path.join(osm_data_path,'{}_{}.feather'.format(country,asset_type)))
    assets.geometry = assets.geometry.progress_apply(pygeos.from_wkb)
    
    # reproject
    assets['geometry'] = reproject(assets,current_crs="epsg:4326",approximate_crs ="epsg:3857")
    
    # make STRtree
    tree = pygeos.STRtree(assets.geometry.values)
    
    return assets,tree

def read_polygon_assets(osm_data_path,asset_type='airports',country='DEU'):
    """[summary]

    Args:
        asset_type (str, optional): [description]. Defaults to 'airports'.
        country (str, optional): [description]. Defaults to 'DEU'.

    Returns:
        [type]: [description]
    """    
    # load data
    assets = pd.read_feather(os.path.join(osm_data_path,'{}_{}.feather'.format(country,asset_type)))
    assets.geometry = assets.geometry.progress_apply(pygeos.from_wkb)
    
    # reproject
    assets['geometry'] = reproject(assets,current_crs="epsg:4326",approximate_crs ="epsg:3857")
    
    # make STRtree
    tree = pygeos.STRtree(assets.geometry.values)
    
    return assets,tree

def clip_assets_to_hazard_zone(assets,tree,convex_hull_hazard):
    """[summary]

    Args:
        assets ([type]): [description]
        tree ([type]): [description]
        convex_hull_hazard ([type]): [description]

    Returns:
        [type]: [description]
    """    
    # clip to hazard area
    assets = assets.iloc[tree.query(convex_hull_hazard,predicate='intersects')].reset_index(drop=True)
    
    return assets
    
def buffer_assets(assets,buffer_size=100):
    """[summary]

    Args:
        assets ([type]): [description]
        buffer_size (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """    
    assets['buffered'] = assets.geometry.progress_apply(lambda x: pygeos.buffer(x,buffer_size)) 

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
    if pygeos.get_type_id(assets.iloc[0].geometry) == 3:
        return  flooded_tree.query_bulk(assets.geometry,predicate='intersects')    
    else:
        return  flooded_tree.query_bulk(assets.buffered,predicate='intersects')


def get_depth(row,flood_dict,get_flood_points):
    """[summary]

    Args:
        row ([type]): [description]
        flood_dict ([type]): [description]
        get_flood_points ([type]): [description]

    Returns:
        [type]: [description]
    """    
    if row.name in flood_dict.keys():
        return get_flood_points.Xaver.iloc[flood_dict[row.name]]
    else:
        return 0

def get_damage_per_asset(asset,df_ds,assets,grid_size=90):
    """[summary]

    Args:
        asset ([type]): [description]
        df_ds ([type]): [description]
        assets ([type]): [description]
        grid_size (int, optional): [description]. Defaults to 90.

    Returns:
        [type]: [description]
    """    
    get_flood_points = df_ds.iloc[asset[1]['flood_point'].values].reset_index()
    get_flood_points.geometry= pygeos.buffer(get_flood_points.geometry,radius=grid_size/2,cap_style='square').values
    get_flood_points = get_flood_points.loc[pygeos.intersects(get_flood_points.geometry.values,assets.iloc[asset[0]].geometry)]
    
    vthreshold = 0
    vhalf = 0.5
    maxdam = 1000
    maxdam_asset = 100000

    if len(get_flood_points) == 0:
        return asset[0],0
    else:
        
        if pygeos.get_type_id(assets.iloc[asset[0]].geometry) == 1:
            get_flood_points['overlay_meters'] = get_flood_points.apply(lambda row: pygeos.length(pygeos.intersection(row.geometry,assets.iloc[asset[0]].geometry)),axis=1) 
            return asset[0],get_flood_points.apply(lambda x: get_fragility(x.Xaver,vthreshold,vhalf)*maxdam*x.overlay_meters,axis=1).sum()            
        elif  pygeos.get_type_id(assets.iloc[asset[0]].geometry) == 3:
            get_flood_points['overlay_m2'] = get_flood_points.apply(lambda row: pygeos.area(pygeos.intersection(row.geometry,assets.iloc[asset[0]].geometry)),axis=1) 
            return asset[0],get_flood_points.apply(lambda x: get_fragility(x.Xaver,vthreshold,vhalf)*maxdam*x.overlay_m2,axis=1).sum()          
        else:
            return asset[0],get_flood_points.apply(lambda x: get_fragility(x.Xaver,vthreshold,vhalf)*maxdam_asset,axis=1).sum()          

def damage_assessment(asset_type='railways',country='DEU',hazard_type='flood'):


    source_path = os.path.join("C:\Dropbox","VU","Projects","RECEIPT","receipt_storylines")
    osm_data_path = os.path.join(source_path,'osm_data')
    hazard_data_path = os.path.join(source_path,'hazard_data')

    tqdm.pandas()

    # load hazard file
    hazard_file = os.path.join(hazard_data_path,'Xaver-NorthernGermany','NorthGermany_MaxWaterDepth_dykes6.5m.tif')
    df_ds = read_flood_as_dataframe(hazard_file)
    convex_hull_hazard = rough_flood_extent(df_ds)

    if asset_type in ['railways','roads']:
        ### get line assets, clip and buffer them
        assets,tree = read_line_assets(osm_data_path,asset_type,country)
        assets = clip_assets_to_hazard_zone(assets,tree,convex_hull_hazard)
        assets = buffer_assets(assets,buffer_size=100)
    
    elif asset_type in ['airports']:
    ### get poly assets, clip and buffer them
        assets,tree = read_polygon_assets(osm_data_path,asset_type,country)
        assets = clip_assets_to_hazard_zone(assets,tree,convex_hull_hazard)
        
    elif asset_type in ['telecom']:
    ### get point assets, clip and buffer them
        assets,tree = read_point_assets(osm_data_path,asset_type,country)
        assets = clip_assets_to_hazard_zone(assets,tree,convex_hull_hazard)
        assets = buffer_assets(assets,buffer_size=100)

   # get STRtree for flood
    flood_overlay = pd.DataFrame(overlay_hazard_assets(df_ds,assets).T,columns=['asset','flood_point'])

    ### estimate damage
    collect_damages = []
    for asset in tqdm(flood_overlay.groupby('asset'),total=len(flood_overlay.asset.unique())):
        collect_damages.append(get_damage_per_asset(asset,df_ds,assets))
        
    damaged_assets = assets.merge(pd.DataFrame(collect_damages,columns=['index','damage']),left_index=True,right_on='index')

    return damaged_assets

if __name__ == "__main__":

    out = damage_assessment(asset_type='telecom',country='DEU',hazard_type='flood')

    print(out)
