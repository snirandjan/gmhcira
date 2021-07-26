import os,sys
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import pygeos
from tqdm import tqdm
import pyproj
import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.mask import mask
from rasterio.features import shapes
from shapely.geometry import mapping
from multiprocessing import Pool,cpu_count
from functools import partial

def paths():

    source_path = os.path.join("C:\Dropbox","VU","Projects","RECEIPT","receipt_storylines")
    osm_data_path = os.path.join(source_path,'osm_data')
    hazard_data_path = os.path.join(source_path,'hazard_data')

    return source_path,osm_data_path,hazard_data_path

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

def reproject_assets(df_ds,current_crs="epsg:3857",approximate_crs = "epsg:4326"):
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

def clip_hazard_maps():

    source_path,osm_data_path,hazard_data_path = paths()

    region_data_path = os.path.join(source_path,'region_data')

    for event in ['Xaver','Xynthia','EmiliaRomagna']:
  
        nuts2_regions = gpd.read_file(os.path.join(region_data_path,'exposed_nuts2.shp'))
        
        if event in ['Xaver','Xynthia']:
            nuts2_regions = nuts2_regions.to_crs(epsg=3857)
        else:
            nuts2_regions = nuts2_regions.to_crs(epsg=5659)

        event_nuts = nuts2_regions.loc[nuts2_regions.event==event].dissolve().buffer(10000)
        geoms = [mapping(event_nuts.geometry.iloc[0])]

        if event == 'Xaver':
            hazard_file = os.path.join(hazard_data_path,'Xaver-NorthernGermany','NorthGermany_MaxWaterDepth_dykes6.5m.tif')
        elif event == 'Xynthia':
            hazard_file = os.path.join(hazard_data_path,'Xynthia-WesternFrance','WestFrance_depth.tif')
        elif event == 'EmiliaRomagna':
            hazard_file = os.path.join(hazard_data_path,'EmiliaRomagnaRegion','RER_depth.tif')

        with rasterio.open(hazard_file) as src:
            out_image, out_transform = rasterio.mask.mask(src, geoms , crop=True)
            out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "compress": "LZW"})

        with rasterio.open(os.path.join(hazard_data_path,'events',"{}.tif".format(event)), "w", **out_meta) as dest:
            dest.write(out_image)     

def load_flood_as_dataframe(hazard_file,country='DEU'):
    """[summary]

    Args:
        hazard_file ([type]): [description]

    Returns:
        [type]: [description]
    """    
    with xr.open_rasterio(hazard_file) as ds:

        if country == 'DEU':
            df_ds = ds.to_dataframe(name='Xaver').reset_index()
            df_ds = df_ds.rename(columns={'Xaver': 'hazard_intensity'})
        elif country == 'FRA':
            df_ds = ds.to_dataframe(name='Xynthia').reset_index()
            df_ds = df_ds.rename(columns={'Xynthia': 'hazard_intensity'})
        elif country == 'ITA':
            df_ds = ds.to_dataframe(name='Xaver').reset_index()
            df_ds = df_ds.rename(columns={'Xaver': 'hazard_intensity'})                

        df_ds = df_ds.loc[(df_ds.hazard_intensity != -9999) & (df_ds.hazard_intensity != 0)].reset_index(drop=True)
        df_ds['geometry'] = pygeos.points(df_ds.x,y=df_ds.y)

    return df_ds

#         ddata = dd.from_pandas(df_ds, npartitions=32)

#         with ProgressBar():
#             df_ds['geometry'] = ddata.map_partitions(lambda df_ds: df_ds.apply((lambda row: get_geoms(row)),axis=1),meta=('object')).compute(scheduler='processes') 
       

def rough_flood_extent(df_ds):
    """[summary]

    Args:
        df_ds ([type]): [description]

    Returns:
        [type]: [description]
    """
    return pygeos.envelope(pygeos.multipoints(df_ds['geometry'].values))


def load_assets(osm_data_path,asset_type='airports',country='DEU',reproject=True):
    """[summary]

    Args:
        asset_type (str, optional): [description]. Defaults to 'airports'.
        country (str, optional): [description]. Defaults to 'DEU'.

    Returns:
        [type]: [description]
    """    
    # load data
    assets = pd.read_feather(os.path.join(osm_data_path,'{}_{}.feather'.format(country,asset_type)))
    assets.geometry = pygeos.from_wkb(assets.geometry.values) #assets.geometry.progress_apply(pygeos.from_wkb)
    
    # reproject
    if reproject:
        if country in ['FRA','DEU']:
            assets['geometry'] = reproject_assets(assets,current_crs="epsg:4326",approximate_crs ="epsg:3857")
        else:
            assets['geometry'] = reproject_assets(assets,current_crs="epsg:4326",approximate_crs ="epsg:5659")

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
    if (pygeos.get_type_id(assets.iloc[0].geometry) == 3) | (pygeos.get_type_id(assets.iloc[0].geometry) == 6):
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

def get_damage_per_asset(asset,df_ds,assets,curves,maxdam,grid_size=90):
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

    
    asset_type = assets.iloc[asset[0]].asset
    maxdam_asset = maxdam.loc[asset_type].MaxDam
    
    water_depths = curves[asset_type].index.values
    fragility_values = curves[asset_type].values
    
    asset_geom = assets.iloc[asset[0]].geometry
        
    if len(get_flood_points) == 0:
        return asset[0],0
    else:
        
        if pygeos.get_type_id(asset_geom) == 1:
            get_flood_points['overlay_meters'] = pygeos.length(pygeos.intersection(get_flood_points.geometry.values,asset_geom))
            
            return asset[0],np.sum((np.interp(get_flood_points.hazard_intensity.values*100,water_depths,fragility_values))*get_flood_points.overlay_meters*maxdam_asset)
        
        elif  pygeos.get_type_id(asset_geom) == 3:
            get_flood_points['overlay_m2'] = pygeos.area(pygeos.intersection(get_flood_points.geometry.values,asset_geom))
            
            return asset[0],get_flood_points.apply(lambda x: np.interp(x.hazard_intensity*100, water_depths, fragility_values)*maxdam_asset*x.overlay_m2,axis=1).sum()     
        
        else:
            return asset[0],np.sum((np.interp(get_flood_points.hazard_intensity.values*100,water_depths,fragility_values))*maxdam_asset)     

def load_curves_maxdam(data_path): 
    """[summary]

    Args:
        data_path ([type]): [description]

    Returns:
        [type]: [description]
    """
    # load curves and maximum damages as separate inputs
    curves = pd.read_excel(data_path,sheet_name='flooding_curves',skiprows=8,index_col=[0])
    maxdam=pd.read_excel(data_path,sheet_name='flooding_curves',index_col=[0]).iloc[:5]
    
    curves.columns = maxdam.columns

    #transpose maxdam so its easier work with the dataframe
    maxdam = maxdam.T

    #interpolate the curves to fill missing values
    curves = curves.interpolate()

    # fill maxdam damages when value is unknown ### THIS SHOULD BE REMOVED WHEN ALL VALUES ARE KNOWN
    maxdam.MaxDam = maxdam.MaxDam.fillna(1000)
    
    return curves,maxdam

def damage_assessment(asset_type='railways',country='DEU',hazard_type='flood',**kwargs):
    """[summary]

    Args:
        asset_type (str, optional): [description]. Defaults to 'railways'.
        country (str, optional): [description]. Defaults to 'DEU'.
        hazard_type (str, optional): [description]. Defaults to 'flood'.
    """

    source_path,osm_data_path,hazard_data_path = paths()

    print('Damage Assessment for {} started in {}'.format(asset_type,country))

    if 'hazard' in kwargs:
         df_ds = kwargs['hazard']
    else:
        # load hazard file
        if country == 'DEU':
            hazard_file = os.path.join(hazard_data_path,'events','Xaver.tif')
        elif country == 'FRA':
            hazard_file = os.path.join(hazard_data_path,'events','Xynthia.tif')
        elif country == 'ITA':
            hazard_file = os.path.join(hazard_data_path,'events','EmiliaRomagna.tif')

        df_ds = load_flood_as_dataframe(hazard_file)
        print('Hazard data loaded for {} in {}'.format(asset_type,country))

    convex_hull_hazard = rough_flood_extent(df_ds)

    # load curves and maxdam
    curves,maxdam = load_curves_maxdam(data_path=os.path.join('..','data','infra_vulnerability_data.xlsx'))

    point_assets = ['health','telecom']
    polygon_assets = ['airports','educational_facilities','waste_solid','waste_water','water_supply']
    line_assets = ['railways','roads']

    assets,tree = load_assets(osm_data_path,asset_type,country)
    assets = clip_assets_to_hazard_zone(assets,tree,convex_hull_hazard)

    if (asset_type in line_assets) | (asset_type in point_assets):
        ### get line assets, clip and buffer them
        assets = buffer_assets(assets,buffer_size=100)
    
    elif asset_type == 'power':
        power_lines = buffer_assets(assets.loc[assets.asset.isin(['cable','minor_cable','line','minor_line'])],buffer_size=100).reset_index(drop=True)
        power_poly = assets.loc[assets.asset.isin(['plant','substation'])].reset_index(drop=True)
        power_points = buffer_assets(assets.loc[assets.asset.isin(['power_tower','power_pole'])],buffer_size=100).reset_index(drop=True)

    print('Asset data loaded for {} in {}'.format(asset_type,country))

    if asset_type != 'power':
        # get STRtree for flood
        flood_overlay = pd.DataFrame(overlay_hazard_assets(df_ds,assets).T,columns=['asset','flood_point'])

        ### estimate damage
        collect_damages = []
        for asset in tqdm(flood_overlay.groupby('asset'),total=len(flood_overlay.asset.unique()),desc='{} in {} damage calculation'.format(asset_type,country)):
            collect_damages.append(get_damage_per_asset(asset,df_ds,assets,curves,maxdam))

        damaged_assets = assets.merge(pd.DataFrame(collect_damages,columns=['index','damage']),left_index=True,right_on='index')

        if (asset_type in line_assets) | (asset_type in point_assets):
            damaged_assets = damaged_assets.drop(['buffered'],axis=1) 

    else:
        # lines
        flood_overlay_lines = pd.DataFrame(overlay_hazard_assets(df_ds,power_lines).T,columns=['asset','flood_point'])
  
        collect_line_damages = []
        for asset in tqdm(flood_overlay_lines.groupby('asset'),total=len(flood_overlay_lines.asset.unique()),desc='{} lines in {} damage calculation'.format(asset_type,country)):
            collect_line_damages.append(get_damage_per_asset(asset,df_ds,power_lines,curves,maxdam))

        damaged_lines = power_lines.merge(pd.DataFrame(collect_line_damages,columns=['index','damage']),left_index=True,right_on='index')
        damaged_lines = damaged_lines.drop(['buffered'],axis=1) 

        # polygons
        flood_overlay_poly = pd.DataFrame(overlay_hazard_assets(df_ds,power_poly).T,columns=['asset','flood_point'])
  
        collect_poly_damages = []
        for asset in tqdm(flood_overlay_poly.groupby('asset'),total=len(flood_overlay_poly.asset.unique()),desc='{} poly in {} damage calculation'.format(asset_type,country)):
            collect_poly_damages.append(get_damage_per_asset(asset,df_ds,power_poly,curves,maxdam))
        
        damaged_poly = assets.merge(pd.DataFrame(collect_poly_damages,columns=['index','damage']),left_index=True,right_on='index')

        # points
        flood_overlay_points = pd.DataFrame(overlay_hazard_assets(df_ds,power_points).T,columns=['asset','flood_point'])
  
        collect_point_damages = []
        for asset in tqdm(flood_overlay_points.groupby('asset'),total=len(flood_overlay_points.asset.unique()),desc='{} point in {} damage calculation'.format(asset_type,country)):
            collect_point_damages.append(get_damage_per_asset(asset,df_ds,power_points,curves,maxdam))
        
        damaged_points = power_points.merge(pd.DataFrame(collect_point_damages,columns=['index','damage']),left_index=True,right_on='index')
        damaged_points = damaged_points.drop(['buffered'],axis=1) 

        damaged_assets = pd.concat([damaged_lines,damaged_poly,damaged_points]).reset_index(drop=True)

    damaged_assets.to_csv(os.path.join(source_path,'damage_data','{}_{}.csv'.format(asset_type,country)))

    return damaged_assets

def event_assessment(event='Xaver',parallel=True):

    source_path,osm_data_path,hazard_data_path = paths()

    if event == 'Xaver':
        country = 'DEU'
        hazard_file = os.path.join(hazard_data_path,'Xaver-NorthernGermany','NorthGermany_MaxWaterDepth_dykes6.5m.tif')
    elif event == 'Xynthia':
        country = 'FRA'
        hazard_file = os.path.join(hazard_data_path,'Xynthia-WesternFrance','WestFrance_depth.tif')
    elif event == 'EmiliaRomagna':
        country = 'ITA'
        hazard_file = os.path.join(hazard_data_path,'EmiliaRomagnaRegion','RER_depth.tif')

    all_assets = ['airports','education_facilities','health','power','railways','roads','telecom','waste_solid','waste_water','water_supply']

    if parallel:
        with Pool(cpu_count()-2) as pool: 
            out = pool.map(partial(damage_assessment, country=country), all_assets)
    else:

        df_ds = load_flood_as_dataframe(hazard_file)

        out = []
        for asset_type in all_assets:
            out.append(damage_assessment(asset_type=asset_type,country=country,hazard_type='flood',hazard=df_ds))

    # collect total damage per asset and return that
    get_total_damages = []
    for asset in out:
        get_total_damages.append(pd.DataFrame(asset.groupby(by='asset').sum()['damage']).reset_index())

    df_total_damages = pd.concat(get_total_damages)
    df_total_damages.to_csv(os.path.join(source_path,'damage_data','total_damage_{}.csv'.format(event)))

    return df_total_damages

if __name__ == "__main__":

    event_damage_per_asset = (event_assessment(event='EmiliaRomagna',parallel=True))
    print(event_damage_per_asset)