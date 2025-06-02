
"""
Underlying code to calculate the damage to infrastrcture. 
  
@Author: Sadhana Nirandjan & Elco Koks  - Institute for Environmental studies, VU University Amsterdam
"""

import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
import shapely
from shapely.strtree import STRtree
from shapely.ops import transform
import csv
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import ast  
import os 

########################################################################################################################
################         functions          #############################################################
########################################################################################################################

def overlay_hazard_assets(df_ds,assets):
    """
    Overlay hazard assets on a dataframe of spatial geometries.
    Arguments:
        *df_ds*: GeoDataFrame containing the spatial geometries of the hazard data. 
        *assets*: GeoDataFrame containing the infrastructure assets.
    Returns:
        *geopandas.GeoSeries*: A GeoSeries containing the spatial geometries of df_ds that intersect with the infrastructure assets.
    """
    hazard_tree = shapely.STRtree(df_ds.geometry.values)
    if (shapely.get_type_id(assets.iloc[0].geometry) == 3) | (shapely.get_type_id(assets.iloc[0].geometry) == 6): # id types 3 and 6 stand for polygon and multipolygon
        return  hazard_tree.query(assets.geometry,predicate='intersects')    
    else:
        return  hazard_tree.query(assets.buffered,predicate='intersects')
    
def overlay_shapely(df1, df2):
    """
    Fast overlay using Shapely (equivalent to old pygeos functionality)
    Arguments:
        * df1: DataFrame with spatial data to be clipped
        * df2: DataFrame with spatial data for mask
        * spat_tree: Spatial tree with coordinates in Shapely geometry format
        
    Returns:
        DataFrame with clipped geometries
    """
    # Build a spatial tree from the geometries in df1
    spat_tree = STRtree(df1.geometry)
    
    # Filter features in df1 that intersect df2 geometries
    mask_union = df2.geometry.union_all()

    intersects_idx = spat_tree.query(mask_union, predicate="intersects")
    df1 = df1.iloc[intersects_idx] # use list with indices of df1
    
    return df1.reset_index(drop=True)

def crosses_shapely(df1, df2):
    """
    Fast overlay using Shapely (equivalent to old pygeos functionality)
    Arguments:
        * df1: DataFrame with spatial data to be clipped
        * df2: DataFrame with spatial data for mask
        * spat_tree: Spatial tree with coordinates in Shapely geometry format
        
    Returns:
        DataFrame with clipped geometries
    """
    # Build a spatial tree from the geometries in df1
    spat_tree = STRtree(df1.geometry)
    
    # Filter features in df1 that intersect df2 geometries
    mask_union = df2.geometry.union_all()
    #print('mask_union created')

    intersects_idx = spat_tree.query(mask_union, predicate="contains")
    #df1_contains = df1.iloc[intersects_idx] # use list with indices of df1 #geometries that crosses
    #df1_to_be_clipped = df1.drop(df1.index[intersects_idx]) #geometries that do not cross
    mask = df1.index.isin(intersects_idx)
    #print('create df1_crosses')
    df1_contains = df1[mask]
    #print('create df1_within')
    df1_to_be_clipped = df1[~mask]
    
    return df1_contains.reset_index(drop=True), df1_to_be_clipped.reset_index(drop=True)
    
def clip_shapely(df1, df2, chunk_size=10000):
    """
    Fast clipping using geopandas' clip function with progress bar
    Arguments:
        * df1: DataFrame with spatial data to be clipped
        * df2: DataFrame with spatial data for mask
        * chunk_size: Number of rows to process in each chunk
        
    Returns:
        DataFrame with clipped geometries
    """
    #print(gpd.options.use_pygeos) 
    #print('Perform geometry intersection')
    df1 = overlay_shapely(df1, df2)
    #print('Apply geometry contains predicate')
    df1, df1_to_be_clipped  = crosses_shapely(df1, df2) # Clip using the rewritten function
    
    # Initialize an empty list to store the clipped data
    clipped_list = []

    if not df1_to_be_clipped.empty: 
        # Use tqdm to track progress for chunks
        for start_idx in tqdm(range(0, len(df1_to_be_clipped), chunk_size), desc="Clipping geometries"):
            end_idx = min(start_idx + chunk_size, len(df1_to_be_clipped))
            df1_chunk = df1_to_be_clipped.iloc[start_idx:end_idx]  # Get a chunk of df1_to_be_clipped
    
            # Clip this chunk with df2
            clipped_chunk = gpd.clip(df1_chunk, df2)
    
            # Ensure the geometries are not empty
            clipped_chunk = clipped_chunk.loc[~clipped_chunk.geometry.is_empty]
    
            # Append the clipped chunk to the list
            clipped_list.append(clipped_chunk)
        
        # Concatenate all chunks back into a single GeoDataFrame
        df1_clipped = gpd.GeoDataFrame(pd.concat(clipped_list, ignore_index=True), crs=df1.crs)
        df1_clipped.reset_index(drop=True, inplace=True) # Reset index after clipping

        df1 = pd.concat([df1_clipped, df1], ignore_index=True)
        
    return df1

def clip_shapely_wo_progress(df1, df2):
    """
    Fast clipping using geopandas' clip function
    Arguments:
        * df1: DataFrame with spatial data to be clipped
        * df2: DataFrame with spatial data for mask
        
    Returns:
        DataFrame with clipped geometries
    """
    # Use geopandas' built-in clip function for faster operation
    df1 = gpd.clip(df1, df2)
    
    # Ensure the geometries are not empty
    df1 = df1.loc[~df1.geometry.is_empty]
    
    # Reset index after clipping
    df1.reset_index(drop=True, inplace=True)
    
    return df1

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
    Calculate exposure and damage for a given asset based on hazard information.
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
        # print(asset_geom.geom_type)
        # print(maxdam_asset)
        if asset_geom.geom_type in ['LineString', 'MultiLineString']:
            overlay_meters = shapely.length(shapely.intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell
            return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_meters*maxdam_asset), np.sum(overlay_meters) #return asset number, total damage for asset number (damage factor * meters * max. damage)
        elif asset_geom.geom_type in ['MultiPolygon','Polygon']:
            overlay_m2 = shapely.area(shapely.intersection(get_hazard_points[:,1],asset_geom))
            if '/unit' in unit_maxdam:
                converted_maxdam = maxdam_asset / shapely.area(asset_geom) #convert to maxdam/m2
                # print(f'This is converted maxdam{converted_maxdam}')
                # print(f'This is overlay_m2 {overlay_m2}')
                # print(f'This is the damage factor {np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values)}')
                return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_m2*converted_maxdam), np.sum(overlay_m2)
            else:
                return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*overlay_m2*maxdam_asset), np.sum(overlay_m2) #return asset number, total damage for asset number (damage factor * meters * max. damage)
        elif asset_geom.geom_type == 'Point':
            return np.sum((np.interp(np.float16(get_hazard_points[:,0]),hazard_intensity,fragility_values))*maxdam_asset), 1

def create_pathway_dict(data_path, flood_data_path, eq_data_path, landslide_data_path, cyclone_data_path, output_path):
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
                    'coastal': flood_data_path, 
                    'windstorm': cyclone_data_path, 
                    'earthquake': eq_data_path,
                    'earthquake_update': eq_data_path,
                    'landslide_rf': landslide_data_path,
                    'landslide_eq': landslide_data_path,
                    'output_path': output_path,}

    return pathway_dict

def find_folders_by_starting_name(root_dir, starting_name):
    """
    Recursively searches for directories within a root directory whose names start with a given prefix.

    Args:
        root_dir (Path): The root directory to start the search from (as a pathlib.Path object).
        starting_name (str): The prefix to match at the beginning of folder names.

    Returns:
        list: A list of matching folder names (not full paths) that start with the given prefix.
    """    
    # Use rglob to recursively search for directories starting with a specific a-2 code
    folder_names = [folder.name for folder in root_dir.rglob('*') 
                    if folder.is_dir() and folder.name.startswith(starting_name)]
    
    return folder_names


def get_fathom_files(root_dir, year, scenario=None):
    """
    Retrieves Fathom flood hazard .tif files based on year and scenario.

    For baseline year (2020), files are expected directly under the year folder.
    For future years (2030, 2050, 2080), files are located in subdirectories by scenario.

    Args:
        root_dir (Path): Root directory containing Fathom data folders.
        year (str): Year of interest ('2020', '2030', '2050', or '2080').
        scenario (str, optional): Climate scenario (e.g., 'SSP3_7.0') for future years.

    Returns:
        list: List of .tif file paths (as Path objects). Empty if directory is missing or inputs are invalid.
    """
    #years = [2020, 2030, 2050, 2080]
    #scenarios = ['SSP1_2.6', 'SSP2_4.5', 'SSP3_7.0', 'SSP5_8.5']

    if year == '2020':
        data_dir = root_dir / year
        if data_dir.exists():
            tif_files = [data_dir / file for file in os.listdir(data_dir) if file.endswith('.tif')]
        else:
            tif_files = []
            print('The following directory does not exist: {}'.format(data_dir))    
    elif year == '2030' or year == '2050' or year == '2080':
        data_dir = root_dir / year / scenario
        if data_dir.exists():
            tif_files = [data_dir / file for file in os.listdir(data_dir) if file.endswith('.tif')]    
        else:
            tif_files = []
            print('The following directory does not exist: {}'.format(data_dir))  
    else:
        print('ERROR: invalid input, please check your flood specifications')

    return tif_files

def get_future_ls_rf_file(landslide_rf_data, susceptibility_map_rf_ls): 
    """
    Returns the appropriate rainfall-triggered landslide susceptibility map 
    based on the specified future scenario.

    Args:
        landslide_rf_data (dict): Dictionary containing landslide metadata, including the 'scenario' key.
        susceptibility_map_rf_ls (Path): Path to the baseline susceptibility map.

    Returns:
        Path: Path to the scenario-specific susceptibility map file.
    """
    #if (landslide_data['scenario'] == None):
    #    continue
    if landslide_rf_data['scenario'] == 'SSP126':
        susceptibility_map_rf_ls = susceptibility_map_rf_ls.parent / 'susc_prec_trig_cdri_ssp126.tif'
    elif landslide_rf_data['scenario'] == 'SSP585':
        susceptibility_map_rf_ls = susceptibility_map_rf_ls.parent / 'susc_prec_trig_cdri_ssp585.tif'
        
    return susceptibility_map_rf_ls

def fathom_countries_dict(ISO3):
    """
    Maps a given ISO3 country code to the corresponding Fathom-2 country name used in 
    the dataset or file directory structure.

    This mapping accounts for naming conventions used in the Fathom flood hazard model data.
    Some entries return `None` if the country is not supported or has no mapped name in the dataset.
    The function also handles custom or merged regional codes used in specific contexts 
    (e.g., 'XKO' for Kosovo or 'XNC' for Cyprus).

    Args:
        ISO3 (str): The ISO 3166-1 alpha-3 country code (e.g., 'NLD', 'IND', 'BRA').

    Returns:
        str or None: The corresponding country name used in the Fathom dataset. Returns `None` 
                     if no mapping exists or the country is not included.

    Raises:
        KeyError: If the ISO3 code is not found in the dictionary.

    Notes:
        - 'Russia_east' should be merged with 'Russia' and 'Alaska', 'Alaska_West', 'Hawaii' with 'USA', 
        - Merging already completed for AUS, BRA, CAN, CHN (Hong_Kong + China), IDN, KAZ, FJI (Fiji_east + Fiji_west), and KIR (Kiribati_east + Kiribati_west)
    """

    fathom_countries_dict = {'IC': 'Spain', 
                             'ABW': 'Aruba', 
                             'AFG': 'Afghanistan', 
                             'AGO': 'Angola', 
                             'AIA': 'Anguilla', 
                             'ALA': 'finland', 
                             'ALB': 'albania', 
                             'AND': 'andorra', 
                             'ARE': 'UAE', 
                             'ARG': 'Argentina', 
                             'ARM': 'Armenia', 
                             'ASM': 'American_Samoa', 
                             'ATF': None, 
                             'ATG': 'Antigua_and_Barbuda', 
                             'AUS': 'Australia', 
                             'AUT': 'Austria', 
                             'AZE': 'Azerbaijan', 
                             'BDI': 'burundi', 
                             'BEL': 'belgium', 
                             'BEN': 'Benin', 
                             'BES': None, 
                             'BFA': 'Burkina', 
                             'BGD': 'Bangladesh', 
                             'BGR': 'Bulgaria', 
                             'BHR': 'Bahrain', 
                             'BHS': 'Bahamas', 
                             'BIH': 'bosnia',
                             'BLM': None, #Saint-Barthélemy
                             'BLR': 'belarus', 
                             'BLZ': 'Belize', 
                             'BMU': None, 
                             'BOL': 'Bolivia', 
                             'BRA': 'Brazil', 
                             'BRB': 'Barbados', 
                             'BRN': 'Brunei', 
                             'BTN': 'Bhutan', 
                             'BVT': None, #Bouvet Island
                             'BWA': 'Botswana', 
                             'CAF': 'Central_African_Republic', 
                             'CAN': 'Canada', 
                             'CCK': None, 
                             'CHE': 'switzerland', 
                             'CHL': 'Chile', 
                             'CHN': 'China', 
                             'CIV': "Cote_d_Ivoire", 
                             'CMR': 'Cameroon', 
                             'COD': 'Democratic_Republic_of_Congo', 'COG': 'Republic_of_Congo', 'COK': 'Cook_Islands', 'COL': 'Colombia', 'COM': 'Comoros', 
                             'CPV': 'Cape_Verde', 'CRI': 'Costa_Rica', 'CUB': 'Cuba', 'CUW': None, 'CXR': None, 
                             'CYM': 'Cayman_Islands', 'CYP': 'Cyprus', 'CZE': 'Czech_Republic', 'DEU': 'Germany', 'DJI': 'Djibouti', 
                             'DMA': 'Dominica', 'DNK': 'denmark', 'DOM': 'Dominican_Republic', 'DZA': 'Algeria', 'ECU': 'Ecuador', 
                             'EGY': 'Egypt', 'ERI': 'Eritrea', 'ESH': 'WestSahara', 'ESP': 'Spain', 'EST': 'estonia', 'ETH': 'Ethiopia', 
                             'FIN': 'finland', 'FJI': 'Fiji', 'FLK': 'Falkland_Islands', 'FRA': 'France', 'FRO': 'faroe_ids', 'FSM': 
                             'Micronesia', 'GAB': 'Gabon', 'GBR': None, 'GEO': 'Georgia', 'GGY': None, 'GHA': 'Ghana', 
                             'GIB': 'gibraltar', 'GIN': 'Guinea', 'GLP': 'Guadeloupe', 'GMB': 'Gambia', 'GNB': 'Guinea-Bissau', 
                             'GNQ': 'Equatorial_Guinea', 'GRC': 'greece', 'GRD': 'Grenada', 'GRL': None, 'GTM': 'Guatemala', 
                             'GUF': 'French_Guiana', 'GUM': None, #Guam
                             'GUY': 'Guyana', 
                             'HMD': None, #Heard Island and McDonald Island
                             'HND': 'Honduras', 'HRV': 'Croatia', 
                             'HTI': 'Haiti', 'HUN': 'Hungary', 'IDN': 'Indonesia', 
                             'IMN': None, # Isle of Man
                             'IND': 'India', 
                             'IOT': None, #British Indian Ocean Territory
                             'IRL': 'ireland', 'IRN': 'Iran', 'IRQ': 'Iraq', 'ISL': 'Iceland', 'ISR': 'Israel', 'ITA': 'Italy', 
                             'JAM': 'Jamaica', 'JEY': None, 'JOR': 'Jordan', 'JPN': 'Japan', 'KAZ': 'Kazakhstan', 'KEN': 'Kenya', 
                             'KGZ': 'Kyrgyzstan', 'KHM': 'cambodia', 'KIR': 'Kiribati', 'KNA': 'St_Kitts_and_Nevis', 'KOR': 'SouthKorea', 
                             'KWT': 'kuwait', 'LAO': 'laos', 'LBN': 'Lebanon', 'LBR': 'Liberia', 'LBY': 'Libya', 'LCA': 'St_Lucia', 'LIE': 
                             'liechstein', 'LKA': 'Sri_Lanka', 'LSO': 'Lesotho', 'LTU': 'lithuania', 'LUX': 'luxembourg', 'LVA': 'latvia', 
                             'MAF': None, 'MAR': 'Morocco', 'MCO': 'monaco', 'MDA': 'Moldova', 'MDG': 'Madagascar', 'MDV': 'Maldives', 
                             'MEX': 'Mexico', 'MHL': 'Marshall_Islands', 'MKD': 'macedonia', 'MLI': 'Mali', 'MLT': 'Malta', 'MMR': 'Myanmar', 'MNE': 
                             'montenegro', 'MNG': 'Mongolia', 
                             'MNP': None, #Northern Mariana Islands
                             'MOZ': 'Mozambique', 'MRT': 'Mauritania', 'MSR': 'Montserrat', 'MTQ': 'Martinique', 
                             'MUS': 'Mauritius', 'MWI': 'malawi', 'MYS': 'Malaysia', 'MYT': 'Mayotte', 'NAM': 'Namibia', 'NCL': 'New_Caledonia', 'NER': 'Niger', 
                             'NFK': None, 'NGA': 'Nigeria', 'NIC': 'Nicaragua', 'NIU': None, 'NLD': 'Netherlands', 'NOR': 'norway', 'NPL': 'nepal', 'NRU': None, 
                             'NZL': 'New_Zealand', 'OMN': 'Oman', 'PAK': 'Pakistan', 'PAN': 'Panama', 'PCN': None, 'PER': 'Peru', 'PHL': 'Philippines', 
                             'PLW': 'Palau', 'PNG': 'Papua_New_Guinea', 'POL': 'Poland', 'PRI': 'Puerto_Rico', 'PRK': 'NorthKorea', 'PRT': 'portugal', 'PRY': 'Paraguay', 
                             'PSE': 'Israel', 'PYF': 'French_Polynesia', 'QAT': 'Qatar', 'REU': 'Reunion', 'ROU': 'Romania', 'RUS': 'Russia', 'RWA': 'rwanda', 'SAU': 'saudiarabia', 
                             'SDN': 'Sudan_and_South_Sudan', 'SEN': 'Senegal', 'SGP': 'singapore', 'SGS': None, 
                             'SHN': None, # Saint Helena, Ascension and Tris
                             'SJM': None, 'SLB': 'Solomon_Islands', 
                             'SLE': 'sierraleone', 'SLV': 'El_salvador', 'SMR': 'sanmarino', 'SOM': 'Somalia', 'SPM': None, 'SRB': 'Serbia', 'SSD': 'Sudan_and_South_Sudan', 
                             'STP': 'Saotome', 'SUR': 'Suriname', 'SVK': 'slovakia', 'SVN': 'Slovenia', 'SWE': 'sweden', 'SWZ': 'Swaziland', 'SXM': None, 'SYC': 'Seychelles', 
                             'SYR': 'Syria', 'TCA': 'Turks_and_Caicos_Islands', 'TCD': 'Chad', 'TGO': 'Togo', 'THA': 'Thailand', 'TJK': None, 'TKL': None, 
                             'TKM': 'Turkmenistan', 'TLS': 'Timor-Leste', 'TON': 'Tonga', 'TTO': 'Trinidad_and_Tobago', 'TUN': 'Tunisia', 'TUR': 'Turkey', 'TUV': 'tuvalu', 
                             'TWN': 'Taiwan', 'TZA': 'tanzania', 'UGA': 'Uganda', 'UKR': 'Ukraine', 'UMI': None, 'URY': 'Uruguay', 
                             'USA': None, 'UZB': 'Uzbekistan', 
                             'VAT': 'vatican', 'VCT': 'St_Vincent_and_the_Grenadines', 'VEN': 'Venezuela', 'VGB': 'British_Virgin_Islands', 
                             'VIR': 'US_Virgin_Islands', 'VNM': 'Vietnam', 'VUT': 'Vanuatu', 'WLF': None, 'WSM': 'Samoa', 'XAD': 'Cyprus', 
                             'XCL': None, #Clipperton Islands
                             'XKO': 'kosovo', 
                             'XPI': None, # Paracel Islands
                             'XSP': None, # Spratly Islands
                             'YEM': 'Yemen', 'ZAF': 'South_Africa', 'ZMB': 'Zambia', 
                             'XNC': 'Cyprus', 
                             'ZWE': 'Zimbabwe'}
    
    return fathom_countries_dict[ISO3]

def read_hazard_data(hazard_data_path,data_path,hazard_type,ISO3,flood_data=None,eq_data=None,globalv3_own=True):
    """
    Reads hazard data files for a given hazard type and location.

    Parameters:
        hazard_data_path (Path): Base directory path where hazard data is stored.
        data_path (Path): Path to supporting data files (e.g., Excel metadata).
        hazard_type (str): Type of hazard to read data for. Supported types include:
            - 'fluvial', 'pluvial', 'coastal' (flooding hazards)
            - 'windstorm'
            - 'earthquake', 'earthquake_update'
            - 'landslide_rf', 'landslide_eq'
        ISO3 (str): ISO3 country code used to locate country-specific data.
        flood_data (dict, optional): Dictionary with flood data metadata such as:
            - 'version' (str): e.g., 'Fathom_v2' or 'Fathom_v3'
            - 'year' (int): Year of the flood data
            - 'scenario' (str): Scenario name, e.g., 'SSP1_2.6'
        eq_data (str, optional): Earthquake data source identifier, e.g., 'GAR', 'GIRI', or 'GEM'.
        globalv3_own (bool, optional): Flag indicating if Fathom v3 data is locally available.

    Returns:
        list of Path: List of Path objects pointing to hazard data files relevant to the specified hazard type.
    """
    if hazard_type in ['coastal', 'fluvial', 'pluvial']:
        if flood_data['version'] == 'Fathom_v2':
            fathom_code = fathom_countries_dict(ISO3)
            #fathom_code = country_df.loc[country_df['ISO_3digit'] == ISO3, 'Fathom_countries'].item()
            #adjust pathway to merged files
            if ISO3 in ['CHL', 'FJI', 'KIR', 'AUS', 'BRA', 'CAN', 'CHN', 'IDN', 'KAZ', 'RUS']: 
                hazard_data_path = Path('/scistor/ivm/')  / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'flooding' / 'fathomv2_merged'
        elif flood_data['version'] == 'Fathom_v3':
            if globalv3_own:
                if flood_data['year'] == 2020:
                    if hazard_type == 'coastal':
                        hazard_data = hazard_data_path / ISO3 / 'COASTAL_DEFENDED' 
                        if hazard_data.exists():
                            return list(hazard_data.iterdir())
                        else:
                            print('Please check whether COASTAL_DEFENDED maps are available for {}'.format(ISO3))
                            return []
                    elif hazard_type == 'pluvial':
                        hazard_data = hazard_data_path / ISO3 / 'PLUVIAL_DEFENDED' 
                        if hazard_data.exists():
                            return list(hazard_data.iterdir())
                        else:
                            print('Please check whether PLUVIAL_DEFENDED maps are available for {}'.format(ISO3))
                            return []
                    elif hazard_type == 'fluvial':
                        hazard_data = hazard_data_path / ISO3 / 'FLUVIAL_DEFENDED' 
                        if hazard_data.exists():
                            return list(hazard_data.iterdir())
                        else:
                            print('Please check whether FLUVIAL_DEFENDED maps are available for {}'.format(ISO3))
                            return []
                else:
                    print('Fathom v3 data is only available for historical conditions: {}'.format(ISO3))
            else:
                country_df = pd.read_excel(data_path / 'global_information_advanced_fathom_check.xlsx',sheet_name = 'Sheet1') # finalize this file and adjust name
                fathom_code = country_df.loc[country_df['ISO_3digit'] == ISO3, 'a-2'].item()

        if fathom_code is not None or hazard_type == 'coastal' and flood_data['version'] == 'Fathom_v2':
            if hazard_type == 'fluvial':
                if flood_data['version'] == 'Fathom_v2':
                    hazard_data = hazard_data_path / fathom_code / 'fluvial_undefended' 
                    if hazard_data.exists():
                        return list(hazard_data.iterdir())
                    else:
                        print('Please check whether FLUVIAL_UNDEFENDED maps are available for {}'.format(ISO3))
                        return []
                elif flood_data['version'] == 'Fathom_v3':
                    folders = find_folders_by_starting_name(hazard_data_path, '{}_'.format(fathom_code))
                    hazard_data = hazard_data_path / folders[0] / 'FLUVIAL_UNDEFENDED'
                    if hazard_data.exists():
                        return get_fathom_files(hazard_data, flood_data['year'], flood_data['scenario'])
                    else:
                        print('Please check whether FLUVIAL_UNDEFENDED maps are available for {}'.format(ISO3))
                        #hazard_data = hazard_data_path / folders[0] / 'FLUVIAL_DEFENDED'
                        #return get_fathom_files(hazard_data, flood_data['year'], flood_data['scenario'])
        
            elif hazard_type == 'pluvial':
                if flood_data['version'] == 'Fathom_v2':
                    hazard_data = hazard_data_path / fathom_code / 'pluvial' 
                    if hazard_data.exists():
                        return list(hazard_data.iterdir())
                    else:
                        print('Please check whether FLUVIAL_UNDEFENDED maps are available for {}'.format(ISO3))
                        return []
                elif flood_data['version'] == 'Fathom_v3':
                    folders = find_folders_by_starting_name(hazard_data_path, '{}_'.format(fathom_code))
                    hazard_data = hazard_data_path / folders[0] / 'PLUVIAL_DEFENDED' 
                    return get_fathom_files(hazard_data, flood_data['year'], flood_data['scenario'])
        
            elif hazard_type == 'coastal':
                if flood_data['version'] == 'Fathom_v2': #no coastal data available, so use aqueduct instead
                    hazard_data = Path('/scistor/ivm/')  / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'coastal_flooding' / 'baseline' # Aqueduct data
                    #hazard_data = Path(pathlib.Path('Z:') / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'coastal_flooding' / 'with_subsidence') # Aqueduct data
                    return list(hazard_data.iterdir())
                elif flood_data['version'] == 'Fathom_v3':
                    folders = find_folders_by_starting_name(hazard_data_path, '{}_'.format(fathom_code))
                    hazard_data = hazard_data_path / folders[0] / 'COASTAL_UNDEFENDED' 
                    return get_fathom_files(hazard_data, flood_data['year'], flood_data['scenario'])
        else:
            print(f'No folder exists for {ISO3}')
            return []
    
    elif hazard_type == 'windstorm':
        hazard_data = hazard_data_path 
        return list(hazard_data.iterdir())

    elif hazard_type in ['earthquake', 'earthquake_update']:
        if eq_data == 'GAR': 
            data_lst = []
            hazard_data = hazard_data_path
            rp_lst = list(hazard_data.iterdir())
            for rp_folder in rp_lst:
                temp_lst = [file for file in rp_folder.iterdir() if file.suffix == '.tif']
                data_lst.extend(temp_lst)  # Use extend instead of append to flatten the list
            return data_lst
        
        elif eq_data == 'GIRI': 
            hazard_data = hazard_data_path 
            return [file for file in hazard_data.iterdir() if file.suffix == '.tif']
            
        elif eq_data == 'GEM':
            hazard_data = hazard_data_path
            data_lst = list(hazard_data.iterdir())
            data_lst = [file for file in data_lst if file.suffix == '.csv']
            return data_lst

    #elif hazard_type == 'earthquake':
    #    hazard_data = hazard_data_path
    #    return list(hazard_data.iterdir())

    elif hazard_type == 'landslide_rf':
        hazard_data = hazard_data_path / 'rainfall' / 'global-norm-hist_180_180.tif' #'global-norm-hist.tif'  #'{}_l24-norm-hist.tif'.format(ISO3)
        return [hazard_data]

    elif hazard_type == 'landslide_eq':
        #use only one rp for the triggering conditions
        if eq_data == 'GAR': 
            hazard_data = hazard_data_path.parent / 'earthquakes' / 'GAR' / 'raw' / 'rp_475' / 'gar17pga475.tif'
        elif eq_data == 'GIRI': 
            hazard_data = hazard_data_path.parent / 'earthquakes' / 'GIRI' / 'PGA_475y.tif'
        elif eq_data == 'GEM':
            hazard_data = hazard_data_path.parent / 'earthquakes' / 'GEM' / 'v2023_2_PGA_rock_475.csv' 
        return [hazard_data]

def read_windstorm_map(wind_data_path,bbox,diameter_distance=0.1000000000000000056/2):
    """
    Load and clip windstorm data from a NetCDF file, convert wind speeds, and create buffered geometries.

    Args:
        wind_data_path (Path or str): Path to the NetCDF wind data file.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) in WGS84 coordinates.
        diameter_distance (float, optional): Buffer distance around points in degrees (default ~0.05).

    Returns:
        pandas.DataFrame: Wind speed data with buffered polygon geometries within the bbox.
    """
    # load data from NetCDF file
    with xr.open_dataset(wind_data_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], allow_one_dimensional_raster=True)
        ds['band_data'] = ds['band_data']/0.88*1.11 #convert 10-min sustained wind speed to 3-s gust wind speed: 0.88 to convert to 1-minute sustained and 1.11 to 3-second gust wind speed
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0)] 
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1) #already provided in m/s, no further conversions needed

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector

def read_flood_map_fathomv2(flood_map_path, bbox, diameter_distance=0.0008333333333333333868/2):
    """
    Load and clip flood map data from a NetCDF file and convert to buffered geometries.

    Args:
        flood_map_path (Path or str): Path to the flood map NetCDF file.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) in WGS84 coordinates.
        diameter_distance (float, optional): Buffer size for geometry creation (default ~0.00042).

    Returns:
        geopandas.GeoDataFrame: Flood data with buffered polygon geometries clipped to bbox.
    """
    # load data from NetCDF file
    with xr.open_dataset(flood_map_path, engine="rasterio") as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], allow_one_dimensional_raster=True)

        # Apply conditions to filter the data
        ds = ds.where((ds['band_data'] > 0) & 
                               (ds['band_data'] <= 1500) & 
                               (ds['band_data'] != 999), drop=True)


        # Handle NaN in band_data by setting x and y to NaN
        ds = ds.where(ds['band_data'].notnull(), other=np.nan)

        # Convert band_data to float16 for memory efficiency

        ds['band_data'] = ds['band_data'].astype(np.float16)
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        ds_vector = ds_vector.dropna(subset=['band_data']) #drop nans
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector.reset_index(drop=True)
    
def read_flood_map_aqueduct(flood_map_path, bbox, diameter_distance=0.008333333333333333218/2):
    """
    Load and clip Aqueduct flood map data from a NetCDF file, converting points to buffered geometries.

    Args:
        flood_map_path (Path or str): Path to the flood map NetCDF file.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) in WGS84 coordinates.
        diameter_distance (float, optional): Buffer size for geometry creation (default ~0.00417).

    Returns:
        geopandas.GeoDataFrame: Flood data with buffered polygon geometries clipped to bbox.
    """
    # load data from NetCDF file
    with xr.open_dataset(flood_map_path, engine="rasterio") as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], allow_one_dimensional_raster=True)
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 1500) & (ds_vector.band_data != 999)]
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector.reset_index(drop=True)

def read_flood_map_fathomv3(flood_map_path, bbox, diameter_distance=0.000277777777777780488/2):
    """
    Load and clip Fathom v3 flood map data from a NetCDF file, converting depths to meters and creating buffered geometries.

    Args:
        flood_map_path (Path or str): Path to the flood map NetCDF file.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) in WGS84 coordinates.
        diameter_distance (float, optional): Buffer size for geometry creation (default ~0.00014).

    Returns:
        geopandas.GeoDataFrame: Flood data with depth in meters and buffered polygon geometries clipped to bbox.
    """
    # load data from NetCDF file
    with xr.open_dataset(flood_map_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 1500)]
        ds_vector['band_data'] = ds_vector['band_data']/100 #to meters
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector
    
def read_flood_map_fathomv2_gridded(flood_map_path, bbox, country_border_geometries, diameter_distance=0.0008333333333333333868/2):
    """
    Read and grid Fathom v2 flood map data clipped to a bounding box and country borders, returning buffered geometries.

    Args:
        flood_map_path (Path or str): Path to the flood map NetCDF file.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) in WGS84.
        country_border_geometries (GeoSeries or GeoDataFrame): Country boundary geometries for clipping.
        diameter_distance (float, optional): Buffer size for geometry creation (default ~0.00083/2).

    Returns:
        geopandas.GeoDataFrame or pandas.DataFrame: Flood data points with buffered geometries within the country borders.
    """
    #read hazard data and clip
    hazard_map = xr.open_dataset(flood_map_path, engine="rasterio")
    try:
        hazard_country = hazard_map.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], allow_one_dimensional_raster=True)
    except Exception as e:
        if "No data found in bounds. Data variable: band_data" in str(e):
            print("No data found in bounds. Returning empty dataframe.")
            return pd.DataFrame(columns=['band_data', 'geometry'])
        else:
            raise

    #make grid based on hazard_map
    gridded = create_grid(shapely.box(hazard_country.rio.bounds()[0],hazard_country.rio.bounds()[1],hazard_country.rio.bounds()[2],
                                          hazard_country.rio.bounds()[3]),1)

    # Create a buffer around the country geometry
    country_border_geometries = country_border_geometries.buffer(0.001)  # rio.clip may exclude cells that are not entirely within the border
        
    # get all bounds
    all_bounds = gpd.GeoDataFrame(gridded,columns=['geometry']).bounds
    
    # create empty dataframe and empty list
    df = pd.DataFrame(columns=['band_data', 'geometry'])
    df_list = []
    
    # loop over all grids
    for bounds in tqdm(all_bounds.itertuples(),total=len(all_bounds)):
        # subset hazard
        subset_hazard = hazard_country.rio.clip_box(
        minx=bounds.minx,
        miny=bounds.miny,
        maxx=bounds.maxx,
        maxy=bounds.maxy,
        allow_one_dimensional_raster=True)
    
        subset_hazard['band_data'] = subset_hazard.band_data.rio.write_nodata(np.nan, inplace=True)
        subset_hazard['band_data'] = subset_hazard['band_data'].astype(np.float16)

        try:
            subset_hazard = subset_hazard.rio.clip(country_border_geometries.geometry, country_border_geometries.crs)
        except Exception as e: #to catch cases where there is no overlay
            continue
        
        #remove data that will not be used
        ds_vector = subset_hazard['band_data'].to_dataframe().reset_index() #transform to dataframe
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 1500) & (ds_vector['band_data'] != 999)]
    
        if not ds_vector.empty:
            #print("Filtered ds_vector shape:", ds_vector.shape)  # Print shape 
            ds_vector = ds_vector.drop(['band','spatial_ref'],axis=1)
            df_list.append(ds_vector)

    if df_list:
        # create geometry values and drop lat lon columns    
        df = pd.concat(df_list, ignore_index=True)
        df['geometry'] = [shapely.points(x) for x in list(zip(df['x'],df['y']))]
        df = df.drop(['x','y'],axis=1)
        df['geometry'] = shapely.buffer(df.geometry, distance=diameter_distance, cap_style='square').values
        df = df.drop_duplicates(subset=['geometry']) #avoid duplicates!
    else:
        df = pd.DataFrame(columns=['geometry'])

    return df  

def read_flood_map_fathomv3_gridded(flood_map_path, bbox, country_border_geometries, diameter_distance=0.000277777777777780488/2):
    """
    Read and grid Fathom v3 flood map data clipped by bbox and country borders, returning buffered geometries.

    Args:
        flood_map_path (Path or str): Path to flood map NetCDF file.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) in WGS84.
        country_border_geometries (GeoSeries or GeoDataFrame): Country boundary geometries for clipping.
        diameter_distance (float, optional): Buffer size for geometry creation (default ~0.00028/2).

    Returns:
        geopandas.GeoDataFrame or pandas.DataFrame: Flood data points with buffered geometries within country borders.
    """
    #read hazard data and clip
    hazard_map = xr.open_dataset(flood_map_path, engine="rasterio")
    try:
        hazard_country = hazard_map.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
    except Exception as e:
        if "No data found in bounds. Data variable: band_data" in str(e):
            print("No data found in bounds. Returning empty dataframe.")
            return pd.DataFrame(columns=['band_data', 'geometry'])
        else:
            raise

    #make grid based on hazard_map
    gridded = create_grid(shapely.box(hazard_country.rio.bounds()[0],hazard_country.rio.bounds()[1],hazard_country.rio.bounds()[2],
                                          hazard_country.rio.bounds()[3]),0.5)
        
    # Create a buffer around the country geometry
    country_border_geometries = country_border_geometries.buffer(0.001)  # rio.clip may exclude cells that are not entirely within the border

    # get all bounds
    all_bounds = gpd.GeoDataFrame(gridded,columns=['geometry']).bounds
    
    # create empty dataframe and empty list
    df = pd.DataFrame(columns=['band_data', 'geometry'])
    df_list = []
    
    # loop over all grids
    for bounds in tqdm(all_bounds.itertuples(),total=len(all_bounds)):
        # subset hazard
        subset_hazard = hazard_country.rio.clip_box(
        minx=bounds.minx,
        miny=bounds.miny,
        maxx=bounds.maxx,
        maxy=bounds.maxy,
        allow_one_dimensional_raster=True)
    
        subset_hazard['band_data'] = subset_hazard.band_data.rio.write_nodata(np.nan, inplace=True)
        subset_hazard['band_data'] = subset_hazard['band_data']/100 #to meters
        subset_hazard['band_data'] = subset_hazard['band_data'].astype(np.float16)

        try:
            subset_hazard = subset_hazard.rio.clip(country_border_geometries.geometry, country_border_geometries.crs)
        except Exception as e: #to catch cases where there is no overlay
            continue
        
        #remove data that will not be used
        ds_vector = subset_hazard['band_data'].to_dataframe().reset_index() #transform to dataframe
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0) & (ds_vector.band_data <= 1500)]
    
        if not ds_vector.empty:
            print("Filtered ds_vector shape:", ds_vector.shape)  # Print shape 
            # create geometry values and drop lat lon columns
            ds_vector = ds_vector.drop(['band','spatial_ref'],axis=1)
            df_list.append(ds_vector)

    if df_list:
        # create geometry values and drop lat lon columns    
        df = pd.concat(df_list, ignore_index=True)
        df['geometry'] = [shapely.points(x) for x in list(zip(df['x'],df['y']))]
        df = df.drop(['x','y'],axis=1)
        df['geometry'] = shapely.buffer(df.geometry, distance=diameter_distance, cap_style='square').values
        df = df.drop_duplicates(subset=['geometry']) #avoid duplicates!
    else:
        df = pd.DataFrame(columns=['geometry'])
    
    return df  

def reduce_precision(geometry, precision=3):
    """Reduces the precision of a geometry's coordinates."""
    project = lambda x, y, z=None: (round(x, precision), round(y, precision))
    return transform(project, geometry)

def buffer_and_union(assets, buffer_distance=0.01, simplify_tolerance=0.1, precision=3, n_chunks=8, num_cores=-1):
    """
    Buffer, reduce precision, union, and simplify geometries in parallel.

    Args:
        assets (GeoDataFrame): Input geometries to process.
        buffer_distance (float): Buffer distance around geometries.
        simplify_tolerance (float): Tolerance for geometry simplification.
        precision (int): Decimal places to round geometry coordinates.
        n_chunks (int): Number of chunks for parallel processing.
        num_cores (int): CPU cores to use (-1 = all).

    Returns:
        shapely.Geometry: Unioned and simplified geometry.
    """
    # Apply the reduced precision to the geometries
    assets['geometry'] = assets['geometry'].apply(reduce_precision, precision=precision)
    
    # Function to process each chunk
    def process_chunk(chunk):
        return chunk.geometry.buffer(buffer_distance).unary_union.simplify(simplify_tolerance)

    # Split the data into chunks
    chunks = np.array_split(assets, n_chunks)

    # Parallel processing of the chunks
    results = Parallel(n_jobs=num_cores)(delayed(process_chunk)(chunk) for chunk in chunks)

    # Combine the results from the chunks and perform final union
    unioned_roads = gpd.GeoSeries(results).unary_union
    
    return unioned_roads

def create_grid(bbox,height):
    """
    Create a grid of square polygons covering the given bounding box.

    Args:
        bbox (shapely.geometry): Geometry or bounding box to cover.
        height (float): Size of each square grid cell.

    Returns:
        list[shapely.geometry.Polygon]: List of square polygons forming the grid.
    """
    # set xmin,ymin,xmax,and ymax of the grid
    xmin, ymin = shapely.total_bounds(bbox)[0],shapely.total_bounds(bbox)[1]
    xmax, ymax = shapely.total_bounds(bbox)[2],shapely.total_bounds(bbox)[3]
    
    #estimate total rows and columns
    rows = int(np.ceil((ymax-ymin) / height))
    cols = int(np.ceil((xmax-xmin) / height))

    # set corner points
    x_left_origin = xmin
    x_right_origin = xmin + height
    y_top_origin = ymax
    y_bottom_origin = ymax - height

    # create actual grid
    res_geoms = []
    for countcols in range(cols):
        y_top = y_top_origin
        y_bottom = y_bottom_origin
        for countrows in range(rows):
            res_geoms.append((
                ((x_left_origin, y_top), (x_right_origin, y_top),
                (x_right_origin, y_bottom), (x_left_origin, y_bottom)
                )))
            y_top = y_top - height
            y_bottom = y_bottom - height
        x_left_origin = x_left_origin + height
        x_right_origin = x_right_origin + height

    # return grid as shapely polygons
    return shapely.polygons(res_geoms)

def read_giri_earthquake_map(earthquake_map_path,bbox,diameter_distance=0.004999972912597225316/2):
    """
    Load and process earthquake hazard data from a NetCDF file within a bounding box.

    Args:
        earthquake_map_path (Path): Path to the NetCDF earthquake map file.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) to clip data.
        diameter_distance (float, optional): Buffer size for creating square geometries around points.

    Returns:
        geopandas.GeoDataFrame: Processed earthquake hazard data with buffered geometries.
    """     
    # load data from NetCDF file
    with xr.open_dataset(earthquake_map_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], allow_one_dimensional_raster=True)

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
    """
    Load and process GAR earthquake hazard data from a NetCDF file clipped to a bounding box.

    Args:
        earthquake_map_path (Path): Path to the NetCDF earthquake hazard file.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) for clipping the data.
        diameter_distance (float, optional): Buffer size to create square geometries around points.

    Returns:
        geopandas.GeoDataFrame: Processed earthquake hazard data with buffered geometries.
    """    
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

def h3_to_polygon(h3_index):
    """
    Convert an H3 index to a Shapely Polygon.

    Args:
        h3_index (str): H3 hexagon index.

    Returns:
        shapely.geometry.Polygon: Polygon of the hex boundary.
    """   
    # Get the boundary of the hexagon in (lat, lon) pairs
    boundary = h3.h3_to_geo_boundary(h3_index)
    # Convert to (lon, lat) pairs and create a Polygon
    return shapely.Polygon([(lon, lat) for lat, lon in boundary])

def overlay_hazard_bbox(df_ds,bbox_bbox):
    """
    Filter hazard geometries intersecting a bounding box.

    Args:
        df_ds (GeoDataFrame): Hazard geometries.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy).

    Returns:
        GeoDataFrame: Filtered geometries intersecting bbox.
    """
    bbox_polygon = shapely.box(*bbox) #create polygon using bbox coordinates
    
    #overlay 
    hazard_tree = shapely.STRtree(df_ds.geometry.values)
    intersect_index = hazard_tree.query(bbox_polygon,predicate='intersects')
    
    return df_ds.iloc[intersect_index].reset_index(drop=True)

def read_earthquake_map_csv(earthquake_map_path,bbox):
    """
    Read earthquake data from a CSV with lat/lon and band_data columns,
    filter by band_data values, clip to bbox, and convert points to H3 hex polygons.

    Args:
        earthquake_map_path (str or Path): Path to CSV file with earthquake data.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) to clip data.

    Returns:
        GeoDataFrame: Data clipped to bbox with geometries as H3 hexagons.
    
    Notes:
        - using h3 geometries: https://pypi.org/project/h3/
        - example Notebooks: https://github.com/uber/h3-py-notebooks
        - more info: https://h3geo.org/docs/quickstart
    """ 
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
    """
    Read landslide map data from a NetCDF file and process it into a GeoDataFrame with buffered geometries.
    
    Args:
        landslide_map_path (str or Path): Path to the NetCDF file containing landslide map data.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) to clip the data.
        diameter_distance (float, optional): Buffer distance around each point (default is ~0.0041666 degrees).
    
    Returns:
        GeoDataFrame: Data filtered, clipped, and with buffered square geometries.
    """    
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
         susc_footprint = pathlib.Path(landslide_map_path).parent.parent.parent / 'landslides' / 'susceptibility_giri' / 'susc_earthquake_trig_cdri.tif'
    elif hazard_type in ['landslide_rf']:
         susc_footprint = pathlib.Path(landslide_map_path).parent.parent / 'susceptibility_giri' / 'susc_prec_trig_cdri.tif'
     
    # load data from NetCDF file
    with xr.open_dataset(susc_footprint) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], allow_one_dimensional_raster=True)
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 1) & (ds_vector.band_data <= 5)]  #also omit class 1 in this early phase, because won't be needed anyway following table in GIRI report
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector

def read_susceptibility_map_gridded(landslide_map_path, hazard_type, bbox, country_border_geometries, diameter_distance=0.0008333333333333522519/2):
    """
    Reads a landslide susceptibility map for a given hazard type, clips by bounding box, 
    filters susceptibility classes, and buffers points into polygons.

    Args:
        landslide_map_path (str or Path): Reference path to derive susceptibility footprint path.
        hazard_type (str): Type of landslide hazard ('landslide_eq' or 'landslide_rf').
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) to clip the data.
        diameter_distance (float, optional): Buffer distance for points to polygons.

    Returns:
        GeoDataFrame: Filtered susceptibility points buffered as polygons.
    """
    if hazard_type in ['landslide_eq']:
        susc_footprint = pathlib.Path(landslide_map_path).parent.parent.parent / 'landslides' / 'susceptibility_giri' / 'susc_earthquake_trig_cdri.tif'
    elif hazard_type in ['landslide_rf']:
        susc_footprint = pathlib.Path(landslide_map_path).parent.parent / 'susceptibility_giri' / 'susc_prec_trig_cdri.tif'
     
    #read hazard data and clip
    susc_map = xr.open_dataset(susc_footprint, engine="rasterio")
    susc_country = susc_map.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], allow_one_dimensional_raster=True)
    
    #make grid based on susc_map
    gridded = create_grid(shapely.box(susc_country.rio.bounds()[0],susc_country.rio.bounds()[1],susc_country.rio.bounds()[2],
                                          susc_country.rio.bounds()[3]),15)

    # Create a buffer around the country geometry
    country_border_geometries = country_border_geometries.buffer(0.001)  # rio.clip may exclude cells that are not entirely within the border
        
    # get all bounds
    all_bounds = gpd.GeoDataFrame(gridded,columns=['geometry']).bounds
    
    # create empty dataframe and empty list
    df = pd.DataFrame(columns=['band_data', 'geometry'])
    df_list = []
    
    # loop over all grids
    for bounds in tqdm(all_bounds.itertuples(),total=len(all_bounds)):
        # subset hazard
        subset_hazard = susc_country.rio.clip_box(
        minx=bounds.minx,
        miny=bounds.miny,
        maxx=bounds.maxx,
        maxy=bounds.maxy,
        allow_one_dimensional_raster=True)
    
        subset_hazard['band_data'] = subset_hazard.band_data.rio.write_nodata(np.nan, inplace=True)

        try:
            subset_hazard = subset_hazard.rio.clip(country_border_geometries.geometry, country_border_geometries.crs)
        except Exception as e: #to catch cases where there is no overlay
            continue
        
        #remove data that will not be used
        ds_vector = subset_hazard['band_data'].to_dataframe().reset_index() #transform to dataframe
        ds_vector = ds_vector.loc[(ds_vector.band_data > 1) & (ds_vector.band_data <= 5)]
    
        if not ds_vector.empty:
            print("Filtered ds_vector shape:", ds_vector.shape)  # Print shape 
            # create geometry values and drop lat lon columns
            ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
            ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)
    
            ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values
    
            df_list.append(ds_vector)

    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        df = df.drop_duplicates(subset=['geometry']) #avoid duplicates!
    else:
        df = pd.DataFrame(columns=['geometry'])
    
    return df  

def read_susceptibility_map_cropped(susc_path, diameter_distance=0.0008333333333333522519/2):
    """
    Reads a cropped susceptibility map from a NetCDF file and returns points 
    with susceptibility classes filtered (2 to 5).
    
    Args:
        susc_path (str or Path): Path to the NetCDF susceptibility dataset.
        diameter_distance (float, optional): Buffer distance around points (currently not applied).
        
    Returns:
        pd.DataFrame: DataFrame with filtered susceptibility points and geometry as shapely Points.
    """     
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
    
def create_damage_csv_without_exposure(damage_output, hazard_type, pathway_dict, country_code, sub_system, hazard_scenario=None):
    """
    Create a CSV file containing damage information.

    Args:
        damage_output (dict): Dictionary containing damage information keyed by tuples.
        hazard_type (str): Type of hazard (e.g., 'earthquake', 'flood', 'landslide_eq').
        pathway_dict (dict): Dictionary containing file paths, including 'output_path'.
        country_code (str): Country code string.
        sub_system (str): Subsystem considered.
        hazard_scenario (str, optional): Scenario identifier for hazard (default is None).

    Returns:
        None
    """
    if hazard_type in ['landslide_eq', 'landslide_rf']:

        hazard_output_path = pathway_dict['output_path'] / 'damage' / country_code
        hazard_output_path.mkdir(parents=True, exist_ok=True)
        
        ## Check if the directory exists
        #if not hazard_output_path.exists():
        #    # Create the directory
        #    hazard_output_path.mkdir(parents=True, exist_ok=True)
        
        if hazard_scenario != None:
            csv_file_path = hazard_output_path / '{}_{}{}_{}.csv'.format(country_code, hazard_type, hazard_scenario, sub_system)
        else:
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
        hazard_output_path = pathway_dict['output_path'] / 'damage' / country_code
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
    
    Args:
        asset: Tuple where asset[1]['hazard_point'] are indices into hazard_numpified.
        hazard_numpified: Array or list with hazard data; geometries may be separate.
        asset_geom: Shapely geometry of the asset.
        hazard_intensity: NumPy array of hazard intensities (not used directly here).
        fragility_values: NumPy array of damage factors for the asset.
        maxdam_asset: Maximum damage value for the asset (float or str convertible).
        unit_maxdam: Unit descriptor for max damage, e.g. '/unit' indicates per unit area.
    
    Returns:
        np.ndarray: 2D array with damage and return periods, or empty array if no overlay.
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
    
    Args:
        asset: Tuple, where asset[1]['hazard_point'] holds indices for hazard_numpified.
        hazard_numpified: NumPy array of [return_period, geometry] or similar structure.
        asset_geom: Shapely geometry of the asset.
        hazard_intensity: Array of hazard intensities or special indicator like ['Exposure to hazard'].
        fragility_values: Array of fragility damage factors corresponding to hazard intensities.
        maxdam_asset: Maximum damage value for asset.
        unit_maxdam: Unit descriptor of max damage (e.g., '/unit' means per unit area).
    
    Returns:
        float or np.ndarray: Damage value(s) for the asset.
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
    Load vulnerability curves and maximum damage data for a given hazard and infrastructure type.

    Parameters:
        data_path (Path or str): Base directory with vulnerability data files.
        hazard_type (str): Hazard type (e.g., 'pluvial', 'windstorm', 'earthquake').
        infra_type (str): Infrastructure type (e.g., 'primary', 'railway').
        database_id_curves (bool): If True, read curve IDs from database; else use defaults.
        database_maxdam (bool): If True, read max damage from database; else use defaults.

    Returns:
        tuple: (vulnerability_curves_df, max_damage_series, units_series)
    """
    vul_data = data_path / 'Vulnerability'
    
    # Load assumptions file containing curve - maxdam combinations per infrastructure type
    if hazard_type in ['pluvial','fluvial','coastal']: 
        assumptions = pd.read_excel(vul_data / 'S1_Assumptions_Test.xlsx',sheet_name = 'Flooding assumptions',header=[1])
    elif hazard_type == 'windstorm':
        assumptions = pd.read_excel(vul_data / 'S1_Assumptions_Test.xlsx',sheet_name = 'Windstorm assumptions',header=[1])
    elif hazard_type in ['earthquake', 'earthquake_update']:
        assumptions = pd.read_excel(vul_data / 'S1_Assumptions_Test.xlsx',sheet_name = 'Earthquake assumptions',header=[1])
    elif hazard_type in ['landslide_eq', 'landslide_rf']:
        assumptions = pd.read_excel(vul_data / 'S1_Assumptions_Test.xlsx',sheet_name = 'Landslide assumptions',header=[1])

    if database_id_curves==False:
        #get assumptions from dictionary
        if hazard_type in ['earthquake', 'earthquake_update']:
            if infra_type in ['unclassified', 'primary', 'secondary', 'tertiary', 'residential', 
                                    'trunk', 'trunk_link',  'motorway','motorway_link',  'primary_link','secondary_link', 'tertiary_link','road', 'track' ]:
                assump_curves = ['E7.1', 'E7.6', 'E7.7', 'E7.8', 'E7.9', 'E7.10', 'E7.11', 'E7.12', 'E7.13', 'E7.14' ]
            elif infra_type in ['railway']:
                assump_curves = ['E8.11', 'E8.16','E8.17','E8.18','E8.19','E8.20','E8.21','E8.22','E8.23','E8.24']        
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
    if hazard_type in ['pluvial','fluvial','coastal']:  
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_conversions.xlsx',sheet_name = 'F_Vuln_Depth',index_col=[0],header=[0,1,2,3,4])
        infra_curves =  curves[assump_curves]
    elif hazard_type == 'windstorm':
        curves = pd.read_excel(vul_data / 'Table_D2_Hazard_Fragility_and_Vulnerability_Curves_V1.1.0_conversions.xlsx',sheet_name = 'W_Vuln_V10m_3sec',index_col=[0],header=[0,1,2,3,4])
        infra_curves =  curves[assump_curves]
    elif hazard_type in ['earthquake', 'earthquake_update']:
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
        #following database
        maxdam_dict = {'cable': 1594,
                         'distribution_line': 89,
                         'transmission_line': 160,
                         'plant': 136651800,
                         'substation': 16704521,
                         'power_tower': 91145,
                         'power_pole': 85619,
                         'primary':997, 
                         'secondary':397, 
                         'tertiary':238, 
                         'trunk':1089, 
                         'motorway':2539, 
                         'residential':119,
                         'road':119,
                         'track':119,
                         'railway': 606,
                         'airport': 119,
                         'terminal':144,
                         'runway': 4833,
                         'water_treatment_plant' :88606067 ,
                         'water_storage_tank': 708849,
                         'reservoir_covered': 708849 ,
                         'water_well': 354424 ,
                         'water_tower': 708849 ,
                         'waste_transfer_station':679477,
                         'wastewater_treatment_plant' :65028019 ,
                         'communication_tower':133714, 
                         'mast':67204,
                         'clinic':1135,
                         'doctors':1135,
                         'hospital':1135,
                         'dentist':1135,
                         'pharmacy':1135,
                         'physiotherapist':1135,
                         'alternative':1135,
                         'laboratory':1135,
                         'optometrist':1135,
                         'rehabilitation':1135,
                         'blood_donation':1135,
                         'birthing_center':1135,
                         'college':625,
                         'kindergarten':625,
                         'library':625,
                         'school':625,
                         'university':625,
                      
                      }
        
        unit_dict = {
            'cable': 'euro/m',
            'distribution_line': 'euro/m',
            'transmission_line': 'euro/m',
            'plant': 'euro/unit',
            'substation': 'euro/unit',
            'power_tower': 'euro/unit',
            'power_pole': 'euro/unit',
            'primary': 'euro/m',
            'secondary': 'euro/m',
            'tertiary': 'euro/m',
            'trunk': 'euro/m',
            'motorway': 'euro/m',
            'residential': 'euro/m',
            'road': 'euro/m',
            'track': 'euro/m',
            'railway': 'euro/m',
            'airport': 'euro/m2',
            'terminal':'euro/m2',
            'runway': 'euro/m',
            'water_treatment_plant' : 'euro/unit',
            'water_storage_tank': 'euro/unit' ,
            'reservoir_covered': 'euro/unit' ,
            'water_well': 'euro/unit' ,
            'water_tower': 'euro/unit',
            'waste_transfer_station':'euro/unit',
            'wastewater_treatment_plant': 'euro/unit',
            'communication_tower': 'euro/unit',
            'mast': 'euro/unit',
            'clinic':'euro/m2',
            'doctors':'euro/m2',
            'hospital':'euro/m2',
            'dentist':'euro/m2',
            'pharmacy':'euro/m2',
            'physiotherapist':'euro/m2',
            'alternative':'euro/m2',
            'laboratory':'euro/m2',
            'optometrist':'euro/m2',
            'rehabilitation':'euro/m2',
            'blood_donation':'euro/m2',
            'birthing_center':'euro/m2',
            'college':'euro/m2',  
            'kindergarten':'euro/m2',
            'library':'euro/m2',
            'school':'euro/m2',
            'university':'euro/m2',   
        }
         
        # Retrieve the appropriate values
        infra_maxdam = pd.Series([str(maxdam_dict[infra_type])], index=['default'])
        infra_maxdam.name = 'Amount'
        infra_units = pd.Series([unit_dict[infra_type]], index=['default'])
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
    Update asset exposure data based on landslide return period susceptibility classification.

    Parameters:
        overlay_rf (DataFrame): Landslide return period classifications with 'cond_classes'.
        get_susc_data (list or array): Susceptibility data, first element used for logic.
        overlay_assets (DataFrame): Asset data including 'hazard_point' to update.
        susc_point (tuple): Hazard point identifier used to filter assets.

    Returns:
        DataFrame: Modified overlay_assets with updated return periods or removed entries.
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
    Adjust asset exposure based on earthquake susceptibility classification.

    Parameters:
        overlay_eq (DataFrame): Earthquake classification data with 'cond_classes'.
        get_susc_data (list or array): Susceptibility indicator, first element used.
        overlay_assets (DataFrame): Asset data including 'hazard_point' to update.
        susc_point (tuple): Hazard point identifier for filtering assets.

    Returns:
        DataFrame: Updated overlay_assets with adjusted return periods or removed entries.
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

def filter_landslide_rf_rps(trig_rp, overlay_assets_ls_rp):
    """
    Update landslide asset return periods based on a given rainfall-trigger return period.

    Parameters:
        trig_rp (int): Triggering return period to filter and reassign.
        overlay_assets_ls_rp (DataFrame): Asset data with current return periods.

    Returns:
        DataFrame: Asset data with updated return periods for the trigger event.
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
                                                                                                    np.where((return_period_trig == 200) & (return_period == 33), 20, 
                                                                                                             np.where((return_period_trig == 200) & (return_period == 20), 10, 
                                                                                                                      np.where((return_period_trig == 200) & (return_period == 10), 7, 
                                                                                                                               np.where((return_period_trig == 200) & (return_period == 7), 5, 
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
    Return geometries from df_ds that intersect with the given country boundary.

    Parameters:
        df_ds (GeoDataFrame): Hazard spatial geometries.
        country_border_geometries (GeoDataFrame): Country boundary geometries.

    Returns:
        GeoDataFrame: Subset of df_ds intersecting the boundary.
    """
    #overlay 
    hazard_tree = shapely.STRtree(df_ds.geometry.values)
    if (shapely.get_type_id(country_border_geometries.iloc[0]) == 3) | (shapely.get_type_id(country_border_geometries.iloc[0]) == 6): # id types 3 and 6 stand for polygon and multipolygon
        intersect_index = hazard_tree.query(country_border_geometries.geometry,predicate='intersects')
    intersect_index = np.unique(np.concatenate(intersect_index))
    
    return df_ds.iloc[intersect_index].reset_index(drop=True)

def overlay_hazard_assetbuffer(df_ds,assetbuffer):
    """
    Return geometries from df_ds that intersect with the given asset buffer.

    Parameters:
        df_ds (GeoDataFrame): Hazard spatial geometries.
        assetbuffer (Geometry or GeoSeries): Buffer geometry around assets.

    Returns:
        GeoDataFrame: Subset of df_ds intersecting the asset buffer.
    """
    #overlay 
    hazard_tree = shapely.STRtree(df_ds.geometry.values)
    intersect_index = hazard_tree.query(assetbuffer,predicate='intersects')
    
    return df_ds.iloc[intersect_index].reset_index(drop=True)

def read_rainfall_map(rf_data_path,bbox,diameter_distance=0.25/2):
    """
    Load and process rainfall data from a NetCDF file within a bounding box.

    Parameters:
        rf_data_path (str): Path to the rainfall NetCDF file.
        bbox (tuple): Bounding box (minx, miny, maxx, maxy) to clip the data.
        diameter_distance (float, optional): Buffer distance for geometry points (default is 0.125).

    Returns:
        GeoDataFrame: DataFrame with buffered point geometries and rainfall values > 0.
    """     
    # load data from NetCDF file
    with xr.open_dataset(rf_data_path) as ds:
        
        # convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], allow_one_dimensional_raster=True)
        
        ds_vector = ds['band_data'].to_dataframe().reset_index() #transform to dataframe
        
        #remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 0)] 
        
        # create geometry values and drop lat lon columns
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x','y','band','spatial_ref'],axis=1)

        ds_vector['geometry'] = shapely.buffer(ds_vector.geometry, distance=diameter_distance, cap_style='square').values

        return ds_vector

def landslide_damage_and_overlay(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type,
                                 maxdams,infra_units,geom_dict,country_code,sub_system,infra_type,collect_output):
    """
    Calculate landslide damage, exposure, and occurrence by overlaying hazard data with assets.

    Args:
        overlay_assets (GeoDataFrame): Assets with hazard overlay information, including return periods.
        infra_curves (dict): Fragility curves for infrastructure damage assessment.
        susc_numpified (DataFrame): Numerical susceptibility data for assets.
        assets_infra_type (DataFrame): Asset information including infrastructure type and geometry.
        hazard_type (str): Type of landslide hazard ('landslide_eq' or 'landslide_rf').
        maxdams (array-like): Array of maximum damage values to consider.
        infra_units (list): Units corresponding to max damage values.
        geom_dict (dict): Mapping of asset IDs to their geometries.
        country_code (str): Country identifier for output indexing.
        sub_system (str): Subsystem identifier for output indexing.
        infra_type (str): Infrastructure type for output indexing.
        collect_output (dict): Dictionary to accumulate aggregated damage outputs.

    Returns:
        tuple: (collect_output, damages_collection_trig_rp, trig_rp_lst)
            - collect_output (dict): Aggregated damage by country, return period, and other indices.
            - damages_collection_trig_rp (dict): Detailed asset-level damages, exposure, and landslide counts per trigger RP.
            - trig_rp_lst (list): Sorted list of landslide trigger return periods processed.
    """
    trig_rp_lst = sorted(overlay_assets['return_period_trig'].unique()) #get list of unique RPs for landslide trigger
    damages_collection_trig_rp = {key: {} for key in trig_rp_lst}
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
                asset_damages_per_curve_rp = asset_damages_per_curve_rp.join(assets_infra_type[['osm_id']], how='left')
                asset_damages_per_curve_rp['return_period_trig'] = trig_rp
                asset_damages_per_curve_rp['return_period_landslide'] = rp
                osm_id_column = asset_damages_per_curve_rp.pop('osm_id')
                asset_damages_per_curve_rp.insert(0, 'osm_id', osm_id_column)
                collect_asset_damages_per_curve_rp[rp] = asset_damages_per_curve_rp
        
        damages_collection_trig_rp[trig_rp] = collect_asset_damages_per_curve_rp
            
    return collect_output, damages_collection_trig_rp, trig_rp_lst

def landslide_damage_and_overlay_per_asset(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type,
                                 maxdams,infra_units,geom_dict,country_code,sub_system,infra_type,pathway_dict,collect_output,hazard_scenario):
    """
    Compute landslide damage, exposure, and occurrence at the asset level, then save detailed results.

    Args:
        overlay_assets (GeoDataFrame): Assets with landslide hazard overlay including trigger return periods.
        infra_curves (dict): Fragility curves indexed by infrastructure curve IDs.
        susc_numpified (DataFrame): Susceptibility data for assets.
        assets_infra_type (DataFrame): Asset information with infrastructure type and geometry.
        hazard_type (str): Landslide hazard type ('landslide_eq' or 'landslide_rf').
        maxdams (array-like): Maximum damage values to evaluate.
        infra_units (list): Units corresponding to max damage values.
        geom_dict (dict): Mapping of asset IDs to their geometries.
        country_code (str): Country identifier for output paths and indexing.
        sub_system (str): Subsystem identifier.
        infra_type (str): Infrastructure type identifier.
        pathway_dict (dict): Dictionary of file paths for saving output.
        collect_output (dict): Dictionary for accumulating aggregated damage results.
        hazard_scenario (str): Scenario identifier for hazard type.

    Returns:
        dict: Updated collect_output with aggregated damage results.
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
                save_path = pathway_dict['output_path'] / 'damage' / country_code / f'{hazard_type}{hazard_scenario}'  / f'{country_code}_{hazard_type}_ls{rp}_trig{trig_rp}_{sub_system}_{infra_type}.parquet'
                save_path.parent.mkdir(parents=True, exist_ok=True)
                damaged_assets.to_parquet(save_path)

    return collect_output

def get_damage_and_overlay_per_asset_rp(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam_asset,unit_maxdam):
    """
    Calculate damage and overlay metrics for a specific asset based on hazard exposure.

    Args:
        asset (tuple): Asset info, including identifier (asset[0]) and hazard points (asset[1]['hazard_point']).
        hazard_numpified (ndarray): Array of hazard data points.
        asset_geom (shapely geometry): Geometry of the asset.
        hazard_intensity (ndarray): Hazard intensity values from fragility curve.
        fragility_values (ndarray): Fragility values corresponding to hazard intensities.
        maxdam_asset (float or str): Maximum damage value for the asset.
        unit_maxdam (str): Unit of the maximum damage value.

    Returns:
        tuple of ndarray: 
            - Damage estimates per hazard overlay and their return periods.
            - Overlay measures (area, length, or count) and their return periods.
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
        if asset_geom.geom_type in ['LineString', 'MultiLineString']:
            overlay_meters = shapely.length(shapely.intersection(get_hazard_points[:,1],asset_geom)) # get the length of exposed meters per hazard cell
            #damage = np.float16(np.full(len(get_hazard_points[:,0]), fragility_values[0])) * overlay_meters * maxdam_asset
            damage = np.full(len(get_hazard_points[:,0]), fragility_values[0]) * overlay_meters * maxdam_asset
            return np.vstack([damage, get_hazard_points[:,0]]), np.vstack([overlay_meters, get_hazard_points[:,0]])

        elif asset_geom.geom_type in ['MultiPolygon','Polygon']:
            overlay_m2 = shapely.area(shapely.intersection(get_hazard_points[:,1],asset_geom))
            if '/unit' in unit_maxdam:
                converted_maxdam = maxdam_asset / shapely.area(asset_geom) #convert to maxdam/m2
                damage = np.full(len(get_hazard_points[:,0]), fragility_values[0]) * overlay_m2 * converted_maxdam
                return np.vstack([damage, get_hazard_points[:,0]]), np.vstack([overlay_m2, get_hazard_points[:,0]])
            else:
                damage = np.full(len(get_hazard_points[:,0]), fragility_values[0]) * overlay_m2 * maxdam_asset
                return np.vstack([damage, get_hazard_points[:,0]]), np.vstack([overlay_m2, get_hazard_points[:,0]])
        elif asset_geom.geom_type in ['Point']:
            damage = np.full(len(get_hazard_points[:,0]),  fragility_values[0]) * maxdam_asset
            overlay = np.ones_like(get_hazard_points[:, 0])
            return np.vstack([damage, get_hazard_points[:,0]]), np.vstack([overlay, get_hazard_points[:,0]])

def read_liquefaction_map(liquefaction_map_path, bbox, vertical_diameter_distance=0.01051720562427702239/2, horizontal_diameter_distance=0.01083941445811754771/2):
    """
    Load and process liquefaction raster data from a NetCDF file within a bounding box, converting points to rectangular polygons.

    Args:
        liquefaction_map_path (str or Path): Path to the liquefaction NetCDF raster file.
        bbox (tuple): Bounding box as (minx, miny, maxx, maxy) in WGS84 coordinates.
        vertical_diameter_distance (float, optional): Half-height of rectangular buffer. Defaults to ~0.00526.
        horizontal_diameter_distance (float, optional): Half-width of rectangular buffer. Defaults to ~0.00542.

    Returns:
        geopandas.GeoDataFrame: DataFrame with filtered liquefaction points converted to rectangular polygon geometries.
    """
    from shapely import affinity
    
    # Load data from NetCDF file
    with xr.open_dataset(liquefaction_map_path) as ds:
        
        # Convert data to WGS84 CRS
        ds.rio.write_crs(4326, inplace=True)
        ds = ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], allow_one_dimensional_raster=True)
        
        # Transform to DataFrame
        ds_vector = ds['band_data'].to_dataframe().reset_index()
        
        # Remove data that will not be used
        ds_vector = ds_vector.loc[(ds_vector.band_data > 1)]
        
        # Create geometry values
        ds_vector['geometry'] = [shapely.points(x) for x in list(zip(ds_vector['x'],ds_vector['y']))]
        ds_vector = ds_vector.drop(['x', 'y', 'band', 'spatial_ref'], axis=1)

        # Create rectangular buffers
        def create_rectangle(point):
            # Create a rectangular buffer using cap_style='square'
            buffered = point.buffer(1, cap_style=3) # Create a square buffer
            # Scale the square buffer to the desired rectangle size
            return affinity.scale(buffered, 
                                  xfact=horizontal_diameter_distance, 
                                  yfact=vertical_diameter_distance, 
                                  origin='center')

        # Apply the function to create rectangles
        ds_vector['geometry'] = ds_vector['geometry'].apply(create_rectangle)

        return ds_vector.reset_index(drop=True)
        
def overlay_dataframes(df1,df2):
    """
    Return geometries from df1 that intersect with any geometry in df2.

    Args:
        df1 (GeoDataFrame): GeoDataFrame with spatial geometries to filter.
        df2 (GeoDataFrame): GeoDataFrame with spatial geometries to overlay.

    Returns:
        GeoDataFrame: Subset of df1 with geometries intersecting df2.
    """
    if df2.empty:
        return pd.DataFrame(columns=df1.columns)
    else:
        #overlay 
        hazard_tree = shapely.STRtree(df1.geometry.values)
        intersect_index = hazard_tree.query(df2.geometry.values, predicate='intersects')
        intersect_index = np.unique(intersect_index[1])
        return df1.iloc[intersect_index].reset_index(drop=True)

def eq_liquefaction_matrix_liqres(hazard_map,cond_map):
    """
    Filter earthquake hazard cells by applying a liquefaction-conditional matrix at liquefaction map resolution.

    Args:
        hazard_map (GeoDataFrame): Earthquake hazard data with intensity values.
        cond_map (GeoDataFrame): Liquefaction condition data.

    Returns:
        GeoDataFrame: Filtered liquefaction map with relevant earthquake hazard data.
    """
    bins = [0, 0.092, 0.18, 0.34, 0.65, float('inf')]  # Adjust the thresholds as needed
    labels = ['1', '2', '3', '4', '5']
    
    ## Create a new column 'classes' based on the thresholds
    hazard_map['classes'] = pd.cut(hazard_map['band_data'], bins=bins, labels=labels, right=False, include_lowest=True)

    overlay_hazardpoints = pd.DataFrame(overlay_hazard_assets(hazard_map, cond_map).T, 
                                        columns=['cond_point', 'hazard_point']) #get df with overlays of liquefaction cells with hazards cells

    # Convert DataFrame to numpy array
    hazard_numpified = hazard_map.to_numpy()
    cond_numpified = cond_map.to_numpy()
    cond_points = []
    pga_lst = []

    hazard_classes = hazard_numpified[:, 2]  # get haz class
    pga_values = hazard_numpified[:, 0]  # get PGA
    cond_classes = cond_numpified[:, 0]      # get con class
    
    # Get the hazard and condition class pairs
    for i, con_point in tqdm(enumerate(overlay_hazardpoints['cond_point'].unique())):
        # Get liquefaction category for cond point
        cond_class = cond_classes[con_point]
    
        # Get all haz_points associated with this cond point
        haz_points = overlay_hazardpoints[overlay_hazardpoints['cond_point'] == con_point]['hazard_point']
        
        # Get the lowest cond point value for this hazard point
        eq_class = hazard_classes[haz_points].min()
    
        # Build condition to decide whether to drop the hazard point
        if not (
            (eq_class == '1' and cond_class in [2, 3, 4, 5]) or
            (eq_class == '2' and cond_class in [2, 3, 4]) or
            (eq_class == '3' and cond_class in [2, 3]) or
            (eq_class == '4' and cond_class == 2)
        ):
            cond_points.append(con_point) 
            pga_lst.append(pga_values[haz_points].min())


    haz_map_filtered = cond_map.iloc[cond_points] 
    haz_map_filtered['band_data'] = pga_lst
    
    return haz_map_filtered 

def eq_liquefaction_matrix_liqres_new(hazard_map, cond_map):
    """
    Apply earthquake and liquefaction matrix and drop irrelevant hazard cells, 
    outputting at liquefaction map resolution.
    Arguments:
        *hazard_map*: GeoDataFrame containing earthquake data. 
        *cond_map*: GeoDataFrame containing liquefaction data.
    Returns:
        *geopandas.DataFrame*: A dataframe containing relevant earthquake data.
    """
    # Define bins and labels for hazard classification
    bins = [0, 0.092, 0.18, 0.34, 0.65, float('inf')]  # Adjust thresholds as needed
    labels = ['1', '2', '3', '4', '5']
    
    # Create a new column 'classes' in hazard_map based on thresholds
    hazard_map['classes'] = pd.cut(hazard_map['band_data'], bins=bins, labels=labels, right=False, include_lowest=True)

    # Get overlays between hazard_map and cond_map
    overlay_hazardpoints = overlay_hazard_assets(hazard_map, cond_map)
    overlay_df = pd.DataFrame(
        overlay_hazardpoints.T, columns=['cond_point', 'hazard_point']
    )
    
    # Convert hazard and condition maps to numpy arrays for fast access
    hazard_classes = hazard_map['classes'].to_numpy()
    pga_values = hazard_map['band_data'].to_numpy()
    cond_classes = cond_map['band_data'].to_numpy()

    # Process each unique cond_point in bulk using groupby
    results = []
    grouped = overlay_df.groupby('cond_point')

    for cond_point, group in grouped:
        cond_class = cond_classes[cond_point]  # Get condition class for the cond_point
        haz_points = group['hazard_point'].to_numpy()  # Get associated hazard points
        
        # Get the minimum hazard class for this cond_point
        eq_class = hazard_classes[haz_points].min()

        # Check if this cond_point should be retained
        if not (
            (eq_class == '1' and cond_class in [2, 3, 4, 5]) or
            (eq_class == '2' and cond_class in [2, 3, 4]) or
            (eq_class == '3' and cond_class in [2, 3]) or
            (eq_class == '4' and cond_class == 2)
        ):
            results.append({
                'cond_point': cond_point,
                'band_data': pga_values[haz_points].min()  # Minimum PGA value for the group
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Filter cond_map to keep only the relevant points
    haz_map_filtered = cond_map.iloc[results_df['cond_point']].copy()
    haz_map_filtered['band_data'] = results_df['band_data'].to_numpy()
    
    return haz_map_filtered

def create_damage_csv(damage_output, hazard_type, pathway_dict, country_code, sub_system):
    """
    Create a CSV file containing damage information.

    Arguments:
        damage_output (dict): Dictionary with keys as tuples and values as tuples/lists containing damage data.
        hazard_type (str): Type of hazard (e.g., 'earthquake', 'flood').
        pathway_dict (dict): Dictionary with file paths (expects 'data_path' key with Path object).
        country_code (str): Country code string.
        sub_system (str): Subsystem name.

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

