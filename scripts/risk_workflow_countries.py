"""
Code to calculate the damage to infrastrcture for multiple hazards using hazard overlays and 
vulnerability functions. It supports parallel processing and handles multiple hazard types (earthquake, flood, landslide, tropical cyclones). 

@Author: Sadhana Nirandjan & Elco Koks  - Institute for Environmental studies, VU University Amsterdam
"""

from pathlib import Path
import traceback
import concurrent.futures
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from tqdm import tqdm

import osm_extract
import damage_functions

########################################################################################################################
###########################          define paths          #############################################################
########################################################################################################################

base_path = Path('/scistor/ivm/') 
liquefaction_data_path = base_path / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'liquefaction' # liquefaction
susceptibility_map_eq_ls = base_path / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'landslides' / 'susceptibility_giri' / 'susc_earthquake_trig_cdri.tif' # pathway to susceptibility map for earthquake-triggered landslides
susceptibility_map_rf_ls = base_path / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'landslides' / 'susceptibility_giri' / 'susc_prec_trig_cdri.tif' # pathway to susceptibility map for rainfall-triggered landslides

########################################################################################################################
########################          Damage assessment        #############################################################
########################################################################################################################

def damage_calculation(country_code, pathway_dict, cis_dict, hazard_types, eq_data, flood_data, landslide_rf_data, vuln_settings, overwrite=False):
    """
    Calculates hazard-related damages to critical infrastructure at the subsystem level.

    This function overlays infrastructure assets with hazard data (e.g., earthquake, flood,
    landslide) and applies vulnerability functions to estimate damages. Exposure and damage results are disaggregated
    by vulnerability curve and hazard return period, and saved as Parquet files.

    Args:
        country_code (str): ISO3 country code.
        pathway_dict (dict): Dictionary with paths to input and output directories.
        cis_dict (dict): Critical infrastructure systems and their subsystems.
        hazard_types (list): List of hazard types (e.g., ['eq', 'flood', 'landslide']).
        eq_data (dict): Earthquake hazard data.
        flood_data (dict): Flood hazard data.
        landslide_rf_data (dict): Rainfall-induced landslide hazard data.
        vuln_settings (dict): How to read the vuln curves and max damages
        overwrite (bool): Whether to overwrite existing output files.

    Returns:
        None: Outputs a Parquet file per subsystem with estimated damages 
        per vulnerability curve and hazard return period at the national level.
    """
    if country_code == '-99':
        print('Please check country or file, ISO3 equals -99')
    else:
        # load country geometry file and create geometry to clip
        #ne_countries = gpd.read_file(data_path / "natural_earth" / "ne_10m_admin_0_countries.shp") #https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/
        #bbox = ne_countries.loc[ne_countries['ISO_A3']==country_code].geometry.envelope.values[0].bounds
        #country_border_geometries = ne_countries.loc[ne_countries['ISO_A3']==country_code].geometry
        data_path = pathway_dict['data_path']
        gadm_countries = gpd.read_parquet(data_path / "gadm" / "gadm_410_simplified_admin0_income") 
        bbox = gadm_countries.loc[gadm_countries['GID_0']==country_code].geometry.envelope.values[0].bounds
        country_border_geometries = gadm_countries.loc[gadm_countries['GID_0']==country_code].geometry

        # if running Fathom v2, take the v3 data for missing countries
        missing_v2_lst = ['BES', 'CUW', 'BVT', 'BMU', 'CCK', 'CXR', 'SGS', 'IMN', 'IOT', 'MNP', 'NFK', 'NIU', 'NRU', 'PCN', 'ATF', 'GUM', 'TKL', 'GRL', 'GBR', 'TJK' , 'SJM']        
        if any(h in hazard_types for h in ['fluvial', 'pluvial']) and country_code in missing_v2_lst and flood_data['version'] == 'Fathom_v2':
            print(f'Missing Fathom V2 data for this {country_code}. Use V3 instead.')
            flood_data = {'version': 'Fathom_v3', #'Fathom_v2' = own Fathom data or 'Fathom_v3' = shared by WB
                            'year': '2020', #None (for Fathom_v2), '2020', '2030', '2050', '2080'
                            'scenario': None}  #None (for Fathom_v2 and v3), 'SSP1_2.6', 'SSP2_4.5', 'SSP3_7.0', 'SSP5_8.5'
            pathway_dict['fluvial'] = base_path / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'flooding' / 'worldbank_fathom' # fathom_v3 flood data
            pathway_dict['pluvial'] = base_path / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'flooding' / 'worldbank_fathom' # fathom_v3 flood data

        for ci_system in cis_dict: 
            for sub_system in cis_dict[ci_system]:
                infra_type_lst = cis_dict[ci_system][sub_system]
                asset_loc = pathway_dict['output_path'] / 'extracts' / country_code / sub_system
                if asset_loc.exists():
                    assets = gpd.read_parquet(asset_loc)
                    assets = gpd.GeoDataFrame(assets).set_crs(4326).to_crs(3857) # convert assets to epsg3857 (system in meters)
                    print(f'{country_code} OSM data has already been extracted for {sub_system} and has now been loaded')
                else:
                    assets = osm_extract.osm_extraction(country_code, sub_system, country_border_geometries, asset_loc)
                    assets = assets.to_crs(3857) # convert assets to epsg3857 (system in meters)
     
                for hazard_type in hazard_types:
                    #continue with risk assessment if folder does not exist yet or overwrite statement is True
                    if not (pathway_dict['output_path'] / 'damage' / country_code / hazard_type).exists() or overwrite == True:
                        # read hazard data
                        hazard_data_path = pathway_dict[hazard_type]
                        hazard_data_list = damage_functions.read_hazard_data(hazard_data_path,data_path,hazard_type,country_code,flood_data,eq_data)
                        if hazard_type in ['pluvial','fluvial','coastal','windstorm','landslide_eq','landslide_rf', 'windstorm']: hazard_data_list = [file for file in hazard_data_list if file.suffix == '.tif'] #put this code in read hazard data
                        
                        # # #TEMP CODE
                        # if country_code in ['NPL'] and hazard_type in ['fluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom'] or country_code in ['MYS'] and hazard_type in ['fluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare'] or country_code in ['ERI'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare'] or country_code in ['JOR'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail'] or country_code in ['CUB'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare'] or country_code in ['PRY'] and hazard_type in ['fluvial'] and sub_system in ['power'] or country_code in ['ARE'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail'] or country_code in ['ESP'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom'] or country_code in ['BGD'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail'] or country_code in ['SRB'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare','education'] or country_code in ['JPN'] and hazard_type in ['coastal'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water'] or country_code in ['IND'] and hazard_type in ['coastal'] and sub_system in ['power', 'road', 'air', 'rail'] or country_code in ['DNK'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply'] or country_code in ['MEX'] and hazard_type in ['coastal'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply'] or country_code in ['ATF'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water']:
                        #     # or country_code in ['IND'] and hazard_type in ['fluvial'] and sub_system in ['power'] or country_code in ['PAK'] and hazard_type in ['pluvial'] and sub_system in ['water_supply']:
                        #     #{"P_1in5.tif", "P_1in10.tif", "P_1in20.tif", "P_1in50.tif", "P_1in75.tif", "P_1in100.tif", "P_1in200.tif", "P_1in250.tif","P_1in500.tif", "P_1in1000.tif"}
                        #     #{"1in5.tif", "1in10.tif", "1in20.tif", "1in50.tif", "1in100.tif", "1in200.tif", "1in500.tif", "1in1000.tif"} #v3
                        #     if country_code in ['NPL'] and hazard_type in ['fluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom']:
                        #         wanted = {} # Define the file names you want to keept
                        #     elif country_code in ['MYS'] and hazard_type in ['fluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare']:
                        #         wanted = {} # Define the file names you want to keept
                        #     elif country_code in ['ERI'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare']:
                        #         wanted = {} # Define the file names you want to keep
                        #     elif country_code in ['JOR'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail']:
                        #         wanted = {} # Define the file names you want to keep
                        #     elif country_code in ['CUB'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare']:
                        #         wanted = {} # Define the file names you want to keep
                        #     elif country_code in ['PRY'] and hazard_type in ['fluvial'] and sub_system in ['power']:
                        #         wanted = {"1in10.tif"} # Define the file names you want to keep
                        #     elif country_code in ['GUY'] and hazard_type in ['fluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid',]:
                        #         wanted = {} # Define the file names you want to keep
                        #     elif country_code in ['GUY'] and hazard_type in ['fluvial'] and sub_system in ['waste_water']:
                        #         wanted = {"1in5.tif", "1in10.tif", "1in1000.tif"} # Define the file names you want to keep
                        #     elif country_code in ['SSD'] and hazard_type in ['fluvial'] and sub_system in ['power']:
                        #         wanted = {"1in5.tif", "1in10.tif", "1in20.tif", "1in50.tif", "1in100.tif", "1in200.tif", "1in1000.tif"}  # Define the file names you want to keep
                        #     elif country_code in ['ARE'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['ARE'] and hazard_type in ['pluvial'] and sub_system in ['rail']:
                        #         wanted = {"1in5.tif", "1in10.tif", "1in50.tif", "1in1000.tif"}  # Define the file names you want to keep
                        #     elif country_code in ['ESP'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['ESP'] and hazard_type in ['pluvial'] and sub_system in ['telecom']:
                        #         wanted = {"1in5.tif"}  # Define the file names you want to keep
                        #     elif country_code in ['UZB'] and hazard_type in ['fluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['UZB'] and hazard_type in ['fluvial'] and sub_system in ['education']:
                        #         wanted = {"1in5.tif", "1in10.tif", "1in20.tif", "1in50.tif", "1in100.tif", "1in200.tif"}  # Define the file names you want to keep
                        #     elif country_code in ['DEU'] and hazard_type in ['pluvial'] and sub_system in ['power']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['DEU'] and hazard_type in ['pluvial'] and sub_system in ['road']:
                        #         wanted = {"1in10.tif"}  # Define the file names you want to keep
                        #     elif country_code in ['BGD'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['SRB'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['SRB'] and hazard_type in ['pluvial'] and sub_system in ['education']:
                        #         wanted = {"1in5.tif", "1in10.tif", "1in20.tif", "1in50.tif", "1in200.tif", "1in1000.tif"} #v3
                        #     elif country_code in ['JPN'] and hazard_type in ['coastal'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['JPN'] and hazard_type in ['coastal'] and sub_system in ['waste_water']:
                        #         wanted = {"1in5.tif", "1in10.tif", "1in20.tif", "1in50.tif", "1in1000.tif"} #v3
                        #     elif country_code in ['IND'] and hazard_type in ['coastal'] and sub_system in ['power', 'road', 'air']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['IND'] and hazard_type in ['coastal'] and sub_system in ['rail']:
                        #         wanted = {"1in5.tif", "1in10.tif", "1in1000.tif"} #v3
                        #     elif country_code in ['DNK'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['DNK'] and hazard_type in ['pluvial'] and sub_system in ['water_supply']:
                        #         wanted = {"1in5.tif", "1in10.tif", "1in20.tif", "1in50.tif", "1in100.tif", "1in200.tif", "1in500.tif", "1in1000.tif"} #v3
                        #     elif country_code in ['MEX'] and hazard_type in ['coastal'] and sub_system in ['power', 'road', 'air', 'rail']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['MEX'] and hazard_type in ['coastal'] and sub_system in ['water_supply']:
                        #         wanted = {"1in5.tif", "1in10.tif", "1in20.tif", "1in50.tif", "1in100.tif", "1in200.tif", "1in500.tif", "1in1000.tif"} #v3
                        #     elif country_code in ['ATF'] and hazard_type in ['pluvial'] and sub_system in ['power', 'road', 'air', 'rail', 'water_supply', 'waste_solid']:
                        #         wanted = {}  # Define the file names you want to keep
                        #     elif country_code in ['ATF'] and hazard_type in ['pluvial'] and sub_system in ['waste_water']:
                        #         wanted = {"1in5.tif", "1in10.tif", "1in20.tif", "1in50.tif", "1in1000.tif"} #v3
                        #     hazard_data_list = [f for f in hazard_data_list if f.name in wanted] # Filter the list

                        collect_output = {}
                        for single_footprint in hazard_data_list:                      
                            hazard_name = single_footprint.parts[-1].split('.')[0]
                            
                            # load hazard map
                            if hazard_type in ['pluvial','fluvial','coastal']:
                                if flood_data['version'] == 'Fathom_v2':
                                    if hazard_type in ['pluvial','fluvial']:
                                        if country_code in ['IND', 'KIR'] and hazard_type == 'pluvial':
                                            print('load hazard map using a gridded approach')
                                            hazard_map = damage_functions.read_flood_map_fathomv2_gridded(single_footprint, bbox, country_border_geometries)
                                        else:
                                            hazard_map = damage_functions.read_flood_map_fathomv2(single_footprint, bbox)
                                    elif hazard_type in ['coastal']:
                                        hazard_map = damage_functions.read_flood_map_aqueduct(single_footprint, bbox)
                                elif flood_data['version'] == 'Fathom_v3':
                                    if country_code not in ['GRL', 'ECU', 'SOM', 'TCD', 'NGA', 'NER', 'SAU', 'THA', 'NOR', 'NGA', 'MEX', 'ATF', 'NER', 'IND', 'SSD', 'SWE', 'JPN', 'ETH', 'FRA', 'PRY', 'IRN', 'CAF', 'ZAF', 'ETH', 'ZMB', 'LBY', 'ESP', 'NZL', 'PER', 'PNG', 'PHL', 'COL', 'PRT', 'PYF', 'SHN', 'KIR', 'FSM', 'FJI', 'UMI', 'KAZ', 'KGZ', 'UZB', 'AFG', 'TKM']: #['DZA', 'KAZ', 'ARG', 'IND', 'AUS', 'BRA', 'USA', 'CHN', 'CAN', 'RUS'] #10 largest countries
                                        hazard_map = damage_functions.read_flood_map_fathomv3(single_footprint,bbox)
                                    else:
                                        hazard_map = damage_functions.read_flood_map_fathomv3_gridded(single_footprint, bbox, country_border_geometries)
                                hazard_map = damage_functions.overlay_shapely(hazard_map, country_border_geometries)
                            elif hazard_type in ['windstorm']:
                                hazard_map = damage_functions.read_windstorm_map(single_footprint,bbox)
                                hazard_map = damage_functions.overlay_shapely(hazard_map, country_border_geometries)
                            elif hazard_type == 'earthquake': 
                                if eq_data == 'GAR': 
                                    hazard_map = damage_functions.read_gar_earthquake_map(single_footprint, bbox) #GAR
                                elif eq_data == 'GIRI': 
                                    hazard_map = damage_functions.read_giri_earthquake_map(single_footprint, bbox) #GIRI
                                elif eq_data == 'GEM':
                                    hazard_map = damage_functions.read_earthquake_map_csv(single_footprint, bbox) #GEM
                                hazard_map = damage_functions.overlay_shapely(hazard_map, country_border_geometries)
                                if not hazard_map.empty:
                                    liquefaction_map_path = liquefaction_data_path / 'liquefaction_v1_deg.tif'
                                    cond_map = damage_functions.read_liquefaction_map(liquefaction_map_path, bbox) 
                                    hazard_map = damage_functions.overlay_dataframes(hazard_map,cond_map) #get hazard polygons that overlay with cond_map
                                    if not hazard_map.empty: hazard_map = damage_functions.eq_liquefaction_matrix_liqres(hazard_map,cond_map) #apply liquefaction earthquake matrix and drop hazard points that are irrelevant
                            elif hazard_type == 'earthquake_update': 
                                if eq_data == 'GAR': 
                                    hazard_map = damage_functions.read_gar_earthquake_map(single_footprint, bbox) #GAR
                                elif eq_data == 'GIRI': 
                                    hazard_map = damage_functions.read_giri_earthquake_map(single_footprint, bbox) #GIRI
                                elif eq_data == 'GEM':
                                    hazard_map = damage_functions.read_earthquake_map_csv(single_footprint, bbox) #GEM
                                hazard_map = damage_functions.overlay_shapely(hazard_map, country_border_geometries)                          
                            elif hazard_type in ['landslide_eq', 'landslide_rf']:
                                if hazard_type == 'landslide_eq':
                                    if eq_data == 'GAR': 
                                        cond_map = damage_functions.read_gar_earthquake_map(single_footprint, bbox) #GAR
                                    elif eq_data == 'GIRI': 
                                        cond_map = damage_functions.read_giri_earthquake_map(single_footprint, bbox) #GIRI
                                    elif eq_data == 'GEM':
                                        cond_map = damage_functions.read_earthquake_map_csv(single_footprint, bbox) #GEM
                                    cond_map = damage_functions.overlay_shapely(cond_map, country_border_geometries)
                                    
                                    # Define the thresholds for the classes
                                    bins = [0, 0.05, 0.15, 0.25, 0.35, 0.45, float('inf')]  # Adjust the thresholds as needed
                                    labels = ['NaN', '1', '2', '3', '4', '5']
                                    
                                    # Create a new column 'classes' based on the thresholds
                                    cond_map['cond_classes'] = pd.cut(cond_map['band_data'], bins=bins, labels=labels, right=False, include_lowest=True)

                                    if country_code not in ['KIR', 'FJI', 'UMI', 'NZL', 'CHL', 'JPN', 'ESP', 'IDN', 'BRA', 'CHN', 'AUS',  'GRL', 'USA', 'CAN', 'RUS']:
                                        susc_map = damage_functions.read_susceptibility_map(susceptibility_map_eq_ls, hazard_type, bbox)
                                    else:
                                        susc_map = damage_functions.read_susceptibility_map_gridded(susceptibility_map_eq_ls, hazard_type, bbox, country_border_geometries)
                                elif hazard_type == 'landslide_rf':
                                    cond_map = damage_functions.read_rainfall_map(single_footprint,bbox)
                                    cond_map = damage_functions.overlay_shapely(cond_map, country_border_geometries)
                                    
                                    # Define the thresholds for the classes
                                    bins = [0, 0.3, 2.0, 3.7, 5.0, float('inf')]  # Adjust the thresholds as needed
                                    labels = ['1', '2', '3', '4', '5']
                                    
                                    # Create a new column 'classes' based on the thresholds
                                    cond_map['cond_classes'] = pd.cut(cond_map['band_data'], bins=bins, labels=labels, right=False, include_lowest=True)
                                    
                                    if country_code not in ['KIR', 'FJI', 'ATF', 'CHL', 'UMI' , 'MMR', 'NZL','IDN', 'BRA', 'CHN', 'AUS', 'USA', 'GRL', 'CAN', 'RUS']:
                                        susc_map = damage_functions.read_susceptibility_map(susceptibility_map_rf_ls, hazard_type, bbox)
                                    else:
                                        susc_map = damage_functions.read_susceptibility_map_gridded(susceptibility_map_rf_ls, hazard_type, bbox, country_border_geometries)
                                susc_map = damage_functions.overlay_shapely(susc_map, country_border_geometries) #overlay with exact administrative border

                            # convert hazard data to epsg 3857
                            if hazard_type in ['landslide_eq', 'landslide_rf']:
                                cond_map = gpd.GeoDataFrame(cond_map).set_crs(4326).to_crs(3857)
                                susc_map = gpd.GeoDataFrame(susc_map).set_crs(4326).to_crs(3857)
                            elif hazard_type in ['pluvial', 'fluvial', 'coastal', 'windstorm', 'earthquake'] :
                                hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                            elif hazard_type in ['earthquake_update']: 
                                if sub_system in ['water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare', 'education']:
                                    hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                                elif sub_system in ['road', 'rail']:
                                    if not hazard_map.empty:
                                        liquefaction_map_path = liquefaction_data_path / 'liquefaction_v1_deg.tif'
                                        cond_map = damage_functions.read_liquefaction_map(liquefaction_map_path, bbox) 
                                        hazard_map = damage_functions.overlay_dataframes(hazard_map,cond_map) #get hazard polygons that overlay with cond_map
                                        if not hazard_map.empty: hazard_map = damage_functions.eq_liquefaction_matrix_liqres(hazard_map,cond_map) #apply liquefaction earthquake matrix and drop hazard points that are irrelevant
                                    hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                                elif sub_system in ['power', 'air']:
                                    if not hazard_map.empty:
                                        liquefaction_map_path = liquefaction_data_path / 'liquefaction_v1_deg.tif'
                                        cond_map = damage_functions.read_liquefaction_map(liquefaction_map_path, bbox) 
                                        hazard_eqliq_map = damage_functions.overlay_dataframes(hazard_map,cond_map) #get hazard polygons that overlay with cond_map
                                        if not hazard_eqliq_map.empty: hazard_eqliq_map = damage_functions.eq_liquefaction_matrix_liqres(hazard_eqliq_map,cond_map) #apply liquefaction earthquake matrix and drop hazard points that are irrelevant

                                        # two hazard maps needed, one for linear infra and the other for non-linear infra within sub-system
                                        hazard_eqliq_map = gpd.GeoDataFrame(hazard_eqliq_map).set_crs(4326).to_crs(3857)
                                    else:
                                        hazard_eqliq_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                                    hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)                                        
                        
                            # Loop through unique infrastructure types within the subsystem
                            output_dfs = [] # empty list to collect asset level damages for output at the sub-system level
                            trig_rp_lst_complete = [] #empty list to collect hazard trigger return periods for landsldies
                            for infra_type in infra_type_lst: 
                                assets_infra_type = (assets[assets['asset'] == infra_type].copy().reset_index(drop=True))
                            
                                # create dicts for quicker lookup
                                geom_dict = assets_infra_type['geometry'].to_dict()
                        
                                # read vulnerability and maxdam data:
                                infra_curves,maxdams,infra_units = damage_functions.read_vul_maxdam(data_path,hazard_type, infra_type, vuln_settings[0], vuln_settings[1])
                        
                                # start analysis 
                                print(f'{country_code} runs for {infra_type} for {hazard_type} using the {hazard_name} map')
                        
                                if hazard_type in ['landslide_eq', 'landslide_rf']:
                                    if not assets_infra_type.empty:
                                        # overlay assets
                                        overlay_assets = pd.DataFrame(damage_functions.overlay_hazard_assets(susc_map,damage_functions.buffer_assets(assets_infra_type)).T,columns=['asset','hazard_point'])
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
                                                overlay_assets = damage_functions.matrix_landslide_eq_susc(overlay_cond, get_susc_data, overlay_assets, susc_point) 
                                            elif hazard_type == 'landslide_rf':
                                                overlay_assets = damage_functions.matrix_landslide_rf_susc(overlay_cond, get_susc_data, overlay_assets, susc_point)
                                        else:
                                            overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
                        
                                    #run and output damage calculations for landslides
                                    if not assets_infra_type.empty:
                                        if assets_infra_type['geometry'][0].geom_type in ['LineString', 'MultiLineString']:
                                            collect_output,damages_collection_trig_rp, trig_rp_lst = damage_functions.landslide_damage_and_overlay(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type,
                                                                                          maxdams,infra_units,geom_dict,country_code,sub_system,infra_type,collect_output)
                                            trig_rp_lst_complete.append(trig_rp_lst)
                                            output_dfs.append(damages_collection_trig_rp)                                
                                        else:
                                            collect_output,damages_collection_trig_rp, trig_rp_lst = damage_functions.landslide_damage_and_overlay(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type,
                                                                                                                                  maxdams,infra_units,geom_dict,country_code,sub_system,infra_type,collect_output)
                                            trig_rp_lst_complete.append(trig_rp_lst)
                                            output_dfs.append(damages_collection_trig_rp)

                                elif hazard_type in ['earthquake', 'earthquake_update', 'pluvial', 'fluvial','coastal','windstorm']:
                                    if not assets_infra_type.empty:
                                        # overlay assets
                                        if hazard_type == 'earthquake_update' and infra_type in ['cable', 'runway']:
                                            overlay_assets = pd.DataFrame(damage_functions.overlay_hazard_assets(hazard_eqliq_map,damage_functions.buffer_assets(assets_infra_type)).T,columns=['asset','hazard_point'])
                                        else:
                                            overlay_assets = pd.DataFrame(damage_functions.overlay_hazard_assets(hazard_map,damage_functions.buffer_assets(assets_infra_type)).T,columns=['asset','hazard_point'])
                                    else: 
                                        overlay_assets = pd.DataFrame(columns=['asset','hazard_point']) #empty dataframe
                            
                                    # convert dataframe to numpy array
                                    if hazard_type == 'earthquake_update' and infra_type in ['cable', 'runway']:
                                        hazard_numpified = hazard_eqliq_map.to_numpy()
                                    else:
                                        hazard_numpified = hazard_map.to_numpy()
                        
                                    collect_asset_damages_per_curve = [] # for output at asset level
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
                                                damage_asset, overlay_asset = damage_functions.get_damage_per_asset_and_overlay(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam,unit_maxdam) #for output at asset level
                                                collect_inb.append(damage_asset) #for excel output
                                                collect_damage_asset[asset[0]] = damage_asset # for output at asset level
                                                collect_overlay_asset[asset[0]] = overlay_asset # for exposure output at asset level
                                        
                                            collect_output[country_code, hazard_name, sub_system, infra_type, infra_curve[0], ((maxdams[maxdams == maxdam]).index)[0]] = np.sum(collect_inb) # dictionary to store results for various combinations of hazard maps, infrastructure curves, and maximum damage values.
                                            asset_damage = pd.Series(collect_damage_asset)  # for output at asset level
                                            asset_damage.columns = [infra_curve[0]]  # for output at asset level
                                            collect_asset_damages_per_curve.append(asset_damage)  # for output at asset level
                                            asset_exposure = pd.Series(collect_overlay_asset)  # for exposure output at asset level
                                            asset_exposure.columns = 'overlay'  # for exposure output at asset level
                                        curve_ids_list.append(infra_curve[0])  # for output at asset level
                        
                                    #if collect_asset_damages_per_curve[0].empty == False: #collect_asset_damages_per_curve.empty == False
                                    if any(not series.empty for series in collect_asset_damages_per_curve):
                                        asset_damages_per_curve = pd.concat(collect_asset_damages_per_curve,axis=1)
                                        asset_damages_per_curve.columns = curve_ids_list
                                        asset_damages_per_curve = asset_damages_per_curve.merge(asset_exposure.rename('overlay'), left_index=True, right_index=True) #merge exposure with damages dataframe
                                        asset_damages_per_curve = asset_damages_per_curve.join(assets_infra_type[['osm_id']], how='left')
                                        output_dfs.append(asset_damages_per_curve)

                                #break #delete after testing, otherwise damage will only be assessed for first hazard map

                            if hazard_type in ['landslide_eq', 'landslide_rf']:
                                del susc_map
                                del cond_map
                            else:
                                del hazard_map
                        
                            if hazard_type in ['landslide_eq', 'landslide_rf']:
                                # Specify the outer key to filter by
                                trig_rp_lst_complete = [item for sublist in trig_rp_lst_complete for item in sublist]
                                trig_rp_lst_complete = list(set(trig_rp_lst_complete))
                                for trig_rp in trig_rp_lst_complete:
                                    # Create a filtered list of dictionaries based on the outer key
                                    output_dfs_filtered = [d[trig_rp] for d in output_dfs if trig_rp in d]
                                        
                                    if output_dfs_filtered: #if list is not empty
                                        keys_lst = list(output_dfs_filtered[0].keys())
                                        concatenated_dataframes = {key: [] for key in keys_lst}
                                    
                                        # Loop through each dictionary in the list
                                        for d in output_dfs_filtered:
                                            for rp, df in d.items():
                                                if rp not in concatenated_dataframes: concatenated_dataframes[rp] = []  # Initialize an empty list for this return period
                                                # Append each sub-DataFrame to the corresponding return period list
                                                concatenated_dataframes[rp].append(df)
                                    
                                        # Concatenate DataFrames per return period and output
                                        for rp, dfs in concatenated_dataframes.items():
                                            concatenated_dataframes[rp] = pd.concat(dfs, ignore_index=True)
                                            save_path = pathway_dict['output_path'] / 'damage' / country_code / hazard_type / f'{country_code}_{hazard_type}_ls{rp}_trig{trig_rp}_{sub_system}.parquet'
                                            (save_path.parent).mkdir(parents=True, exist_ok=True)
                                            concatenated_dataframes[rp].to_parquet(save_path, index=False)
                                damage_functions.create_damage_csv_without_exposure(collect_output, hazard_type, pathway_dict, country_code, sub_system) 
                            else:
                                #create parquet with damages at sub_system level
                                if output_dfs:
                                    damages_df = pd.concat(output_dfs, ignore_index=True)
                                    save_path = pathway_dict['output_path'] / 'damage' / country_code / hazard_type / f'{country_code}_{hazard_type}_{hazard_name}_{sub_system}.parquet'
                                    (save_path.parent).mkdir(parents=True, exist_ok=True)
                                    damages_df.to_parquet(save_path, index=False)

                                    del damages_df

                                #create csv file with damages at national level
                                damage_functions.create_damage_csv_without_exposure(collect_output, hazard_type, pathway_dict, country_code, sub_system) 

    print('Run completed for {}'.format(country_code))

def damage_calculation_admin2(country_code, pathway_dict, cis_dict, hazard_types, eq_data, flood_data, landslide_rf_data, vuln_settings, overwrite=False):
    """
    Seperate function to calculate hazard-related damages to critical infrastructure at the subsystem level at the admin 2 level per country for hazard data
    with high resolution. Processing is done in parallel. 

    Args:
        country_code (str): ISO3 country code.
        pathway_dict (dict): Dictionary with paths to input and output directories.
        cis_dict (dict): Critical infrastructure systems and their subsystems.
        hazard_types (list): List of hazard types (e.g., ['eq', 'flood', 'landslide']).
        eq_data (dict): Earthquake hazard data.
        flood_data (dict): Flood hazard data.
        landslide_rf_data (dict): Rainfall-induced landslide hazard data.
        vuln_settings (dict): How to read the vuln curves and max damages
        overwrite (bool): Whether to overwrite existing output files.

    Returns:
        None: Outputs a Parquet file per subsystem with estimated damages 
        per vulnerability curve and hazard return period per admin 2 level.
    """
    if country_code == '-99':
        print('Please check country or file, ISO3 equals -99')
    else:
        # load country geometry file and create geometry to clip
        data_path = pathway_dict['data_path']
        gadm_countries = gpd.read_parquet(data_path / "gadm" / "gadm_410_admin2_complete") 
        gadm_countries = gadm_countries.to_crs(epsg=3857) # Reproject to a projected CRS, such as EPSG:3857 (Web Mercator, units in meters)
        gadm_countries['surface_area'] = gadm_countries.geometry.area # calculate surface area
        gadm_countries = gadm_countries.sort_values(by='surface_area')
        gadm_countries = gadm_countries.to_crs(epsg=4326)
        admin2_lst = gadm_countries.loc[gadm_countries['GID_0']==country_code]['GID_2'].tolist()
        if not len(admin2_lst) == len(set(admin2_lst)): print('Not all admin2 names are unique')
        print(f'run {country_code} for the following admin2s: {admin2_lst}')

        for ci_system in cis_dict: 
            for sub_system in cis_dict[ci_system]:
                
                # #TEMP CODE
                # if 'fluvial' in hazard_types:
                #     if sub_system == 'power':
                #         if country_code == 'AUS': admin2_lst = ['AUS.11.88_1', 'AUS.11.13_1', 'AUS.7.52_1', 'AUS.5.34_1', 'AUS.7.71_1', 'AUS.7.4_1', 'AUS.11.47_1', 'AUS.7.34_1', 'AUS.11.121_1', 'AUS.7.14_1', 'AUS.7.63_1', 'AUS.7.7_1', 'AUS.7.16_1', 'AUS.7.42_1', 'AUS.5.31_1', 'AUS.7.5_1', 'AUS.7.17_1', 'AUS.7.54_1', 'AUS.7.9_1', 'AUS.7.20_1', 'AUS.7.22_1', 'AUS.11.3_1', 'AUS.11.137_1', 'AUS.11.75_1', 'AUS.5.130_1', 'AUS.11.44_1', 'AUS.11.61_1', 'AUS.8.4_1', 'AUS.11.40_1', 'AUS.8.31_1', 'AUS.11.55_1', 'AUS.11.77_1', 'AUS.6.15_1', 'AUS.11.95_1', 'AUS.6.12_1', 'AUS.11.133_1', 'AUS.11.71_1', 'AUS.6.10_1', 'AUS.6.4_1', 'AUS.6.2_1', 'AUS.11.46_1', 'AUS.8.61_1']
                #         # elif country_code == 'USA': admin2_lst = ['USA.2.26_1', 'USA.38.13_1', 'USA.2.11_1', 'USA.3.10_1', 'USA.13.37_1', 'USA.45.23_1', 'USA.29.10_1', 'USA.2.16_1', 'USA.2.25_1', 'USA.2.17_1', 'USA.5.36_1', 'USA.2.6_1', 'USA.23.44_1', 'USA.29.17_1', 'USA.50.34_1', 'USA.2.15_1', 'USA.2.18_1', 'USA.29.13_1', 'USA.51.15_1', 'USA.2.21_1', 'USA.3.9_1', 'USA.2.14_1', 'USA.13.25_1', 'USA.24.70_1', 'USA.3.8_1', 'USA.2.4_1', 'USA.3.3_1', 'USA.2.13_1', 'USA.2.8_1', 'USA.3.1_1', 'USA.29.3_1', 'USA.2.24_1', 'USA.3.11_1', 'USA.2.1_1', 'USA.51.7_1', 'USA.20.2_1', 'USA.2.2_1', 'USA.2.23_1', 'USA.23.45_1', 'USA.29.5_1', 'USA.23.47_1', 'USA.51.19_1', 'USA.45.19_1', 'USA.5.14_1', 'USA.38.23_1', 'USA.51.4_1', 'USA.2.7_1', 'USA.29.8_1', 'USA.2.22_1']
                #         elif country_code == 'KAZ': admin2_lst = ['KAZ.13.10_1', 'KAZ.5.1_1', 'KAZ.9.9_1', 'KAZ.9.6_1', 'KAZ.4.5_1', 'KAZ.7.8_1', 'KAZ.4.2_1', 'KAZ.13.3_1', 'KAZ.10.3_1', 'KAZ.12.3_1', 'KAZ.7.10_1', 'KAZ.8.8_1', 'KAZ.9.8_1', 'KAZ.13.8_1', 'KAZ.13.2_1', 'KAZ.4.3_1', 'KAZ.8.2_1', 'KAZ.9.7_1', 'KAZ.8.1_1', 'KAZ.3.1_1', 'KAZ.12.8_1', 'KAZ.5.14_1', 'KAZ.3.11_1', 'KAZ.3.13_1', 'KAZ.2.3_1', 'KAZ.9.4_1', 'KAZ.1.13_1', 'KAZ.2.6_1', 'KAZ.3.3_1', 'KAZ.10.14_1', 'KAZ.13.11_1', 'KAZ.5.16_1', 'KAZ.3.4_1', 'KAZ.12.2_1', 'KAZ.5.3_1', 'KAZ.12.11_1', 'KAZ.10.5_1', 'KAZ.8.7_1', 'KAZ.10.12_1', 'KAZ.7.6_1', 'KAZ.9.2_1', 'KAZ.8.4_1', 'KAZ.7.3_1', 'KAZ.1.12_1', 'KAZ.7.5_1', 'KAZ.8.9_1', 'KAZ.3.6_1', 'KAZ.6.2_1', 'KAZ.2.14_1', 'KAZ.9.3_1', 'KAZ.5.12_1', 'KAZ.2.12_1', 'KAZ.11.5_1', 'KAZ.5.13_1', 'KAZ.14.3_1', 'KAZ.5.8_1', 'KAZ.10.15_1', 'KAZ.14.4_1', 'KAZ.6.3_1', 'KAZ.6.1_1', 'KAZ.1.15_1', 'KAZ.1.17_1', 'KAZ.2.15_1', 'KAZ.2.11_1', 'KAZ.13.4_1', 'KAZ.13.12_1', 'KAZ.12.10_1', 'KAZ.12.1_1', 'KAZ.1.4_1', 'KAZ.11.1_1', 'KAZ.14.2_1', 'KAZ.7.12_1', 'KAZ.7.1_1', 'KAZ.4.6_1', 'KAZ.1.8_1', 'KAZ.10.9_1', 'KAZ.2.7_1', 'KAZ.5.11_1', 'KAZ.1.6_1', 'KAZ.9.5_1', 'KAZ.11.3_1', 'KAZ.7.13_1', 'KAZ.13.7_1', 'KAZ.14.1_1', 'KAZ.10.10_1', 'KAZ.8.5_1', 'KAZ.13.9_1', 'KAZ.2.13_1', 'KAZ.1.14_1', 'KAZ.2.9_1', 'KAZ.5.2_1', 'KAZ.1.2_1', 'KAZ.12.7_1', 'KAZ.5.9_1', 'KAZ.1.11_1', 'KAZ.3.9_1', 'KAZ.5.6_1', 'KAZ.3.12_1', 'KAZ.14.6_1', 'KAZ.12.4_1', 'KAZ.11.2_1', 'KAZ.2.17_1', 'KAZ.10.13_1', 'KAZ.10.4_1', 'KAZ.4.7_1', 'KAZ.4.1_1', 'KAZ.1.16_1', 'KAZ.7.11_1', 'KAZ.2.10_1', 'KAZ.14.12_1', 'KAZ.14.5_1', 'KAZ.11.7_1', 'KAZ.5.10_1', 'KAZ.6.5_1', 'KAZ.9.1_1', 'KAZ.3.2_1', 'KAZ.2.1_1', 'KAZ.10.16_1', 'KAZ.5.4_1', 'KAZ.5.7_1', 'KAZ.14.10_1', 'KAZ.6.4_1', 'KAZ.3.10_1', 'KAZ.1.3_1', 'KAZ.8.11_1', 'KAZ.10.17_1', 'KAZ.12.12_1', 'KAZ.3.8_1', 'KAZ.8.3_1', 'KAZ.14.8_1', 'KAZ.10.8_1', 'KAZ.1.5_1', 'KAZ.2.2_1', 'KAZ.12.6_1', 'KAZ.3.5_1', 'KAZ.10.11_1', 'KAZ.10.2_1', 'KAZ.10.1_1', 'KAZ.14.7_1', 'KAZ.11.4_1', 'KAZ.1.1_1', 'KAZ.5.5_1', 'KAZ.11.6_1', 'KAZ.8.10_1', 'KAZ.1.7_1', 'KAZ.10.7_1', 'KAZ.2.8_1', 'KAZ.3.7_1', 'KAZ.7.2_1', 'KAZ.12.13_1', 'KAZ.2.4_1', 'KAZ.1.9_1', 'KAZ.12.14_1', 'KAZ.2.16_1', 'KAZ.8.6_1', 'KAZ.10.6_1', 'KAZ.11.8_1', 'KAZ.12.5_1', 'KAZ.7.4_1', 'KAZ.4.8_1', 'KAZ.8.12_1', 'KAZ.4.4_1', 'KAZ.5.17_1', 'KAZ.13.1_1', 'KAZ.2.5_1', 'KAZ.12.9_1', 'KAZ.13.5_1', 'KAZ.7.9_1', 'KAZ.1.10_1']
                #         elif country_code == 'CAN': admin2_lst = ['CAN.8.1_1', 'CAN.6.2_1', 'CAN.8.3_1', 'CAN.6.1_1', 'CAN.13.1_1', 'CAN.8.2_1', 'CAN.11.84_1']
                #     elif sub_system == 'road':
                #         # if country_code == 'USA': admin2_lst = ['USA.2.16_1', 'USA.2.7_1', 'USA.38.13_1', 'USA.45.19_1', 'USA.2.21_1', 'USA.2.19_1', 'USA.24.70_1', 'USA.2.4_1', 'USA.2.23_1', 'USA.2.1_1', 'USA.29.8_1', 'USA.2.6_1', 'USA.23.45_1', 'USA.2.15_1', 'USA.20.2_1', 'USA.3.3_1', 'USA.51.19_1', 'USA.38.19_1', 'USA.45.23_1', 'USA.38.23_1', 'USA.51.15_1', 'USA.2.17_1', 'USA.2.22_1', 'USA.29.13_1', 'USA.3.10_1', 'USA.51.4_1', 'USA.2.26_1', 'USA.2.13_1', 'USA.2.14_1', 'USA.23.44_1', 'USA.5.36_1', 'USA.29.10_1', 'USA.29.3_1', 'USA.3.8_1', 'USA.29.17_1', 'USA.50.34_1', 'USA.29.5_1', 'USA.3.1_1', 'USA.2.18_1', 'USA.2.24_1', 'USA.2.11_1', 'USA.23.47_1', 'USA.3.9_1', 'USA.13.25_1', 'USA.3.11_1', 'USA.13.37_1', 'USA.2.2_1', 'USA.2.25_1', 'USA.5.14_1', 'USA.2.8_1', 'USA.51.7_1']
                #         if country_code == 'KAZ': admin2_lst = ['KAZ.6.2_1', 'KAZ.10.3_1', 'KAZ.10.11_1', 'KAZ.12.5_1', 'KAZ.10.8_1', 'KAZ.13.4_1', 'KAZ.5.4_1', 'KAZ.1.4_1', 'KAZ.14.8_1', 'KAZ.9.8_1', 'KAZ.1.1_1', 'KAZ.10.7_1', 'KAZ.8.11_1', 'KAZ.3.7_1', 'KAZ.1.8_1', 'KAZ.2.5_1', 'KAZ.7.6_1', 'KAZ.11.3_1', 'KAZ.14.7_1', 'KAZ.8.6_1', 'KAZ.2.2_1', 'KAZ.9.7_1', 'KAZ.7.2_1', 'KAZ.12.14_1', 'KAZ.11.8_1', 'KAZ.10.1_1', 'KAZ.13.7_1', 'KAZ.13.5_1', 'KAZ.1.2_1', 'KAZ.1.9_1', 'KAZ.9.9_1', 'KAZ.2.16_1', 'KAZ.12.13_1', 'KAZ.14.10_1', 'KAZ.7.9_1', 'KAZ.7.4_1', 'KAZ.3.6_1', 'KAZ.11.7_1', 'KAZ.7.5_1', 'KAZ.1.3_1', 'KAZ.2.12_1', 'KAZ.5.10_1', 'KAZ.8.12_1', 'KAZ.3.11_1', 'KAZ.1.12_1', 'KAZ.2.17_1', 'KAZ.8.10_1', 'KAZ.14.4_1', 'KAZ.8.3_1', 'KAZ.3.10_1', 'KAZ.13.10_1', 'KAZ.3.4_1', 'KAZ.4.2_1', 'KAZ.2.3_1', 'KAZ.3.8_1', 'KAZ.2.8_1', 'KAZ.9.1_1', 'KAZ.1.15_1', 'KAZ.10.2_1', 'KAZ.12.3_1', 'KAZ.10.16_1', 'KAZ.9.2_1', 'KAZ.8.2_1', 'KAZ.12.1_1', 'KAZ.7.10_1', 'KAZ.5.5_1', 'KAZ.13.8_1', 'KAZ.4.4_1', 'KAZ.4.6_1', 'KAZ.4.8_1', 'KAZ.8.1_1', 'KAZ.6.3_1', 'KAZ.3.13_1', 'KAZ.1.5_1', 'KAZ.12.8_1', 'KAZ.5.6_1', 'KAZ.5.1_1', 'KAZ.12.6_1', 'KAZ.11.2_1', 'KAZ.1.13_1', 'KAZ.13.3_1', 'KAZ.11.1_1', 'KAZ.5.3_1', 'KAZ.5.11_1', 'KAZ.4.3_1', 'KAZ.8.8_1', 'KAZ.14.1_1', 'KAZ.11.4_1', 'KAZ.3.1_1', 'KAZ.11.6_1', 'KAZ.8.7_1', 'KAZ.9.5_1', 'KAZ.13.6_1', 'KAZ.10.4_1', 'KAZ.13.11_1', 'KAZ.10.14_1', 'KAZ.1.14_1', 'KAZ.1.7_1', 'KAZ.14.6_1', 'KAZ.2.14_1', 'KAZ.5.2_1', 'KAZ.7.3_1', 'KAZ.13.2_1', 'KAZ.3.12_1', 'KAZ.10.5_1', 'KAZ.10.12_1', 'KAZ.10.15_1', 'KAZ.13.1_1', 'KAZ.14.5_1', 'KAZ.5.13_1', 'KAZ.8.9_1', 'KAZ.2.7_1', 'KAZ.5.9_1', 'KAZ.9.3_1', 'KAZ.5.15_1', 'KAZ.11.5_1', 'KAZ.6.4_1', 'KAZ.2.15_1', 'KAZ.12.4_1', 'KAZ.14.3_1', 'KAZ.7.7_1', 'KAZ.8.5_1', 'KAZ.13.12_1', 'KAZ.6.1_1', 'KAZ.1.17_1', 'KAZ.5.8_1', 'KAZ.3.5_1', 'KAZ.7.1_1', 'KAZ.2.11_1', 'KAZ.4.1_1', 'KAZ.10.9_1', 'KAZ.2.4_1', 'KAZ.7.12_1', 'KAZ.2.6_1', 'KAZ.4.5_1', 'KAZ.14.2_1', 'KAZ.10.6_1', 'KAZ.7.13_1', 'KAZ.13.9_1', 'KAZ.2.1_1', 'KAZ.10.10_1', 'KAZ.1.6_1', 'KAZ.5.14_1', 'KAZ.14.12_1', 'KAZ.3.3_1', 'KAZ.2.9_1', 'KAZ.1.11_1', 'KAZ.12.12_1', 'KAZ.3.2_1', 'KAZ.12.9_1', 'KAZ.12.7_1', 'KAZ.2.13_1', 'KAZ.9.6_1', 'KAZ.7.8_1', 'KAZ.12.10_1', 'KAZ.5.16_1', 'KAZ.4.7_1', 'KAZ.10.13_1', 'KAZ.9.4_1', 'KAZ.3.9_1', 'KAZ.7.11_1', 'KAZ.6.5_1', 'KAZ.8.4_1', 'KAZ.2.10_1', 'KAZ.12.11_1', 'KAZ.5.17_1', 'KAZ.1.16_1', 'KAZ.5.7_1', 'KAZ.10.17_1', 'KAZ.1.10_1', 'KAZ.12.2_1', 'KAZ.5.12_1']                 
                #         elif country_code == 'CAN': admin2_lst = ['CAN.8.3_1', 'CAN.11.84_1', 'CAN.8.2_1', 'CAN.6.2_1', 'CAN.13.1_1', 'CAN.8.1_1', 'CAN.6.1_1']             
                #     elif sub_system == 'air':
                #         # if country_code == 'USA': admin2_lst = ['USA.29.3_1', 'USA.38.23_1', 'USA.2.23_1', 'USA.5.14_1', 'USA.2.8_1', 'USA.3.3_1', 'USA.2.24_1', 'USA.2.18_1', 'USA.3.1_1', 'USA.3.8_1', 'USA.51.19_1', 'USA.2.14_1', 'USA.29.10_1', 'USA.2.15_1', 'USA.2.21_1', 'USA.29.8_1', 'USA.51.15_1', 'USA.2.17_1', 'USA.29.13_1', 'USA.2.25_1', 'USA.38.19_1', 'USA.2.26_1', 'USA.5.36_1', 'USA.3.9_1', 'USA.13.37_1', 'USA.2.4_1', 'USA.2.16_1', 'USA.2.1_1', 'USA.3.11_1', 'USA.2.6_1', 'USA.51.7_1', 'USA.29.5_1', 'USA.20.2_1', 'USA.13.25_1', 'USA.2.13_1', 'USA.50.34_1', 'USA.45.23_1', 'USA.24.70_1', 'USA.51.4_1', 'USA.45.19_1', 'USA.2.22_1', 'USA.2.11_1', 'USA.29.17_1', 'USA.23.44_1', 'USA.2.2_1', 'USA.23.47_1', 'USA.3.10_1', 'USA.2.19_1', 'USA.2.7_1', 'USA.23.45_1']
                #         if country_code == 'KAZ': admin2_lst = ['KAZ.4.7_1', 'KAZ.1.9_1', 'KAZ.1.15_1', 'KAZ.6.3_1', 'KAZ.12.10_1', 'KAZ.5.14_1', 'KAZ.8.9_1', 'KAZ.8.3_1', 'KAZ.10.3_1', 'KAZ.6.2_1', 'KAZ.7.11_1', 'KAZ.13.10_1', 'KAZ.3.6_1', 'KAZ.3.2_1', 'KAZ.6.4_1', 'KAZ.14.1_1', 'KAZ.14.3_1', 'KAZ.9.7_1', 'KAZ.5.10_1', 'KAZ.6.5_1', 'KAZ.9.2_1', 'KAZ.12.3_1', 'KAZ.5.3_1', 'KAZ.11.6_1', 'KAZ.12.7_1', 'KAZ.3.10_1', 'KAZ.1.8_1', 'KAZ.11.8_1', 'KAZ.12.6_1', 'KAZ.5.17_1', 'KAZ.5.6_1', 'KAZ.1.14_1', 'KAZ.14.5_1', 'KAZ.2.12_1', 'KAZ.2.15_1', 'KAZ.1.2_1', 'KAZ.7.5_1', 'KAZ.14.10_1', 'KAZ.1.17_1', 'KAZ.1.6_1', 'KAZ.1.3_1', 'KAZ.5.12_1', 'KAZ.12.11_1', 'KAZ.3.12_1', 'KAZ.2.14_1', 'KAZ.10.13_1', 'KAZ.2.3_1', 'KAZ.11.2_1', 'KAZ.12.13_1', 'KAZ.9.8_1', 'KAZ.1.7_1', 'KAZ.3.9_1', 'KAZ.5.2_1', 'KAZ.3.11_1', 'KAZ.9.3_1', 'KAZ.2.4_1', 'KAZ.12.9_1', 'KAZ.5.15_1', 'KAZ.14.8_1', 'KAZ.8.4_1', 'KAZ.11.3_1', 'KAZ.10.6_1', 'KAZ.1.16_1', 'KAZ.4.8_1', 'KAZ.5.5_1', 'KAZ.2.1_1', 'KAZ.13.12_1', 'KAZ.11.1_1', 'KAZ.14.7_1', 'KAZ.8.8_1', 'KAZ.11.4_1', 'KAZ.5.16_1', 'KAZ.1.4_1', 'KAZ.9.9_1', 'KAZ.3.5_1', 'KAZ.1.5_1']                   
                #         elif country_code == 'CAN': admin2_lst = ['CAN.8.2_1', 'CAN.6.1_1', 'CAN.8.1_1', 'CAN.6.2_1', 'CAN.11.84_1', 'CAN.13.1_1', 'CAN.8.3_1']                    
                #     elif sub_system == 'rail':
                #         # if country_code == 'USA': admin2_lst = ['USA.51.19_1', 'USA.5.36_1', 'USA.20.2_1', 'USA.2.14_1', 'USA.2.24_1', 'USA.38.13_1', 'USA.29.5_1', 'USA.29.17_1', 'USA.2.16_1', 'USA.2.4_1', 'USA.2.25_1', 'USA.2.7_1', 'USA.23.47_1', 'USA.3.10_1', 'USA.45.23_1', 'USA.2.11_1', 'USA.2.23_1', 'USA.3.9_1', 'USA.2.2_1', 'USA.2.26_1', 'USA.2.8_1', 'USA.38.23_1', 'USA.5.14_1', 'USA.2.21_1', 'USA.2.13_1', 'USA.3.1_1', 'USA.3.11_1', 'USA.2.15_1', 'USA.2.6_1', 'USA.2.19_1', 'USA.29.10_1', 'USA.3.8_1', 'USA.29.3_1', 'USA.50.34_1', 'USA.3.3_1', 'USA.51.7_1', 'USA.29.13_1', 'USA.13.37_1']
                #         if country_code == 'KAZ': admin2_lst = ['KAZ.1.17_1', 'KAZ.12.7_1', 'KAZ.8.1_1', 'KAZ.2.3_1', 'KAZ.4.6_1', 'KAZ.7.3_1', 'KAZ.14.2_1', 'KAZ.11.5_1', 'KAZ.1.11_1', 'KAZ.4.3_1', 'KAZ.5.11_1', 'KAZ.3.11_1', 'KAZ.6.3_1', 'KAZ.6.2_1', 'KAZ.4.2_1', 'KAZ.1.6_1', 'KAZ.3.10_1', 'KAZ.1.14_1', 'KAZ.4.5_1', 'KAZ.14.1_1', 'KAZ.5.5_1', 'KAZ.5.10_1', 'KAZ.13.10_1', 'KAZ.10.4_1', 'KAZ.9.4_1', 'KAZ.5.3_1', 'KAZ.2.13_1', 'KAZ.5.9_1', 'KAZ.9.5_1', 'KAZ.3.8_1', 'KAZ.3.9_1', 'KAZ.13.12_1', 'KAZ.7.11_1', 'KAZ.2.17_1', 'KAZ.12.4_1', 'KAZ.4.7_1', 'KAZ.1.16_1', 'KAZ.4.1_1', 'KAZ.3.12_1', 'KAZ.13.5_1', 'KAZ.3.1_1', 'KAZ.8.10_1', 'KAZ.9.1_1', 'KAZ.3.2_1', 'KAZ.11.1_1', 'KAZ.5.4_1', 'KAZ.14.12_1', 'KAZ.10.14_1', 'KAZ.14.5_1', 'KAZ.11.2_1', 'KAZ.5.13_1', 'KAZ.8.11_1', 'KAZ.2.7_1', 'KAZ.2.8_1', 'KAZ.14.10_1', 'KAZ.1.3_1', 'KAZ.11.3_1', 'KAZ.2.2_1', 'KAZ.12.12_1', 'KAZ.2.15_1', 'KAZ.10.1_1', 'KAZ.12.6_1', 'KAZ.10.7_1', 'KAZ.2.4_1', 'KAZ.8.12_1', 'KAZ.10.8_1', 'KAZ.12.13_1', 'KAZ.9.3_1', 'KAZ.1.7_1', 'KAZ.10.5_1', 'KAZ.10.17_1', 'KAZ.11.4_1', 'KAZ.4.4_1', 'KAZ.8.3_1', 'KAZ.7.1_1', 'KAZ.4.8_1', 'KAZ.13.11_1', 'KAZ.1.9_1', 'KAZ.2.5_1', 'KAZ.2.1_1', 'KAZ.11.6_1', 'KAZ.7.13_1', 'KAZ.10.15_1', 'KAZ.1.10_1', 'KAZ.12.14_1', 'KAZ.1.8_1', 'KAZ.12.11_1', 'KAZ.1.4_1', 'KAZ.7.8_1', 'KAZ.3.5_1', 'KAZ.5.16_1', 'KAZ.9.6_1', 'KAZ.9.2_1', 'KAZ.12.5_1', 'KAZ.5.17_1', 'KAZ.2.9_1', 'KAZ.2.14_1', 'KAZ.13.9_1', 'KAZ.8.8_1', 'KAZ.12.9_1', 'KAZ.1.13_1', 'KAZ.7.10_1', 'KAZ.11.8_1', 'KAZ.3.3_1', 'KAZ.13.2_1', 'KAZ.3.4_1', 'KAZ.5.2_1', 'KAZ.10.13_1', 'KAZ.8.9_1', 'KAZ.12.8_1', 'KAZ.10.16_1', 'KAZ.6.1_1', 'KAZ.10.3_1', 'KAZ.10.10_1', 'KAZ.6.5_1', 'KAZ.7.9_1', 'KAZ.5.14_1', 'KAZ.7.6_1', 'KAZ.1.2_1', 'KAZ.13.3_1', 'KAZ.9.8_1', 'KAZ.14.6_1', 'KAZ.3.6_1', 'KAZ.1.1_1', 'KAZ.12.10_1', 'KAZ.2.12_1', 'KAZ.9.7_1', 'KAZ.6.4_1', 'KAZ.1.5_1', 'KAZ.7.2_1', 'KAZ.14.8_1', 'KAZ.14.7_1', 'KAZ.1.15_1', 'KAZ.9.9_1', 'KAZ.11.7_1', 'KAZ.10.9_1', 'KAZ.14.4_1', 'KAZ.14.3_1', 'KAZ.7.12_1', 'KAZ.7.5_1', 'KAZ.8.4_1', 'KAZ.2.16_1', 'KAZ.12.1_1']                  
                #         elif country_code == 'CAN': admin2_lst = ['CAN.8.2_1', 'CAN.13.1_1', 'CAN.11.84_1', 'CAN.6.1_1', 'CAN.8.1_1', 'CAN.8.3_1', 'CAN.6.2_1']                   
                #     elif sub_system == 'water_supply':
                #         # if country_code == 'USA': admin2_lst = ['USA.2.21_1', 'USA.20.2_1', 'USA.2.26_1', 'USA.2.25_1', 'USA.2.17_1', 'USA.3.1_1', 'USA.2.15_1', 'USA.24.70_1', 'USA.29.5_1', 'USA.51.19_1', 'USA.45.23_1', 'USA.2.4_1', 'USA.29.17_1', 'USA.2.14_1', 'USA.29.13_1', 'USA.13.25_1', 'USA.2.16_1', 'USA.3.10_1', 'USA.45.19_1', 'USA.2.23_1', 'USA.51.7_1', 'USA.2.24_1', 'USA.51.15_1', 'USA.3.3_1', 'USA.38.19_1', 'USA.29.8_1', 'USA.29.3_1', 'USA.2.7_1', 'USA.50.34_1', 'USA.2.22_1', 'USA.23.45_1', 'USA.5.14_1', 'USA.2.6_1', 'USA.13.37_1', 'USA.2.2_1', 'USA.2.19_1']
                #         if country_code == 'KAZ': admin2_lst = ['KAZ.7.7_1', 'KAZ.1.9_1', 'KAZ.10.4_1', 'KAZ.3.13_1', 'KAZ.2.3_1', 'KAZ.4.5_1', 'KAZ.5.15_1', 'KAZ.7.2_1', 'KAZ.2.5_1', 'KAZ.7.12_1', 'KAZ.2.12_1', 'KAZ.10.17_1', 'KAZ.4.4_1', 'KAZ.10.5_1', 'KAZ.7.11_1', 'KAZ.5.4_1', 'KAZ.5.8_1', 'KAZ.5.11_1', 'KAZ.4.3_1', 'KAZ.4.7_1', 'KAZ.13.2_1', 'KAZ.9.8_1', 'KAZ.1.3_1', 'KAZ.10.10_1', 'KAZ.1.1_1', 'KAZ.9.7_1', 'KAZ.2.6_1', 'KAZ.11.5_1', 'KAZ.13.9_1', 'KAZ.13.5_1', 'KAZ.5.17_1', 'KAZ.2.10_1', 'KAZ.12.7_1', 'KAZ.5.6_1', 'KAZ.2.16_1', 'KAZ.9.9_1', 'KAZ.13.12_1', 'KAZ.13.1_1', 'KAZ.5.3_1', 'KAZ.3.11_1', 'KAZ.5.13_1', 'KAZ.10.15_1', 'KAZ.2.2_1', 'KAZ.1.14_1', 'KAZ.10.8_1', 'KAZ.12.6_1', 'KAZ.2.1_1', 'KAZ.2.14_1', 'KAZ.3.4_1', 'KAZ.11.2_1', 'KAZ.10.13_1', 'KAZ.13.8_1', 'KAZ.1.6_1', 'KAZ.2.8_1', 'KAZ.9.2_1', 'KAZ.1.17_1', 'KAZ.4.8_1', 'KAZ.1.11_1', 'KAZ.7.9_1', 'KAZ.2.9_1', 'KAZ.5.5_1', 'KAZ.6.3_1', 'KAZ.11.7_1', 'KAZ.12.10_1', 'KAZ.2.11_1', 'KAZ.1.5_1', 'KAZ.3.2_1', 'KAZ.3.10_1', 'KAZ.11.3_1', 'KAZ.7.13_1', 'KAZ.1.7_1', 'KAZ.11.1_1', 'KAZ.10.9_1', 'KAZ.2.13_1', 'KAZ.10.1_1', 'KAZ.13.7_1', 'KAZ.3.1_1', 'KAZ.6.4_1', 'KAZ.1.4_1', 'KAZ.8.9_1', 'KAZ.12.5_1', 'KAZ.5.10_1', 'KAZ.5.9_1', 'KAZ.3.8_1', 'KAZ.14.2_1', 'KAZ.3.5_1', 'KAZ.2.4_1', 'KAZ.13.4_1', 'KAZ.1.13_1', 'KAZ.1.16_1', 'KAZ.1.15_1', 'KAZ.10.6_1', 'KAZ.8.5_1', 'KAZ.11.8_1', 'KAZ.7.6_1', 'KAZ.7.4_1', 'KAZ.8.1_1', 'KAZ.5.12_1', 'KAZ.7.1_1', 'KAZ.4.2_1', 'KAZ.3.3_1', 'KAZ.12.3_1', 'KAZ.8.4_1', 'KAZ.3.7_1', 'KAZ.9.4_1', 'KAZ.13.3_1', 'KAZ.1.2_1', 'KAZ.1.12_1', 'KAZ.14.10_1', 'KAZ.2.7_1', 'KAZ.8.11_1', 'KAZ.9.3_1', 'KAZ.5.16_1', 'KAZ.8.6_1', 'KAZ.10.14_1', 'KAZ.7.8_1', 'KAZ.12.11_1', 'KAZ.12.2_1', 'KAZ.13.10_1', 'KAZ.12.13_1', 'KAZ.12.8_1', 'KAZ.2.15_1', 'KAZ.7.3_1', 'KAZ.6.2_1', 'KAZ.5.7_1', 'KAZ.3.12_1', 'KAZ.4.1_1', 'KAZ.5.14_1', 'KAZ.7.5_1']
                #         elif country_code == 'CAN': admin2_lst = ['CAN.11.84_1', 'CAN.8.1_1', 'CAN.6.2_1', 'CAN.8.2_1', 'CAN.8.3_1', 'CAN.6.1_1', 'CAN.13.1_1']                   
                #     elif sub_system == 'waste_solid':
                #         if country_code == 'KAZ': admin2_lst = ['KAZ.1.7_1', 'KAZ.2.14_1', 'KAZ.4.7_1', 'KAZ.5.8_1', 'KAZ.9.3_1', 'KAZ.3.9_1', 'KAZ.9.1_1', 'KAZ.1.3_1']
                #         #  elif country_code == 'KAZ': admin2_lst =                   
                #         elif country_code == 'CAN': admin2_lst = ['CAN.8.1_1', 'CAN.6.1_1', 'CAN.8.3_1', 'CAN.11.84_1', 'CAN.6.2_1', 'CAN.8.2_1', 'CAN.13.1_1']                   
                #     elif sub_system == 'waste_water':
                #         # if country_code == 'USA': admin2_lst = ['USA.51.19_1', 'USA.2.24_1', 'USA.24.70_1', 'USA.23.45_1', 'USA.2.26_1', 'USA.3.11_1', 'USA.2.7_1', 'USA.38.23_1', 'USA.29.17_1', 'USA.2.17_1', 'USA.2.11_1', 'USA.2.25_1', 'USA.2.2_1', 'USA.50.34_1', 'USA.29.10_1', 'USA.3.10_1', 'USA.2.6_1', 'USA.13.37_1', 'USA.2.4_1', 'USA.29.3_1', 'USA.38.13_1', 'USA.29.5_1', 'USA.2.23_1', 'USA.51.15_1', 'USA.3.1_1', 'USA.5.36_1']
                #         if country_code == 'KAZ': admin2_lst = ['KAZ.5.7_1', 'KAZ.3.10_1', 'KAZ.14.2_1', 'KAZ.4.8_1', 'KAZ.5.11_1', 'KAZ.4.7_1', 'KAZ.8.9_1', 'KAZ.1.7_1', 'KAZ.5.16_1', 'KAZ.5.9_1', 'KAZ.5.3_1', 'KAZ.12.3_1', 'KAZ.1.6_1', 'KAZ.12.10_1', 'KAZ.4.2_1', 'KAZ.8.8_1', 'KAZ.1.11_1', 'KAZ.1.17_1', 'KAZ.2.1_1', 'KAZ.10.15_1', 'KAZ.12.5_1', 'KAZ.9.3_1', 'KAZ.1.16_1', 'KAZ.3.2_1', 'KAZ.1.14_1', 'KAZ.9.2_1', 'KAZ.10.13_1', 'KAZ.1.3_1', 'KAZ.12.14_1', 'KAZ.12.11_1', 'KAZ.5.13_1', 'KAZ.2.13_1', 'KAZ.9.1_1', 'KAZ.2.15_1', 'KAZ.2.5_1', 'KAZ.1.15_1', 'KAZ.4.1_1', 'KAZ.1.9_1', 'KAZ.7.8_1', 'KAZ.13.2_1', 'KAZ.10.7_1', 'KAZ.6.3_1', 'KAZ.3.8_1', 'KAZ.2.16_1', 'KAZ.13.12_1', 'KAZ.9.8_1', 'KAZ.2.14_1', 'KAZ.5.5_1', 'KAZ.5.17_1', 'KAZ.5.10_1']                   
                #         elif country_code == 'CAN': admin2_lst =  ['CAN.8.1_1', 'CAN.8.3_1', 'CAN.6.2_1', 'CAN.8.2_1', 'CAN.13.1_1', 'CAN.6.1_1', 'CAN.11.84_1']                   
                #     elif sub_system == 'telecom':
                #         # if country_code == 'USA': admin2_lst = ['USA.2.26_1', 'USA.2.19_1', 'USA.20.2_1', 'USA.2.4_1', 'USA.51.19_1', 'USA.29.10_1', 'USA.23.47_1', 'USA.3.8_1', 'USA.3.10_1', 'USA.2.2_1', 'USA.2.22_1', 'USA.2.18_1', 'USA.3.11_1', 'USA.38.23_1', 'USA.5.36_1', 'USA.3.9_1', 'USA.51.4_1', 'USA.3.3_1', 'USA.2.14_1', 'USA.45.19_1', 'USA.2.25_1', 'USA.2.23_1', 'USA.29.3_1', 'USA.51.7_1', 'USA.29.13_1', 'USA.2.8_1', 'USA.2.7_1', 'USA.5.14_1', 'USA.2.16_1', 'USA.2.11_1', 'USA.29.17_1', 'USA.13.37_1', 'USA.45.23_1', 'USA.29.8_1', 'USA.2.17_1', 'USA.29.5_1', 'USA.38.19_1', 'USA.24.70_1', 'USA.38.13_1', 'USA.2.21_1', 'USA.2.15_1', 'USA.51.15_1']
                #         if country_code == 'KAZ': admin2_lst = ['KAZ.4.8_1', 'KAZ.1.6_1', 'KAZ.9.9_1', 'KAZ.10.6_1', 'KAZ.7.6_1', 'KAZ.12.1_1', 'KAZ.1.17_1', 'KAZ.1.1_1', 'KAZ.9.8_1', 'KAZ.9.2_1', 'KAZ.13.8_1', 'KAZ.4.3_1', 'KAZ.7.10_1', 'KAZ.11.8_1', 'KAZ.14.10_1', 'KAZ.12.12_1', 'KAZ.9.5_1', 'KAZ.8.4_1', 'KAZ.7.3_1', 'KAZ.11.1_1', 'KAZ.6.2_1', 'KAZ.1.15_1', 'KAZ.9.7_1', 'KAZ.10.3_1', 'KAZ.2.4_1', 'KAZ.2.13_1', 'KAZ.10.16_1', 'KAZ.8.7_1', 'KAZ.8.6_1', 'KAZ.10.7_1', 'KAZ.2.12_1', 'KAZ.4.6_1', 'KAZ.12.6_1', 'KAZ.3.12_1', 'KAZ.1.12_1', 'KAZ.14.7_1', 'KAZ.10.17_1', 'KAZ.5.1_1', 'KAZ.5.16_1', 'KAZ.2.14_1', 'KAZ.5.11_1', 'KAZ.5.15_1', 'KAZ.5.5_1', 'KAZ.3.9_1', 'KAZ.6.4_1', 'KAZ.1.2_1', 'KAZ.14.8_1', 'KAZ.10.11_1', 'KAZ.12.11_1', 'KAZ.13.11_1', 'KAZ.11.4_1', 'KAZ.1.13_1', 'KAZ.11.6_1', 'KAZ.3.7_1', 'KAZ.1.14_1', 'KAZ.11.2_1', 'KAZ.1.11_1', 'KAZ.13.10_1', 'KAZ.3.6_1', 'KAZ.5.12_1', 'KAZ.2.16_1', 'KAZ.4.7_1', 'KAZ.9.1_1', 'KAZ.8.5_1', 'KAZ.7.2_1', 'KAZ.7.12_1', 'KAZ.12.2_1', 'KAZ.5.9_1', 'KAZ.3.10_1', 'KAZ.8.9_1', 'KAZ.1.7_1', 'KAZ.2.5_1', 'KAZ.2.9_1', 'KAZ.14.6_1', 'KAZ.12.14_1', 'KAZ.5.17_1', 'KAZ.13.2_1', 'KAZ.5.14_1', 'KAZ.5.4_1', 'KAZ.10.12_1', 'KAZ.10.14_1', 'KAZ.5.7_1', 'KAZ.1.5_1', 'KAZ.5.13_1', 'KAZ.2.3_1', 'KAZ.12.4_1', 'KAZ.12.5_1', 'KAZ.10.4_1', 'KAZ.1.9_1', 'KAZ.8.12_1', 'KAZ.13.12_1', 'KAZ.8.10_1', 'KAZ.9.4_1', 'KAZ.5.6_1', 'KAZ.14.5_1', 'KAZ.2.10_1', 'KAZ.8.1_1', 'KAZ.13.5_1', 'KAZ.7.4_1', 'KAZ.8.3_1', 'KAZ.7.1_1', 'KAZ.8.11_1', 'KAZ.2.11_1', 'KAZ.6.5_1', 'KAZ.1.4_1', 'KAZ.2.7_1', 'KAZ.10.10_1', 'KAZ.11.5_1', 'KAZ.7.9_1', 'KAZ.10.9_1', 'KAZ.12.9_1', 'KAZ.10.2_1', 'KAZ.11.3_1', 'KAZ.1.8_1', 'KAZ.12.13_1', 'KAZ.10.15_1', 'KAZ.2.1_1', 'KAZ.8.2_1', 'KAZ.9.3_1', 'KAZ.1.3_1', 'KAZ.4.5_1', 'KAZ.9.6_1', 'KAZ.3.3_1', 'KAZ.5.8_1', 'KAZ.2.15_1', 'KAZ.7.11_1', 'KAZ.3.4_1', 'KAZ.13.3_1', 'KAZ.4.1_1', 'KAZ.5.3_1', 'KAZ.3.11_1', 'KAZ.12.8_1', 'KAZ.2.6_1', 'KAZ.13.1_1', 'KAZ.2.17_1', 'KAZ.2.8_1', 'KAZ.3.8_1', 'KAZ.1.10_1', 'KAZ.14.12_1', 'KAZ.13.7_1', 'KAZ.14.2_1', 'KAZ.4.4_1', 'KAZ.3.1_1', 'KAZ.5.2_1', 'KAZ.6.1_1', 'KAZ.3.2_1', 'KAZ.4.2_1', 'KAZ.14.4_1', 'KAZ.12.7_1', 'KAZ.13.9_1', 'KAZ.5.10_1', 'KAZ.7.5_1', 'KAZ.10.13_1', 'KAZ.3.5_1']                
                #         elif country_code == 'CAN': admin2_lst =  ['CAN.8.3_1', 'CAN.8.1_1', 'CAN.13.1_1', 'CAN.6.1_1', 'CAN.11.84_1', 'CAN.8.2_1', 'CAN.6.2_1']                   
                #     elif sub_system == 'healthcare':
                #         # if country_code == 'USA': admin2_lst = ['USA.2.22_1', 'USA.2.19_1', 'USA.29.17_1', 'USA.2.8_1', 'USA.38.23_1', 'USA.2.16_1', 'USA.13.25_1', 'USA.20.2_1', 'USA.2.26_1', 'USA.2.1_1', 'USA.5.36_1', 'USA.13.37_1', 'USA.2.14_1', 'USA.51.7_1', 'USA.2.2_1', 'USA.2.17_1', 'USA.3.9_1', 'USA.50.34_1', 'USA.29.3_1', 'USA.29.13_1', 'USA.2.24_1', 'USA.29.8_1', 'USA.51.15_1', 'USA.2.18_1', 'USA.2.25_1', 'USA.3.3_1', 'USA.3.1_1', 'USA.3.8_1', 'USA.2.6_1', 'USA.2.4_1', 'USA.38.13_1', 'USA.5.14_1', 'USA.45.19_1', 'USA.23.45_1', 'USA.2.21_1', 'USA.45.23_1', 'USA.3.10_1']
                #         if country_code == 'KAZ': admin2_lst = ['KAZ.9.7_1', 'KAZ.2.11_1', 'KAZ.8.3_1', 'KAZ.10.16_1', 'KAZ.10.9_1', 'KAZ.2.6_1', 'KAZ.8.12_1', 'KAZ.10.17_1', 'KAZ.2.13_1', 'KAZ.11.5_1', 'KAZ.7.13_1', 'KAZ.4.3_1', 'KAZ.13.9_1', 'KAZ.1.14_1', 'KAZ.13.2_1', 'KAZ.14.3_1', 'KAZ.4.4_1', 'KAZ.5.17_1', 'KAZ.6.5_1', 'KAZ.7.6_1', 'KAZ.5.8_1', 'KAZ.2.7_1', 'KAZ.13.12_1', 'KAZ.5.13_1', 'KAZ.8.9_1', 'KAZ.13.1_1', 'KAZ.12.11_1', 'KAZ.5.14_1', 'KAZ.1.1_1', 'KAZ.4.7_1', 'KAZ.6.4_1', 'KAZ.8.8_1', 'KAZ.1.7_1', 'KAZ.10.5_1', 'KAZ.14.6_1', 'KAZ.7.7_1', 'KAZ.3.5_1', 'KAZ.1.11_1', 'KAZ.12.7_1', 'KAZ.3.8_1', 'KAZ.11.7_1', 'KAZ.2.1_1', 'KAZ.14.7_1', 'KAZ.1.2_1', 'KAZ.2.2_1', 'KAZ.11.4_1', 'KAZ.13.11_1', 'KAZ.5.9_1', 'KAZ.3.3_1', 'KAZ.10.14_1', 'KAZ.12.9_1', 'KAZ.11.1_1', 'KAZ.2.4_1', 'KAZ.4.8_1', 'KAZ.5.6_1', 'KAZ.4.2_1', 'KAZ.13.10_1', 'KAZ.9.9_1', 'KAZ.2.14_1', 'KAZ.12.1_1', 'KAZ.1.8_1', 'KAZ.2.16_1', 'KAZ.13.4_1', 'KAZ.3.1_1', 'KAZ.9.1_1', 'KAZ.11.3_1', 'KAZ.7.9_1', 'KAZ.12.6_1', 'KAZ.9.5_1', 'KAZ.9.4_1', 'KAZ.2.8_1', 'KAZ.2.9_1', 'KAZ.1.13_1', 'KAZ.7.4_1', 'KAZ.4.1_1', 'KAZ.10.15_1', 'KAZ.13.3_1', 'KAZ.14.12_1', 'KAZ.10.13_1', 'KAZ.5.2_1', 'KAZ.12.12_1', 'KAZ.1.5_1', 'KAZ.12.4_1', 'KAZ.5.16_1', 'KAZ.8.1_1', 'KAZ.1.10_1', 'KAZ.7.2_1', 'KAZ.6.1_1', 'KAZ.8.5_1', 'KAZ.5.5_1', 'KAZ.1.4_1', 'KAZ.14.8_1', 'KAZ.1.3_1', 'KAZ.3.11_1', 'KAZ.5.7_1', 'KAZ.12.3_1', 'KAZ.1.17_1', 'KAZ.7.12_1', 'KAZ.1.12_1', 'KAZ.8.11_1', 'KAZ.12.5_1', 'KAZ.9.3_1', 'KAZ.7.1_1', 'KAZ.10.12_1', 'KAZ.1.9_1', 'KAZ.14.1_1', 'KAZ.8.2_1', 'KAZ.14.10_1', 'KAZ.2.15_1', 'KAZ.2.3_1', 'KAZ.5.3_1', 'KAZ.2.10_1', 'KAZ.10.8_1', 'KAZ.3.7_1', 'KAZ.5.11_1', 'KAZ.8.7_1', 'KAZ.4.6_1', 'KAZ.8.4_1', 'KAZ.12.13_1', 'KAZ.10.4_1', 'KAZ.11.6_1', 'KAZ.2.12_1', 'KAZ.3.13_1', 'KAZ.8.10_1', 'KAZ.1.16_1', 'KAZ.3.4_1', 'KAZ.13.7_1', 'KAZ.3.2_1', 'KAZ.13.6_1', 'KAZ.14.4_1', 'KAZ.2.17_1', 'KAZ.9.6_1', 'KAZ.7.5_1', 'KAZ.5.10_1', 'KAZ.12.14_1', 'KAZ.5.12_1', 'KAZ.1.15_1', 'KAZ.6.3_1', 'KAZ.3.9_1', 'KAZ.7.3_1', 'KAZ.13.5_1', 'KAZ.3.6_1', 'KAZ.12.8_1', 'KAZ.4.5_1', 'KAZ.10.3_1', 'KAZ.9.2_1', 'KAZ.7.11_1', 'KAZ.3.10_1', 'KAZ.10.7_1', 'KAZ.10.6_1', 'KAZ.11.2_1', 'KAZ.6.2_1', 'KAZ.1.6_1', 'KAZ.12.10_1', 'KAZ.5.15_1', 'KAZ.7.8_1', 'KAZ.5.4_1', 'KAZ.13.8_1', 'KAZ.11.8_1', 'KAZ.10.10_1', 'KAZ.3.12_1', 'KAZ.14.2_1', 'KAZ.7.10_1', 'KAZ.9.8_1', 'KAZ.10.11_1', 'KAZ.12.2_1']              
                #         elif country_code == 'CAN': admin2_lst = ['CAN.6.1_1', 'CAN.8.3_1', 'CAN.13.1_1', 'CAN.6.2_1', 'CAN.8.1_1', 'CAN.11.84_1', 'CAN.8.2_1']                   
                #     elif sub_system == 'education':
                #         # if country_code == 'USA': admin2_lst = ['USA.29.3_1', 'USA.2.7_1', 'USA.29.5_1', 'USA.51.19_1', 'USA.51.7_1', 'USA.3.11_1', 'USA.5.14_1', 'USA.2.16_1', 'USA.51.15_1', 'USA.2.11_1', 'USA.3.10_1', 'USA.24.70_1', 'USA.2.14_1', 'USA.3.9_1', 'USA.2.17_1', 'USA.2.19_1', 'USA.5.36_1', 'USA.2.6_1', 'USA.2.2_1', 'USA.13.37_1', 'USA.2.4_1', 'USA.29.13_1', 'USA.38.19_1', 'USA.51.4_1', 'USA.38.23_1', 'USA.2.18_1', 'USA.2.21_1', 'USA.45.19_1', 'USA.2.22_1', 'USA.2.24_1', 'USA.3.3_1', 'USA.2.8_1', 'USA.2.13_1', 'USA.23.44_1', 'USA.2.25_1', 'USA.20.2_1', 'USA.38.13_1', 'USA.2.23_1', 'USA.2.15_1', 'USA.23.45_1', 'USA.50.34_1', 'USA.45.23_1', 'USA.29.17_1', 'USA.23.47_1', 'USA.13.25_1', 'USA.3.8_1', 'USA.3.1_1', 'USA.2.1_1', 'USA.29.10_1', 'USA.29.8_1', 'USA.2.26_1']
                #         if country_code == 'KAZ': admin2_lst = ['KAZ.5.15_1', 'KAZ.7.3_1', 'KAZ.13.11_1', 'KAZ.2.10_1', 'KAZ.12.8_1', 'KAZ.1.7_1', 'KAZ.5.13_1', 'KAZ.5.10_1', 'KAZ.3.8_1', 'KAZ.2.5_1', 'KAZ.1.17_1', 'KAZ.13.12_1', 'KAZ.12.14_1', 'KAZ.11.1_1', 'KAZ.10.8_1', 'KAZ.12.7_1', 'KAZ.7.1_1', 'KAZ.1.2_1', 'KAZ.2.8_1', 'KAZ.2.6_1', 'KAZ.14.8_1', 'KAZ.2.11_1', 'KAZ.11.6_1', 'KAZ.1.10_1', 'KAZ.10.14_1', 'KAZ.10.9_1', 'KAZ.3.1_1', 'KAZ.12.2_1', 'KAZ.12.5_1', 'KAZ.2.4_1', 'KAZ.5.17_1', 'KAZ.14.7_1', 'KAZ.10.11_1', 'KAZ.7.6_1', 'KAZ.12.9_1', 'KAZ.10.2_1', 'KAZ.4.4_1', 'KAZ.11.3_1', 'KAZ.3.6_1', 'KAZ.10.4_1', 'KAZ.13.9_1', 'KAZ.1.4_1', 'KAZ.14.2_1', 'KAZ.6.1_1', 'KAZ.2.13_1', 'KAZ.2.12_1', 'KAZ.11.2_1', 'KAZ.8.4_1', 'KAZ.13.7_1', 'KAZ.1.11_1', 'KAZ.10.7_1', 'KAZ.14.4_1', 'KAZ.10.12_1', 'KAZ.1.9_1', 'KAZ.6.2_1', 'KAZ.7.8_1', 'KAZ.11.7_1', 'KAZ.1.15_1', 'KAZ.12.11_1', 'KAZ.1.8_1', 'KAZ.10.3_1', 'KAZ.4.7_1', 'KAZ.2.7_1', 'KAZ.12.1_1', 'KAZ.6.5_1', 'KAZ.3.13_1', 'KAZ.2.9_1', 'KAZ.8.12_1', 'KAZ.2.2_1', 'KAZ.3.9_1', 'KAZ.5.14_1', 'KAZ.7.2_1', 'KAZ.2.16_1', 'KAZ.4.6_1', 'KAZ.9.4_1', 'KAZ.5.6_1', 'KAZ.5.11_1', 'KAZ.6.3_1', 'KAZ.2.17_1', 'KAZ.13.10_1', 'KAZ.13.6_1', 'KAZ.13.4_1', 'KAZ.10.10_1', 'KAZ.11.8_1', 'KAZ.14.1_1', 'KAZ.5.7_1', 'KAZ.9.1_1', 'KAZ.10.13_1', 'KAZ.12.3_1', 'KAZ.9.2_1', 'KAZ.1.14_1', 'KAZ.8.6_1', 'KAZ.1.16_1', 'KAZ.5.4_1', 'KAZ.14.6_1', 'KAZ.10.17_1', 'KAZ.7.10_1', 'KAZ.10.6_1', 'KAZ.4.5_1', 'KAZ.8.9_1', 'KAZ.1.12_1', 'KAZ.8.5_1', 'KAZ.3.5_1', 'KAZ.7.5_1', 'KAZ.2.1_1', 'KAZ.8.1_1', 'KAZ.14.3_1', 'KAZ.5.9_1', 'KAZ.5.12_1', 'KAZ.8.11_1', 'KAZ.8.10_1', 'KAZ.12.4_1', 'KAZ.3.3_1', 'KAZ.5.3_1', 'KAZ.7.7_1', 'KAZ.4.1_1', 'KAZ.6.4_1', 'KAZ.3.10_1', 'KAZ.1.13_1', 'KAZ.11.5_1', 'KAZ.2.3_1', 'KAZ.14.12_1', 'KAZ.8.2_1', 'KAZ.7.12_1', 'KAZ.1.3_1', 'KAZ.4.2_1', 'KAZ.10.5_1', 'KAZ.3.2_1', 'KAZ.9.8_1', 'KAZ.3.12_1', 'KAZ.1.1_1', 'KAZ.12.10_1', 'KAZ.8.7_1', 'KAZ.9.3_1', 'KAZ.2.15_1', 'KAZ.5.5_1', 'KAZ.8.3_1', 'KAZ.12.13_1', 'KAZ.5.8_1', 'KAZ.7.11_1', 'KAZ.9.7_1', 'KAZ.5.2_1', 'KAZ.13.8_1', 'KAZ.7.4_1', 'KAZ.8.8_1', 'KAZ.14.10_1', 'KAZ.1.5_1', 'KAZ.13.3_1', 'KAZ.3.4_1', 'KAZ.4.8_1', 'KAZ.13.5_1', 'KAZ.13.2_1', 'KAZ.12.12_1', 'KAZ.9.9_1', 'KAZ.7.9_1', 'KAZ.2.14_1', 'KAZ.1.6_1', 'KAZ.5.16_1', 'KAZ.12.6_1', 'KAZ.3.7_1', 'KAZ.4.3_1', 'KAZ.9.5_1', 'KAZ.7.13_1', 'KAZ.3.11_1', 'KAZ.10.15_1', 'KAZ.11.4_1', 'KAZ.13.1_1', 'KAZ.14.5_1', 'KAZ.9.6_1']             
                #         elif country_code == 'CAN': admin2_lst =  ['CAN.8.2_1', 'CAN.8.3_1', 'CAN.6.2_1', 'CAN.6.1_1', 'CAN.13.1_1', 'CAN.8.1_1', 'CAN.11.84_1']               
                # elif 'coastal' in hazard_types:
                #    if sub_system == 'power':
                #         if country_code == 'CAN': admin2_lst = ['CAN.8.3_1', 'CAN.11.84_1', 'CAN.13.1_1', 'CAN.8.2_1', 'CAN.6.1_1', 'CAN.6.2_1', 'CAN.8.1_1']
                #         elif country_code == 'CHN': admin2_lst =  ['CHN.11.10_1', 'CHN.11.12_1', 'CHN.11.1_1', 'CHN.11.11_1', 'CHN.11.3_1', 'CHN.11.6_1', 'CHN.11.4_1', 'CHN.11.2_1', 'CHN.11.13_1', 'CHN.28.2_1', 'CHN.17.1_1', 'CHN.11.9_1', 'CHN.11.5_1', 'CHN.19.6_1', 'CHN.11.7_1', 'CHN.28.4_1', 'CHN.28.8_1', 'CHN.28.12_1', 'CHN.19.12_1']
                #         # elif country_code == 'USA': admin2_lst = 
                #    elif sub_system == 'road':
                #         if country_code == 'CAN': admin2_lst = ['CAN.13.1_1', 'CAN.6.2_1', 'CAN.6.1_1', 'CAN.8.1_1', 'CAN.11.84_1', 'CAN.8.2_1', 'CAN.8.3_1']
                #         elif country_code == 'CHN': admin2_lst = ['CHN.19.12_1', 'CHN.11.5_1', 'CHN.11.6_1', 'CHN.28.12_1', 'CHN.28.2_1', 'CHN.11.1_1', 'CHN.11.12_1', 'CHN.11.4_1', 'CHN.11.9_1', 'CHN.11.11_1', 'CHN.11.7_1', 'CHN.28.4_1', 'CHN.28.8_1', 'CHN.19.6_1', 'CHN.11.3_1', 'CHN.17.1_1', 'CHN.11.2_1', 'CHN.11.13_1', 'CHN.11.10_1']
                #         # elif country_code == 'USA': admin2_lst = 
                #    elif sub_system == 'air':
                #         if country_code == 'CAN': admin2_lst = ['CAN.8.2_1', 'CAN.6.2_1', 'CAN.8.1_1', 'CAN.13.1_1', 'CAN.6.1_1', 'CAN.11.84_1', 'CAN.8.3_1']
                #         elif country_code == 'CHN': admin2_lst = ['CHN.11.9_1', 'CHN.17.1_1', 'CHN.11.2_1', 'CHN.11.13_1', 'CHN.28.12_1', 'CHN.28.2_1', 'CHN.11.5_1', 'CHN.11.4_1', 'CHN.19.12_1', 'CHN.28.4_1', 'CHN.19.6_1', 'CHN.11.11_1', 'CHN.28.8_1', 'CHN.11.7_1', 'CHN.11.12_1', 'CHN.11.3_1', 'CHN.11.6_1']
                #         # elif country_code == 'USA': admin2_lst = 
                #    elif sub_system == 'rail':
                #         if country_code == 'CAN': admin2_lst = ['CAN.13.1_1', 'CAN.11.84_1', 'CAN.6.1_1']
                #         elif country_code == 'CHN': admin2_lst =  ['CHN.28.2_1', 'CHN.11.1_1', 'CHN.28.8_1', 'CHN.28.12_1', 'CHN.19.12_1', 'CHN.19.6_1', 'CHN.11.12_1', 'CHN.11.3_1', 'CHN.11.7_1', 'CHN.11.6_1', 'CHN.11.2_1', 'CHN.11.13_1', 'CHN.11.9_1', 'CHN.11.4_1', 'CHN.11.10_1', 'CHN.17.1_1', 'CHN.11.5_1', 'CHN.11.11_1', 'CHN.28.4_1']
                #         # elif country_code == 'USA': admin2_lst = 
                #    elif sub_system == 'water_supply':
                #         if country_code == 'CAN': admin2_lst = ['CAN.6.2_1', 'CAN.11.84_1', 'CAN.6.1_1', 'CAN.13.1_1']
                #         elif country_code == 'CHN': admin2_lst = ['CHN.11.12_1', 'CHN.17.1_1', 'CHN.19.6_1', 'CHN.28.4_1', 'CHN.28.12_1', 'CHN.11.5_1', 'CHN.11.1_1', 'CHN.28.8_1', 'CHN.11.6_1', 'CHN.28.2_1', 'CHN.11.3_1', 'CHN.19.12_1', 'CHN.11.9_1', 'CHN.11.7_1']
                #         # elif country_code == 'USA': admin2_lst = 
                #    elif sub_system == 'waste_solid':
                #         if country_code == 'CAN': admin2_lst = ['CAN.6.1_1', 'CAN.8.1_1', 'CAN.13.1_1'] 
                #         elif country_code == 'CHN': admin2_lst = ['CHN.11.7_1', 'CHN.11.3_1']
                #         # elif country_code == 'USA': admin2_lst = 
                #    elif sub_system == 'waste_water':
                #         if country_code == 'CAN': admin2_lst = ['CAN.8.2_1', 'CAN.6.2_1', 'CAN.6.1_1', 'CAN.8.3_1', 'CAN.11.84_1', 'CAN.13.1_1', 'CAN.8.1_1']
                #         elif country_code == 'CHN': admin2_lst = ['CHN.17.1_1', 'CHN.19.12_1', 'CHN.28.8_1', 'CHN.28.4_1', 'CHN.11.12_1', 'CHN.11.9_1', 'CHN.11.5_1', 'CHN.11.3_1', 'CHN.11.1_1', 'CHN.11.7_1', 'CHN.28.12_1', 'CHN.19.6_1', 'CHN.11.13_1']
                #         # elif country_code == 'USA': admin2_lst = 
                #    elif sub_system == 'telecom':
                #         if country_code == 'CAN': admin2_lst = ['CAN.6.2_1', 'CAN.8.1_1', 'CAN.8.3_1', 'CAN.8.2_1', 'CAN.11.84_1', 'CAN.6.1_1', 'CAN.13.1_1']
                #         elif country_code == 'CHN': admin2_lst = ['CHN.11.9_1', 'CHN.17.1_1', 'CHN.11.2_1', 'CHN.11.12_1', 'CHN.11.7_1', 'CHN.11.4_1', 'CHN.28.12_1', 'CHN.19.6_1', 'CHN.11.1_1', 'CHN.11.5_1', 'CHN.28.8_1', 'CHN.19.12_1', 'CHN.28.4_1', 'CHN.11.11_1', 'CHN.11.13_1', 'CHN.28.2_1', 'CHN.11.6_1', 'CHN.11.3_1']
                #         # elif country_code == 'USA': admin2_lst = 
                #    elif sub_system == 'healthcare':
                #         if country_code == 'CAN': admin2_lst = ['CAN.6.2_1', 'CAN.8.1_1', 'CAN.13.1_1', 'CAN.8.3_1', 'CAN.8.2_1', 'CAN.11.84_1', 'CAN.6.1_1']
                #         elif country_code == 'CHN': admin2_lst = ['CHN.17.1_1', 'CHN.11.9_1', 'CHN.11.4_1', 'CHN.28.2_1', 'CHN.28.8_1', 'CHN.11.12_1', 'CHN.11.2_1', 'CHN.11.13_1', 'CHN.11.5_1', 'CHN.11.1_1', 'CHN.19.12_1', 'CHN.11.6_1', 'CHN.19.6_1', 'CHN.11.11_1', 'CHN.11.10_1', 'CHN.11.7_1', 'CHN.28.12_1', 'CHN.11.3_1']
                #         # elif country_code == 'USA': admin2_lst = 
                #    elif sub_system == 'education':
                #         if country_code == 'CAN': admin2_lst = ['CAN.11.84_1', 'CAN.8.1_1', 'CAN.13.1_1', 'CAN.8.3_1', 'CAN.8.2_1', 'CAN.6.1_1', 'CAN.6.2_1']
                #         elif country_code == 'CHN': admin2_lst = ['CHN.11.7_1', 'CHN.11.2_1', 'CHN.11.5_1', 'CHN.11.6_1', 'CHN.11.10_1', 'CHN.28.2_1', 'CHN.28.8_1', 'CHN.28.12_1', 'CHN.17.1_1', 'CHN.11.3_1', 'CHN.11.13_1', 'CHN.11.1_1', 'CHN.11.12_1', 'CHN.11.4_1', 'CHN.19.6_1', 'CHN.19.12_1', 'CHN.11.9_1', 'CHN.11.11_1', 'CHN.28.4_1']
                #         # elif country_code == 'USA': admin2_lst = 
                # elif 'pluvial' in hazard_types:
                #    if sub_system == 'power':
                #         # if country_code == 'RUS': admin2_lst = ['RUS.60.3_1', 'RUS.12.8_1', 'RUS.60.31_1', 'RUS.12.3_1', 'RUS.60.23_1', 'RUS.12.1_1', 'RUS.35.30_1', 'RUS.60.37_1', 'RUS.80.11_1', 'RUS.4.13_1', 'RUS.60.14_1', 'RUS.35.15_1', 'RUS.35.13_1', 'RUS.80.12_1', 'RUS.60.25_1', 'RUS.60.6_1', 'RUS.46.2_1', 'RUS.35.53_1', 'RUS.35.23_1']
                #         if country_code == 'AUS': admin2_lst =  ['AUS.11.95_1', 'AUS.6.2_1', 'AUS.5.34_1', 'AUS.11.138_1', 'AUS.11.40_1', 'AUS.11.121_1', 'AUS.11.109_1', 'AUS.11.75_1', 'AUS.7.56_1', 'AUS.9.22_1', 'AUS.11.133_1', 'AUS.7.42_1', 'AUS.7.69_1', 'AUS.7.6_1', 'AUS.5.138_1', 'AUS.7.54_1', 'AUS.6.15_1', 'AUS.7.34_1', 'AUS.11.46_1', 'AUS.7.52_1', 'AUS.6.14_1', 'AUS.5.20_1', 'AUS.7.24_1', 'AUS.5.6_1', 'AUS.11.137_1', 'AUS.8.4_1', 'AUS.5.145_1', 'AUS.7.16_1', 'AUS.7.39_1', 'AUS.7.9_1', 'AUS.11.71_1', 'AUS.10.46_1', 'AUS.11.77_1', 'AUS.8.61_1', 'AUS.11.3_1', 'AUS.11.13_1', 'AUS.11.47_1', 'AUS.6.12_1', 'AUS.7.2_1', 'AUS.11.61_1', 'AUS.7.21_1', 'AUS.7.46_1', 'AUS.7.7_1', 'AUS.6.10_1', 'AUS.7.17_1', 'AUS.7.12_1', 'AUS.7.19_1', 'AUS.11.72_1', 'AUS.10.19_1', 'AUS.11.55_1', 'AUS.7.22_1', 'AUS.11.88_1', 'AUS.7.4_1', 'AUS.11.44_1', 'AUS.7.63_1', 'AUS.7.47_1', 'AUS.5.30_1', 'AUS.11.139_1', 'AUS.7.5_1', 'AUS.7.43_1', 'AUS.7.3_1', 'AUS.6.4_1', 'AUS.7.20_1', 'AUS.11.22_1', 'AUS.7.71_1', 'AUS.8.31_1', 'AUS.5.130_1', 'AUS.6.17_1', 'AUS.7.14_1', 'AUS.11.28_1', 'AUS.6.7_1', 'AUS.5.31_1', 'AUS.7.25_1']
                #         elif country_code == 'LBY': admin2_lst =  ['LBY.6_1']
                #    elif sub_system == 'road':
                #         # if country_code == 'RUS': admin2_lst = ['RUS.60.31_1', 'RUS.46.2_1', 'RUS.80.12_1', 'RUS.35.53_1', 'RUS.60.3_1', 'RUS.60.25_1', 'RUS.35.15_1', 'RUS.4.13_1', 'RUS.12.3_1', 'RUS.12.1_1', 'RUS.60.6_1', 'RUS.35.23_1', 'RUS.80.11_1', 'RUS.35.30_1', 'RUS.60.23_1', 'RUS.12.8_1', 'RUS.60.14_1', 'RUS.35.13_1', 'RUS.60.37_1']
                #         if country_code == 'AUS': admin2_lst =  ['AUS.7.7_1', 'AUS.7.3_1', 'AUS.11.71_1', 'AUS.11.55_1', 'AUS.6.7_1', 'AUS.7.52_1', 'AUS.6.17_1', 'AUS.7.63_1', 'AUS.11.139_1', 'AUS.5.30_1', 'AUS.7.14_1', 'AUS.11.88_1', 'AUS.11.72_1', 'AUS.7.4_1', 'AUS.7.9_1', 'AUS.11.22_1', 'AUS.11.121_1', 'AUS.11.46_1', 'AUS.11.138_1', 'AUS.8.31_1', 'AUS.7.56_1', 'AUS.5.31_1', 'AUS.7.39_1', 'AUS.8.61_1', 'AUS.7.47_1', 'AUS.6.14_1', 'AUS.11.112_1', 'AUS.11.44_1', 'AUS.7.16_1', 'AUS.7.6_1', 'AUS.11.75_1', 'AUS.11.28_1', 'AUS.6.4_1', 'AUS.7.46_1', 'AUS.11.137_1', 'AUS.6.12_1', 'AUS.11.3_1', 'AUS.7.69_1', 'AUS.5.6_1', 'AUS.5.20_1', 'AUS.5.138_1', 'AUS.7.25_1', 'AUS.6.15_1', 'AUS.11.47_1', 'AUS.10.46_1', 'AUS.11.95_1', 'AUS.7.42_1', 'AUS.6.2_1', 'AUS.7.21_1', 'AUS.7.54_1', 'AUS.7.71_1', 'AUS.8.4_1', 'AUS.5.145_1', 'AUS.7.34_1', 'AUS.7.43_1', 'AUS.11.13_1', 'AUS.7.19_1', 'AUS.11.40_1', 'AUS.7.17_1', 'AUS.10.19_1', 'AUS.11.133_1', 'AUS.7.24_1', 'AUS.5.34_1', 'AUS.9.22_1', 'AUS.7.2_1', 'AUS.7.5_1', 'AUS.11.77_1', 'AUS.7.12_1', 'AUS.11.61_1', 'AUS.11.109_1', 'AUS.7.22_1', 'AUS.6.10_1', 'AUS.5.130_1', 'AUS.7.20_1']
                #         elif country_code == 'LBY': admin2_lst =  ['LBY.6_1']
                #    elif sub_system == 'air':
                #         # if country_code == 'RUS': admin2_lst =  ['RUS.35.13_1', 'RUS.60.23_1', 'RUS.60.25_1', 'RUS.35.53_1', 'RUS.46.2_1', 'RUS.80.12_1', 'RUS.35.30_1', 'RUS.12.3_1', 'RUS.60.37_1', 'RUS.12.1_1', 'RUS.60.3_1', 'RUS.60.14_1', 'RUS.60.6_1', 'RUS.35.15_1', 'RUS.4.13_1', 'RUS.80.11_1', 'RUS.35.23_1', 'RUS.60.31_1', 'RUS.12.8_1']
                #         if country_code == 'AUS': admin2_lst =  ['AUS.6.15_1', 'AUS.6.4_1', 'AUS.6.12_1', 'AUS.11.61_1', 'AUS.11.75_1', 'AUS.7.17_1', 'AUS.6.7_1', 'AUS.11.137_1', 'AUS.6.17_1', 'AUS.11.138_1', 'AUS.11.109_1', 'AUS.7.71_1', 'AUS.7.24_1', 'AUS.11.112_1', 'AUS.11.40_1', 'AUS.11.3_1', 'AUS.7.47_1', 'AUS.5.34_1', 'AUS.5.130_1', 'AUS.7.5_1', 'AUS.7.19_1', 'AUS.7.6_1', 'AUS.7.4_1', 'AUS.7.12_1', 'AUS.7.56_1', 'AUS.11.77_1', 'AUS.7.42_1', 'AUS.5.20_1', 'AUS.5.145_1', 'AUS.11.47_1', 'AUS.7.43_1', 'AUS.5.6_1', 'AUS.11.55_1', 'AUS.7.7_1', 'AUS.8.61_1', 'AUS.5.30_1', 'AUS.6.10_1', 'AUS.11.22_1', 'AUS.7.2_1', 'AUS.7.34_1', 'AUS.7.25_1', 'AUS.10.46_1', 'AUS.7.20_1', 'AUS.8.4_1', 'AUS.5.31_1', 'AUS.6.14_1', 'AUS.11.133_1', 'AUS.11.88_1', 'AUS.7.9_1', 'AUS.11.72_1', 'AUS.11.121_1', 'AUS.7.69_1', 'AUS.7.21_1', 'AUS.7.54_1', 'AUS.6.2_1', 'AUS.11.71_1', 'AUS.11.44_1', 'AUS.11.13_1', 'AUS.11.139_1', 'AUS.7.14_1', 'AUS.10.19_1', 'AUS.7.39_1', 'AUS.11.46_1', 'AUS.7.16_1', 'AUS.11.95_1', 'AUS.7.52_1', 'AUS.11.28_1', 'AUS.7.3_1', 'AUS.7.46_1', 'AUS.5.138_1', 'AUS.7.63_1', 'AUS.7.22_1', 'AUS.8.31_1']
                #         elif country_code == 'LBY': admin2_lst =  ['LBY.6_1']
                #    elif sub_system == 'rail':
                #         if country_code == 'AUS': admin2_lst = ['AUS.5.34_1', 'AUS.11.109_1', 'AUS.7.7_1', 'AUS.11.137_1', 'AUS.6.17_1', 'AUS.11.138_1', 'AUS.7.71_1', 'AUS.7.24_1', 'AUS.8.31_1', 'AUS.7.54_1', 'AUS.8.61_1', 'AUS.11.95_1', 'AUS.11.61_1', 'AUS.7.16_1', 'AUS.7.19_1', 'AUS.7.56_1', 'AUS.7.6_1', 'AUS.11.44_1', 'AUS.7.4_1', 'AUS.7.12_1', 'AUS.6.14_1', 'AUS.11.75_1', 'AUS.5.138_1', 'AUS.8.4_1', 'AUS.6.4_1', 'AUS.7.17_1', 'AUS.7.9_1', 'AUS.5.6_1', 'AUS.5.20_1', 'AUS.5.145_1', 'AUS.11.47_1', 'AUS.7.43_1', 'AUS.11.55_1', 'AUS.7.42_1', 'AUS.11.46_1', 'AUS.7.47_1', 'AUS.5.130_1', 'AUS.11.22_1', 'AUS.10.46_1', 'AUS.7.2_1', 'AUS.7.34_1', 'AUS.6.12_1', 'AUS.7.25_1', 'AUS.7.20_1', 'AUS.5.31_1', 'AUS.6.7_1', 'AUS.11.88_1', 'AUS.7.69_1', 'AUS.11.72_1', 'AUS.11.121_1', 'AUS.7.21_1', 'AUS.5.30_1', 'AUS.11.71_1', 'AUS.11.77_1', 'AUS.11.133_1', 'AUS.7.39_1', 'AUS.7.14_1', 'AUS.7.22_1', 'AUS.6.10_1', 'AUS.11.13_1', 'AUS.11.139_1', 'AUS.10.19_1', 'AUS.7.5_1', 'AUS.6.2_1', 'AUS.7.52_1', 'AUS.7.46_1', 'AUS.11.28_1', 'AUS.7.63_1', 'AUS.11.40_1', 'AUS.7.3_1', 'AUS.11.3_1', 'AUS.6.15_1']
                   
                #    elif sub_system == 'water_supply':
                #         if country_code == 'AUS': admin2_lst =  ['AUS.11.139_1', 'AUS.7.63_1', 'AUS.7.47_1', 'AUS.7.12_1', 'AUS.6.7_1', 'AUS.5.34_1', 'AUS.11.75_1', 'AUS.7.5_1', 'AUS.6.12_1', 'AUS.7.22_1', 'AUS.5.145_1', 'AUS.7.7_1', 'AUS.6.17_1', 'AUS.8.61_1', 'AUS.7.71_1', 'AUS.7.43_1', 'AUS.6.4_1', 'AUS.11.28_1', 'AUS.11.46_1', 'AUS.6.2_1', 'AUS.6.14_1', 'AUS.6.15_1', 'AUS.7.19_1', 'AUS.7.17_1', 'AUS.10.46_1', 'AUS.11.95_1', 'AUS.10.19_1', 'AUS.7.4_1', 'AUS.7.25_1', 'AUS.11.61_1', 'AUS.7.16_1', 'AUS.11.40_1', 'AUS.11.55_1', 'AUS.5.20_1', 'AUS.5.138_1', 'AUS.11.47_1', 'AUS.7.69_1', 'AUS.7.54_1', 'AUS.7.42_1', 'AUS.7.2_1', 'AUS.8.4_1', 'AUS.11.22_1', 'AUS.7.21_1', 'AUS.6.10_1', 'AUS.11.137_1', 'AUS.11.109_1', 'AUS.7.34_1', 'AUS.11.133_1', 'AUS.11.3_1', 'AUS.11.72_1', 'AUS.11.77_1', 'AUS.7.6_1', 'AUS.11.88_1', 'AUS.5.31_1', 'AUS.7.9_1', 'AUS.11.121_1', 'AUS.7.39_1', 'AUS.5.130_1', 'AUS.7.24_1', 'AUS.7.3_1', 'AUS.8.31_1', 'AUS.11.13_1', 'AUS.11.138_1', 'AUS.7.52_1', 'AUS.11.71_1', 'AUS.7.14_1', 'AUS.7.46_1', 'AUS.11.44_1', 'AUS.7.20_1', 'AUS.5.6_1']
                   
                #    elif sub_system == 'waste_solid':
                #         if country_code == 'AUS': admin2_lst = ['AUS.6.4_1', 'AUS.11.46_1', 'AUS.8.61_1', 'AUS.6.2_1']
                   
                #    elif sub_system == 'waste_water':
                #         if country_code == 'AUS': admin2_lst = ['AUS.6.14_1', 'AUS.5.6_1', 'AUS.10.46_1', 'AUS.7.3_1', 'AUS.6.7_1', 'AUS.11.138_1', 'AUS.7.6_1', 'AUS.5.145_1', 'AUS.7.2_1', 'AUS.11.72_1', 'AUS.11.139_1', 'AUS.11.28_1', 'AUS.11.109_1', 'AUS.7.24_1', 'AUS.7.12_1', 'AUS.7.43_1', 'AUS.7.25_1', 'AUS.7.69_1', 'AUS.7.39_1', 'AUS.7.46_1', 'AUS.7.47_1', 'AUS.6.17_1', 'AUS.7.19_1', 'AUS.5.20_1', 'AUS.11.22_1', 'AUS.11.88_1', 'AUS.11.13_1', 'AUS.7.52_1', 'AUS.5.34_1', 'AUS.7.71_1', 'AUS.7.4_1', 'AUS.11.47_1', 'AUS.7.34_1', 'AUS.11.121_1', 'AUS.7.14_1', 'AUS.7.63_1', 'AUS.7.7_1', 'AUS.7.16_1', 'AUS.7.42_1', 'AUS.5.31_1', 'AUS.7.5_1', 'AUS.7.17_1', 'AUS.7.54_1', 'AUS.7.9_1', 'AUS.7.20_1', 'AUS.7.22_1', 'AUS.11.3_1', 'AUS.11.137_1', 'AUS.11.75_1', 'AUS.5.130_1', 'AUS.11.44_1', 'AUS.11.61_1', 'AUS.8.4_1', 'AUS.11.40_1', 'AUS.8.31_1', 'AUS.11.55_1', 'AUS.11.77_1', 'AUS.6.15_1', 'AUS.11.95_1', 'AUS.6.12_1', 'AUS.11.133_1', 'AUS.11.71_1', 'AUS.6.10_1', 'AUS.6.4_1', 'AUS.6.2_1', 'AUS.11.46_1', 'AUS.8.61_1']
                   
                #    elif sub_system == 'telecom':
                #         # if country_code == 'RUS': admin2_lst = ['RUS.4.13_1', 'RUS.35.30_1', 'RUS.60.37_1', 'RUS.60.3_1', 'RUS.80.12_1', 'RUS.60.23_1', 'RUS.46.2_1', 'RUS.12.8_1', 'RUS.35.13_1', 'RUS.60.6_1', 'RUS.60.31_1', 'RUS.12.1_1', 'RUS.35.23_1', 'RUS.35.53_1', 'RUS.60.14_1', 'RUS.12.3_1', 'RUS.60.25_1', 'RUS.35.15_1', 'RUS.80.11_1']
                #         if country_code == 'AUS': admin2_lst = ['AUS.7.52_1', 'AUS.6.4_1', 'AUS.7.20_1', 'AUS.11.95_1', 'AUS.10.46_1', 'AUS.11.61_1', 'AUS.5.138_1', 'AUS.11.88_1', 'AUS.7.16_1', 'AUS.11.22_1', 'AUS.5.30_1', 'AUS.11.44_1', 'AUS.11.3_1', 'AUS.6.12_1', 'AUS.7.6_1', 'AUS.8.4_1', 'AUS.7.63_1', 'AUS.5.21_1', 'AUS.9.22_1', 'AUS.5.31_1', 'AUS.7.14_1', 'AUS.7.2_1', 'AUS.7.17_1', 'AUS.11.47_1', 'AUS.11.133_1', 'AUS.5.130_1', 'AUS.11.72_1', 'AUS.11.40_1', 'AUS.11.13_1', 'AUS.6.14_1', 'AUS.6.2_1', 'AUS.5.145_1', 'AUS.7.29_1', 'AUS.7.3_1', 'AUS.10.19_1', 'AUS.7.70_1', 'AUS.5.34_1', 'AUS.11.46_1', 'AUS.7.4_1', 'AUS.11.75_1', 'AUS.8.31_1', 'AUS.7.54_1', 'AUS.11.28_1', 'AUS.11.112_1', 'AUS.7.34_1', 'AUS.11.139_1', 'AUS.11.71_1', 'AUS.8.61_1', 'AUS.11.137_1', 'AUS.7.71_1', 'AUS.7.39_1', 'AUS.7.43_1', 'AUS.11.55_1', 'AUS.7.9_1', 'AUS.11.138_1', 'AUS.5.6_1', 'AUS.7.56_1', 'AUS.7.47_1', 'AUS.7.69_1', 'AUS.7.46_1', 'AUS.6.7_1', 'AUS.11.109_1', 'AUS.7.25_1', 'AUS.5.20_1', 'AUS.6.10_1', 'AUS.7.22_1', 'AUS.11.77_1', 'AUS.7.21_1', 'AUS.7.5_1', 'AUS.7.24_1', 'AUS.6.17_1', 'AUS.7.42_1', 'AUS.7.19_1', 'AUS.6.15_1', 'AUS.7.12_1', 'AUS.7.7_1', 'AUS.11.121_1']
                #    elif sub_system == 'healthcare':
                #         # if country_code == 'RUS': admin2_lst = ['RUS.60.25_1', 'RUS.60.23_1', 'RUS.46.2_1', 'RUS.35.15_1', 'RUS.35.23_1', 'RUS.12.3_1', 'RUS.60.3_1', 'RUS.35.13_1', 'RUS.80.12_1', 'RUS.35.53_1', 'RUS.12.8_1', 'RUS.60.31_1', 'RUS.60.14_1', 'RUS.35.30_1', 'RUS.60.6_1', 'RUS.60.37_1', 'RUS.4.13_1', 'RUS.80.11_1', 'RUS.12.1_1']
                #         if country_code == 'AUS': admin2_lst = ['AUS.6.17_1', 'AUS.6.10_1', 'AUS.7.42_1', 'AUS.7.71_1', 'AUS.11.138_1', 'AUS.8.61_1', 'AUS.7.16_1', 'AUS.8.4_1', 'AUS.7.39_1', 'AUS.7.17_1', 'AUS.7.70_1', 'AUS.7.24_1', 'AUS.7.2_1', 'AUS.5.30_1', 'AUS.7.46_1', 'AUS.11.88_1', 'AUS.11.28_1', 'AUS.10.19_1', 'AUS.5.6_1', 'AUS.5.31_1', 'AUS.11.61_1', 'AUS.7.34_1', 'AUS.5.20_1', 'AUS.7.6_1', 'AUS.7.3_1', 'AUS.11.72_1', 'AUS.11.3_1', 'AUS.7.54_1', 'AUS.7.12_1', 'AUS.6.12_1', 'AUS.7.9_1', 'AUS.11.139_1', 'AUS.11.13_1', 'AUS.9.22_1', 'AUS.11.77_1', 'AUS.11.121_1', 'AUS.7.69_1', 'AUS.11.55_1', 'AUS.8.31_1', 'AUS.7.14_1', 'AUS.5.130_1', 'AUS.7.47_1', 'AUS.7.56_1', 'AUS.7.29_1', 'AUS.11.46_1', 'AUS.11.47_1', 'AUS.7.63_1', 'AUS.11.22_1', 'AUS.5.145_1', 'AUS.7.25_1', 'AUS.11.71_1', 'AUS.7.22_1', 'AUS.11.44_1', 'AUS.11.133_1', 'AUS.7.20_1', 'AUS.7.19_1', 'AUS.6.4_1', 'AUS.5.34_1', 'AUS.5.138_1', 'AUS.7.21_1', 'AUS.11.137_1', 'AUS.7.43_1', 'AUS.11.112_1', 'AUS.11.95_1', 'AUS.7.52_1', 'AUS.6.2_1', 'AUS.7.4_1', 'AUS.11.75_1', 'AUS.6.7_1', 'AUS.11.109_1', 'AUS.7.5_1', 'AUS.5.21_1', 'AUS.11.40_1', 'AUS.6.14_1', 'AUS.6.15_1', 'AUS.7.7_1', 'AUS.10.46_1']
                #    elif sub_system == 'education':
                #         # if country_code == 'RUS': admin2_lst = ['RUS.35.23_1', 'RUS.35.53_1', 'RUS.60.6_1', 'RUS.12.3_1', 'RUS.60.37_1', 'RUS.35.13_1', 'RUS.60.23_1', 'RUS.60.31_1', 'RUS.80.12_1', 'RUS.46.2_1', 'RUS.12.8_1', 'RUS.60.3_1', 'RUS.60.25_1', 'RUS.35.30_1', 'RUS.12.1_1', 'RUS.60.14_1', 'RUS.80.11_1', 'RUS.35.15_1', 'RUS.4.13_1']
                #         if country_code == 'AUS': admin2_lst =  ['AUS.7.39_1', 'AUS.11.22_1', 'AUS.8.31_1', 'AUS.7.22_1', 'AUS.7.24_1', 'AUS.11.75_1', 'AUS.7.17_1', 'AUS.7.71_1', 'AUS.5.6_1', 'AUS.7.19_1', 'AUS.7.12_1', 'AUS.7.5_1', 'AUS.7.52_1', 'AUS.10.19_1', 'AUS.5.31_1', 'AUS.5.30_1', 'AUS.11.55_1', 'AUS.7.47_1', 'AUS.5.145_1', 'AUS.7.25_1', 'AUS.7.42_1', 'AUS.11.88_1', 'AUS.5.21_1', 'AUS.7.16_1', 'AUS.10.46_1', 'AUS.6.12_1', 'AUS.7.21_1', 'AUS.9.22_1', 'AUS.11.133_1', 'AUS.7.7_1', 'AUS.11.28_1', 'AUS.7.63_1', 'AUS.7.56_1', 'AUS.5.138_1', 'AUS.7.70_1', 'AUS.7.14_1', 'AUS.11.47_1', 'AUS.11.138_1', 'AUS.8.61_1', 'AUS.11.3_1', 'AUS.7.43_1', 'AUS.6.14_1', 'AUS.6.15_1', 'AUS.11.72_1', 'AUS.7.69_1', 'AUS.6.10_1', 'AUS.11.121_1', 'AUS.7.34_1', 'AUS.7.20_1', 'AUS.7.2_1', 'AUS.11.40_1', 'AUS.11.139_1', 'AUS.6.7_1', 'AUS.5.130_1', 'AUS.11.13_1', 'AUS.7.29_1', 'AUS.11.61_1', 'AUS.5.34_1', 'AUS.7.46_1', 'AUS.8.4_1', 'AUS.7.9_1', 'AUS.6.2_1', 'AUS.11.109_1', 'AUS.5.20_1', 'AUS.11.46_1', 'AUS.7.6_1', 'AUS.7.3_1', 'AUS.11.112_1', 'AUS.11.137_1', 'AUS.11.95_1', 'AUS.11.71_1', 'AUS.11.44_1', 'AUS.7.54_1', 'AUS.6.17_1', 'AUS.6.4_1', 'AUS.11.77_1', 'AUS.7.4_1']

                infra_type_lst = cis_dict[ci_system][sub_system]
                asset_loc = pathway_dict['output_path'] / 'extracts' / country_code / sub_system
                if asset_loc.exists():
                    assets_country = gpd.read_parquet(asset_loc)
                    assets_country = gpd.GeoDataFrame(assets_country).set_crs(4326)
                    print(f'{country_code} OSM data has already been extracted for {sub_system} and has now been loaded')
                else:
                    print(f'NOTIFICATION: still need to extract data for {country_code} {sub_system}')

                # ROUND 1
                # Submit tasks and collect futures
                with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
                    future_to_admin2 = {executor.submit(risk_assessment_parallel, admin2, gadm_countries, sub_system, country_code, assets_country, hazard_types, pathway_dict, overwrite, flood_data, eq_data, landslide_rf_data, infra_type_lst, data_path, vuln_settings
                                                        ): admin2 for admin2 in admin2_lst}
                # Iterate over futures to check for exceptions or successful completions.
                admin2_remaining = []
                for future in concurrent.futures.as_completed(future_to_admin2):
                    admin2 = future_to_admin2[future]
                    try:
                        if future.result(): print('Run was successful for {}'.format(country_code))  # This will raise an exception if one occurred in the task.
                    except Exception as e:
                        print(f"Task for admin2 '{admin2}' generated an exception: {e}")
                        traceback.print_exc()  # This prints the full traceback to the consol
                        admin2_remaining.append(admin2)

                # ROUND 2
                if admin2_remaining:
                    admin2_remaining_df = gadm_countries[gadm_countries['GID_2'].isin(admin2_remaining)] # Filter gadm_countries to only include the admin2 values in admin2_remaining
                    admin2_remaining = admin2_remaining_df.sort_values('surface_area')['GID_2'].tolist()
                    print(f"Let's try it again for the following admin2 but not parallel: {admin2_remaining}")
                    # Submit tasks and collect futures
                    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                        future_to_admin2_remaining = {executor.submit(risk_assessment_parallel, admin2, gadm_countries, sub_system, country_code, assets_country, hazard_types, pathway_dict, overwrite, flood_data, eq_data, landslide_rf_data, infra_type_lst, data_path, vuln_settings
                                                            ): admin2 for admin2 in admin2_remaining}                        
                    # for admin2 in admin2_lst:
                    #     risk_assessment_parallel(admin2, gadm_countries, sub_system, country_code, assets_country, hazard_types, pathway_dict, overwrite, flood_data, eq_data, landslide_rf_data,infra_type_lst,data_path, vuln_settings)

                    # Iterate over futures to check for exceptions or successful completions.
                    admin2_remaining2 = []
                    for future in concurrent.futures.as_completed(future_to_admin2_remaining):
                        admin2 = future_to_admin2_remaining[future]
                        try:
                            if future.result(): print('Run was successful for {}'.format(country_code))  # This will raise an exception if one occurred in the task.
                        except Exception as e:
                            print(f"Task for admin2 '{admin2}' generated an exception: {e}")
                            admin2_remaining2.append(admin2)
                    if admin2_remaining2:
                        print(f"Please run the remaining admin2's for {sub_system}: {admin2_remaining2}")
                        
    print('Run completed for {}'.format(country_code))


def risk_assessment_parallel(admin2, gadm_countries, sub_system, country_code, assets_country, hazard_types, pathway_dict, overwrite, flood_data, eq_data, landslide_rf_data,infra_type_lst,data_path, vuln_settings):
    """
    Performs parallelized risk assessment of critical infrastructure at the Admin 2 level 
    for high-resolution hazard data.

    This function overlays infrastructure assets with hazard layers (e.g., earthquake, flood, 
    landslide) and applies vulnerability functions to estimate damages. Results are disaggregated 
    by vulnerability curve and hazard return period, and saved as Parquet files at the subsystem level.

    Args:
        admin2 (str): Admin 2 region code.
        gadm_countries (GeoDataFrame): Country-level administrative boundaries.
        sub_system (str): Name of the infrastructure subsystem.
        country_code (str): ISO3 country code.
        assets_country (GeoDataFrame): Infrastructure assets for the country.
        hazard_types (list): List of hazard types (e.g., ['eq', 'flood', 'landslide']).
        pathway_dict (dict): Paths to input and output directories.
        overwrite (bool): Whether to overwrite existing output files.
        flood_data (dict): Flood hazard layers by return period.
        eq_data (dict): Earthquake hazard layers by return period.
        landslide_rf_data (dict): Rainfall-induced landslide hazard data.
        infra_type_lst (list): List of infrastructure types to process.
        data_path (Path): Base data directory.
        vuln_settings (dict): Vulnerability functions and max damage parameters.

    Returns:
        None: Saves damage estimates per subsystem and admin 2 region as Parquet files.
    """
    bbox = gadm_countries.loc[gadm_countries['GID_2'] == admin2].geometry.envelope.values[0].bounds
    country_border_geometries = gadm_countries.loc[gadm_countries['GID_2'] == admin2].geometry # please note that this is just the admin2 boundary and not the whole country

    # more admin2's that can't be handled using v3 data
    lby_additions = ['LBY.6_1']
    can_additions = ['CAN.8.3_1', 'CAN.8.2_1', 'CAN.6.1_1', 'CAN.13.1_1', 'CAN.11.84_1', 'CAN.6.2_1', 'CAN.8.1_1', 'CAN.3.15_1'] 
    aus_additions = ['AUS.10.19_1', 'AUS.10.46_1', 'AUS.11.109_1', 'AUS.11.112_1', 'AUS.11.121_1', 'AUS.11.133_1', 'AUS.11.137_1', 'AUS.11.138_1', 'AUS.11.139_1', 'AUS.11.13_1', 'AUS.11.22_1', 'AUS.11.28_1', 'AUS.11.3_1', 'AUS.11.40_1', 'AUS.11.44_1', 'AUS.11.46_1', 'AUS.11.47_1', 'AUS.11.55_1', 'AUS.11.61_1', 'AUS.11.71_1', 'AUS.11.72_1', 'AUS.11.75_1', 'AUS.11.77_1', 'AUS.11.88_1', 'AUS.11.95_1', 'AUS.5.130_1', 'AUS.5.138_1', 'AUS.5.145_1', 'AUS.5.20_1', 'AUS.5.30_1', 'AUS.5.31_1', 'AUS.5.34_1', 'AUS.5.6_1', 'AUS.6.10_1', 'AUS.6.12_1', 'AUS.6.14_1', 'AUS.6.15_1', 'AUS.6.17_1', 'AUS.6.2_1', 'AUS.6.4_1', 'AUS.6.7_1', 'AUS.7.12_1', 'AUS.7.14_1', 'AUS.7.16_1', 'AUS.7.17_1', 'AUS.7.19_1', 'AUS.7.2_1', 'AUS.7.20_1', 'AUS.7.21_1', 'AUS.7.22_1', 'AUS.7.24_1', 'AUS.7.25_1', 'AUS.7.3_1', 'AUS.7.34_1', 'AUS.7.39_1', 'AUS.7.4_1', 'AUS.7.42_1', 'AUS.7.43_1', 'AUS.7.46_1', 'AUS.7.47_1', 'AUS.7.5_1', 'AUS.7.52_1', 'AUS.7.54_1', 'AUS.7.56_1', 'AUS.7.6_1', 'AUS.7.63_1', 'AUS.7.69_1', 'AUS.7.7_1', 'AUS.7.71_1', 'AUS.8.31_1', 'AUS.8.4_1', 'AUS.8.61_1', 'AUS.9.22_1'] 
    chn_additions = ['CHN.11.1_1', 'CHN.11.2_1', 'CHN.11.3_1', 'CHN.11.4_1', 'CHN.11.5_1', 'CHN.11.6_1', 'CHN.11.7_1', 'CHN.11.9_1', 'CHN.11.10_1', 'CHN.11.11_1', 'CHN.11.12_1', 'CHN.11.13_1', 'CHN.17.1_1', 'CHN.19.6_1', 'CHN.19.12_1', 'CHN.28.2_1', 'CHN.28.4_1', 'CHN.28.8_1', 'CHN.28.12_1']
    usa_additions = ['USA.13.25_1', 'USA.13.37_1', 'USA.20.2_1', 'USA.23.44_1', 'USA.23.45_1', 'USA.23.47_1', 'USA.24.70_1', 'USA.29.10_1', 'USA.29.13_1', 'USA.29.17_1', 'USA.29.3_1', 'USA.29.5_1', 'USA.29.8_1', 'USA.2.1_1', 'USA.2.11_1', 'USA.2.13_1', 'USA.2.14_1', 'USA.2.15_1', 'USA.2.16_1', 'USA.2.17_1', 'USA.2.18_1', 'USA.2.19_1', 'USA.2.2_1', 'USA.2.21_1', 'USA.2.22_1', 'USA.2.23_1', 'USA.2.24_1', 'USA.2.25_1', 'USA.2.26_1', 'USA.2.4_1', 'USA.2.6_1', 'USA.2.7_1', 'USA.2.8_1', 'USA.3.1_1', 'USA.3.10_1', 'USA.3.11_1', 'USA.3.3_1', 'USA.3.8_1', 'USA.3.9_1', 'USA.38.13_1', 'USA.38.19_1', 'USA.38.23_1', 'USA.45.19_1', 'USA.45.23_1', 'USA.5.14_1', 'USA.5.36_1', 'USA.50.34_1', 'USA.51.15_1', 'USA.51.19_1', 'USA.51.4_1', 'USA.51.7_1']
    additions_admin = can_additions + aus_additions + chn_additions + usa_additions + lby_additions

    #largest admin2 areas (RUS and CAN)
    top30_lst = ['CAN.8.1_1', 'CAN.6.2_1', 'RUS.35.30_1', 'CAN.8.3_1', 'CAN.6.1_1', 'RUS.35.13_1', 'RUS.35.23_1', 'CAN.8.2_1', 'CAN.11.84_1', 'CAN.13.1_1', 
                 'RUS.60.23_1', 'RUS.60.6_1', 'RUS.12.1_1', 'RUS.35.15_1', 'CAN.9.20_1', 'RUS.46.2_1', 'RUS.12.3_1', 'RUS.80.11_1', 'RUS.60.31_1', 'CAN.12.9_1', 
                 'RUS.4.13_1', 'RUS.80.12_1', 'RUS.35.53_1', 'CAN.3.15_1', 'RUS.60.3_1', 'RUS.60.37_1', 'RUS.60.14_1', 'RUS.60.25_1', 'RUS.12.8_1', 'RUS.35.55_1'] + additions_admin
 
    #clip assets to admin2
    if not country_border_geometries.empty:  # If ISO_3digit is in shape_countries
        print(f'Time to clip {sub_system} extraction for {country_code}: {admin2}')    
        assets = damage_functions.clip_shapely(assets_country, country_border_geometries) # Clip using the rewritten function
        if assets['osm_id'].is_unique == False: print(f'NOTIFICATION: there are assets with a duplicate osm_id in this dataframe {country_code}, {admin2}, {sub_system}!')
        assets = assets.to_crs(3857)     

    if not assets.empty:
        print(f'assets not empty for {admin2}')
        for hazard_type in hazard_types:
            if not (pathway_dict['output_path'] / 'damage' / country_code / hazard_type).exists() or overwrite == True: #continue with risk assessment if folder does not exist yet or overwrite statement is True
                print(f'overwriting is True, continue with assessment for {admin2}')
                # read hazard data
                hazard_data_path = pathway_dict[hazard_type]
                hazard_data_list = damage_functions.read_hazard_data(hazard_data_path,data_path,hazard_type,country_code,flood_data,eq_data)
                if hazard_type in ['pluvial','fluvial','coastal','windstorm','landslide_eq','landslide_rf', 'windstorm']: hazard_data_list = [file for file in hazard_data_list if file.suffix == '.tif'] #put this code in read hazard data
                
                collect_output = {}
                for single_footprint in hazard_data_list:                        
                    hazard_name = single_footprint.parts[-1].split('.')[0]

                    if not (pathway_dict['output_path'] / 'damage' / country_code / hazard_type / f'{country_code}_{admin2}_{hazard_type}_{hazard_name}_{sub_system}.parquet').exists():
                        print(f'File does not exist yet for {admin2} and {hazard_name}')
                        # load hazard map
                        if hazard_type in ['pluvial','fluvial','coastal']:
                            if flood_data['version'] == 'Fathom_v2':
                                if hazard_type in ['pluvial','fluvial']:
                                    if admin2 in top30_lst: 
                                        print('load hazard map using a gridded approach')
                                        hazard_map = damage_functions.read_flood_map_fathomv2_gridded(single_footprint, bbox, country_border_geometries)
                                    else:
                                        hazard_map = damage_functions.read_flood_map_fathomv2(single_footprint, bbox)
                                elif hazard_type in ['coastal']:
                                    hazard_map = damage_functions.read_flood_map_aqueduct(single_footprint, bbox)
                            
                            elif flood_data['version'] == 'Fathom_v3':
                                if admin2 in top30_lst: 
                                    hazard_map = damage_functions.read_flood_map_fathomv3_gridded(single_footprint, bbox, country_border_geometries)
                                else:
                                    hazard_map = damage_functions.read_flood_map_fathomv3(single_footprint,bbox)
                            hazard_map = damage_functions.overlay_shapely(hazard_map, country_border_geometries)
                        elif hazard_type in ['windstorm']:
                            hazard_map = damage_functions.read_windstorm_map(single_footprint,bbox)
                            hazard_map = damage_functions.overlay_shapely(hazard_map, country_border_geometries)
                        elif hazard_type == 'earthquake': 
                            if eq_data == 'GAR': 
                                hazard_map = damage_functions.read_gar_earthquake_map(single_footprint, bbox) #GAR
                            elif eq_data == 'GIRI': 
                                hazard_map = damage_functions.read_giri_earthquake_map(single_footprint, bbox) #GIRI
                            elif eq_data == 'GEM':
                                hazard_map = damage_functions.read_earthquake_map_csv(single_footprint, bbox) #GEM
                            hazard_map = damage_functions.overlay_shapely(hazard_map, country_border_geometries)                     
                            if not hazard_map.empty:
                                liquefaction_map_path = liquefaction_data_path / 'liquefaction_v1_deg.tif'
                                cond_map = damage_functions.read_liquefaction_map(liquefaction_map_path, bbox) 
                                hazard_map = damage_functions.overlay_dataframes(hazard_map,cond_map) #get hazard polygons that overlay with cond_map
                                if not hazard_map.empty: hazard_map = damage_functions.eq_liquefaction_matrix_liqres(hazard_map,cond_map) #apply liquefaction earthquake matrix and drop hazard points that are irrelevant
                        elif hazard_type == 'earthquake_update': 
                            if eq_data == 'GAR': 
                                hazard_map = damage_functions.read_gar_earthquake_map(single_footprint, bbox) #GAR
                            elif eq_data == 'GIRI': 
                                hazard_map = damage_functions.read_giri_earthquake_map(single_footprint, bbox) #GIRI
                            elif eq_data == 'GEM':
                                hazard_map = damage_functions.read_earthquake_map_csv(single_footprint, bbox) #GEM
                            hazard_map = damage_functions.overlay_shapely(hazard_map, country_border_geometries)                          
                        elif hazard_type in ['landslide_eq', 'landslide_rf']:
                            if hazard_type == 'landslide_eq':
                                if eq_data == 'GAR': 
                                    cond_map = damage_functions.read_gar_earthquake_map(single_footprint, bbox) #GAR
                                elif eq_data == 'GIRI': 
                                    cond_map = damage_functions.read_giri_earthquake_map(single_footprint, bbox) #GIRI
                                elif eq_data == 'GEM':
                                    cond_map = damage_functions.read_earthquake_map_csv(single_footprint, bbox) #GEM
                                cond_map = damage_functions.overlay_shapely(cond_map, country_border_geometries)
                                
                                # Define the thresholds for the classes
                                bins = [0, 0.05, 0.15, 0.25, 0.35, 0.45, float('inf')]  # Adjust the thresholds as needed
                                labels = ['NaN', '1', '2', '3', '4', '5']
                                
                                # Create a new column 'classes' based on the thresholds
                                cond_map['cond_classes'] = pd.cut(cond_map['band_data'], bins=bins, labels=labels, right=False, include_lowest=True)

                                if country_code not in ['KIR', 'FJI', 'UMI', 'NZL', 'CHL', 'JPN', 'ESP', 'IDN', 'BRA', 'CHN', 'AUS',  'GRL', 'USA', 'CAN', 'RUS']:
                                    susc_map = damage_functions.read_susceptibility_map(susceptibility_map_eq_ls, hazard_type, bbox)
                                else:
                                    susc_map = damage_functions.read_susceptibility_map_gridded(susceptibility_map_eq_ls, hazard_type, bbox, country_border_geometries)
                            elif hazard_type == 'landslide_rf':
                                cond_map = damage_functions.read_rainfall_map(single_footprint,bbox)
                                cond_map = damage_functions.overlay_shapely(cond_map, country_border_geometries)
                                
                                # Define the thresholds for the classes
                                bins = [0, 0.3, 2.0, 3.7, 5.0, float('inf')]  # Adjust the thresholds as needed
                                labels = ['1', '2', '3', '4', '5']
                                
                                # Create a new column 'classes' based on the thresholds
                                cond_map['cond_classes'] = pd.cut(cond_map['band_data'], bins=bins, labels=labels, right=False, include_lowest=True)
                                
                                #susceptibility_map_rf_ls = get_future_ls_rf_file(landslide_rf_data, susceptibility_map_rf_ls)
                                if country_code not in ['KIR', 'FJI', 'ATF', 'CHL', 'UMI' , 'MMR', 'NZL','IDN', 'BRA', 'CHN', 'AUS', 'USA', 'GRL', 'CAN', 'RUS']:
                                    susc_map = damage_functions.read_susceptibility_map(susceptibility_map_rf_ls, hazard_type, bbox)
                                else:
                                    susc_map = damage_functions.read_susceptibility_map_gridded(susceptibility_map_rf_ls, hazard_type, bbox, country_border_geometries)
                            
                            susc_map = damage_functions.overlay_shapely(susc_map, country_border_geometries) #overlay with exact administrative border

                        # convert hazard data to epsg 3857
                        if hazard_type in ['landslide_eq', 'landslide_rf']:
                            cond_map = gpd.GeoDataFrame(cond_map).set_crs(4326).to_crs(3857)
                            susc_map = gpd.GeoDataFrame(susc_map).set_crs(4326).to_crs(3857)
                        elif hazard_type in ['pluvial', 'fluvial', 'coastal', 'windstorm', 'earthquake'] :
                            hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                        elif hazard_type in ['earthquake_update']: 
                            if sub_system in ['water_supply', 'waste_solid', 'waste_water', 'telecom', 'healthcare', 'education']:
                                hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                                #hazard_map.to_parquet(hazard_name)
                            elif sub_system in ['road', 'rail']:
                                if not hazard_map.empty:
                                    liquefaction_map_path = liquefaction_data_path / 'liquefaction_v1_deg.tif'
                                    cond_map = damage_functions.read_liquefaction_map(liquefaction_map_path, bbox) 
                                    hazard_map = damage_functions.overlay_dataframes(hazard_map,cond_map) #get hazard polygons that overlay with cond_map
                                    if not hazard_map.empty: hazard_map = damage_functions.eq_liquefaction_matrix_liqres(hazard_map,cond_map) #apply liquefaction earthquake matrix and drop hazard points that are irrelevant
                                hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                            elif sub_system in ['power', 'air']:
                                if not hazard_map.empty:
                                    liquefaction_map_path = liquefaction_data_path / 'liquefaction_v1_deg.tif'
                                    cond_map = damage_functions.read_liquefaction_map(liquefaction_map_path, bbox) 
                                    hazard_eqliq_map = damage_functions.overlay_dataframes(hazard_map,cond_map) #get hazard polygons that overlay with cond_map
                                    if not hazard_eqliq_map.empty: hazard_eqliq_map = damage_functions.eq_liquefaction_matrix_liqres(hazard_eqliq_map,cond_map) #apply liquefaction earthquake matrix and drop hazard points that are irrelevant

                                    # two hazard maps needed, one for linear infra and the other for non-linear infra within sub-system
                                    hazard_eqliq_map = gpd.GeoDataFrame(hazard_eqliq_map).set_crs(4326).to_crs(3857)
                                else:
                                    hazard_eqliq_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)
                                hazard_map = gpd.GeoDataFrame(hazard_map).set_crs(4326).to_crs(3857)                                        
                    
                        # Loop through unique infrastructure types within the subsystem
                        output_dfs = [] # empty list to collect asset level damages for output at the sub-system level
                        trig_rp_lst_complete = [] #empty list to collect hazard trigger return periods for landsldies
                        for infra_type in infra_type_lst: 
                            assets_infra_type = (assets[assets['asset'] == infra_type].copy().reset_index(drop=True))
                        
                            # create dicts for quicker lookup
                            geom_dict = assets_infra_type['geometry'].to_dict()
                    
                            # read vulnerability and maxdam data:
                            infra_curves,maxdams,infra_units = damage_functions.read_vul_maxdam(data_path,hazard_type, infra_type, vuln_settings[0], vuln_settings[1])
                    
                            # start analysis 
                            print(f'{country_code} runs for {infra_type} for {hazard_type} using the {hazard_name} map')
                    
                            if hazard_type in ['landslide_eq', 'landslide_rf']:
                                if not assets_infra_type.empty:
                                    # overlay assets
                                    overlay_assets = pd.DataFrame(damage_functions.overlay_hazard_assets(susc_map,damage_functions.buffer_assets(assets_infra_type)).T,columns=['asset','hazard_point'])
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
                                            overlay_assets = damage_functions.matrix_landslide_eq_susc(overlay_cond, get_susc_data, overlay_assets, susc_point) 
                                        elif hazard_type == 'landslide_rf':
                                            overlay_assets = damage_functions.matrix_landslide_rf_susc(overlay_cond, get_susc_data, overlay_assets, susc_point)
                                    else:
                                        overlay_assets = overlay_assets.drop((overlay_assets[overlay_assets['hazard_point'] == susc_point[0]]).index) #delete susc from overlay_assets
                    
                                #run and output damage calculations for landslides
                                if not assets_infra_type.empty:
                                    if assets_infra_type['geometry'][0].geom_type in ['LineString', 'MultiLineString']:
                                        collect_output,damages_collection_trig_rp, trig_rp_lst = damage_functions.landslide_damage_and_overlay(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type,
                                                                                    maxdams,infra_units,geom_dict,country_code,sub_system,infra_type,collect_output)
                                        trig_rp_lst_complete.append(trig_rp_lst)
                                        output_dfs.append(damages_collection_trig_rp)                                
                                    else:
                                        #collect_output = landslide_damage(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type)
                                        #print('Reminder to check this function')
                                        #collect_output = landslide_damage(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type)
                                        collect_output,damages_collection_trig_rp, trig_rp_lst = damage_functions.landslide_damage_and_overlay(overlay_assets,infra_curves,susc_numpified,assets_infra_type,hazard_type,
                                                                                                                            maxdams,infra_units,geom_dict,country_code,sub_system,infra_type,collect_output)
                                        trig_rp_lst_complete.append(trig_rp_lst)
                                        output_dfs.append(damages_collection_trig_rp)

                            elif hazard_type in ['earthquake', 'earthquake_update', 'pluvial', 'fluvial','coastal','windstorm']: #other hazard
                                if not assets_infra_type.empty:
                                    # overlay assets
                                    if hazard_type == 'earthquake_update' and infra_type in ['cable', 'runway']:
                                        overlay_assets = pd.DataFrame(damage_functions.overlay_hazard_assets(hazard_eqliq_map,damage_functions.buffer_assets(assets_infra_type)).T,columns=['asset','hazard_point'])
                                    else:
                                        overlay_assets = pd.DataFrame(damage_functions.overlay_hazard_assets(hazard_map,damage_functions.buffer_assets(assets_infra_type)).T,columns=['asset','hazard_point'])
                                else: 
                                    overlay_assets = pd.DataFrame(columns=['asset','hazard_point']) #empty dataframe
                        
                                # convert dataframe to numpy array
                                if hazard_type == 'earthquake_update' and infra_type in ['cable', 'runway']:
                                    hazard_numpified = hazard_eqliq_map.to_numpy()
                                else:
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
                                            damage_asset, overlay_asset = damage_functions.get_damage_per_asset_and_overlay(asset,hazard_numpified,asset_geom,hazard_intensity,fragility_values,maxdam,unit_maxdam) #for output at asset level
                                            collect_inb.append(damage_asset) #for excel output
                                            collect_damage_asset[asset[0]] = damage_asset # for output at asset level
                                            collect_overlay_asset[asset[0]] = overlay_asset # for exposure output at asset level
                                    
                                        collect_output[country_code, hazard_name, sub_system, infra_type, infra_curve[0], ((maxdams[maxdams == maxdam]).index)[0]] = np.sum(collect_inb) #, collect_geom # dictionary to store results for various combinations of hazard maps, infrastructure curves, and maximum damage values.
                                        asset_damage = pd.Series(collect_damage_asset)  # for output at asset level
                                        asset_damage.columns = [infra_curve[0]]  # for output at asset level
                                        collect_asset_damages_per_curve.append(asset_damage)  # for output at asset level
                                        asset_exposure = pd.Series(collect_overlay_asset)  # for exposure output at asset level
                                        asset_exposure.columns = 'overlay'  # for exposure output at asset level
                                    curve_ids_list.append(infra_curve[0])  # for output at asset level
                    
                                #if collect_asset_damages_per_curve[0].empty == False: #collect_asset_damages_per_curve.empty == False
                                if any(not series.empty for series in collect_asset_damages_per_curve):
                                    asset_damages_per_curve = pd.concat(collect_asset_damages_per_curve,axis=1)
                                    asset_damages_per_curve.columns = curve_ids_list
                                    asset_damages_per_curve = asset_damages_per_curve.merge(asset_exposure.rename('overlay'), left_index=True, right_index=True) #merge exposure with damages dataframe
                                    asset_damages_per_curve = asset_damages_per_curve.join(assets_infra_type[['osm_id']], how='left')
                                    output_dfs.append(asset_damages_per_curve)

                            #break #delete after testing, otherwise damage will only be assessed for first hazard map

                        if hazard_type in ['landslide_eq', 'landslide_rf']:
                            del susc_map
                            del cond_map
                        else:
                            del hazard_map
                    
                        #create_damage_csv(collect_output, hazard_type, pathway_dict, country_code, sub_system) #with exposure
                        if hazard_type in ['landslide_eq', 'landslide_rf']:
                            # Specify the outer key to filter by
                            trig_rp_lst_complete = [item for sublist in trig_rp_lst_complete for item in sublist]
                            trig_rp_lst_complete = list(set(trig_rp_lst_complete))
                            for trig_rp in trig_rp_lst_complete:
                                # Create a filtered list of dictionaries based on the outer key
                                output_dfs_filtered = [d[trig_rp] for d in output_dfs if trig_rp in d]
                                    
                                if output_dfs_filtered: #if list is not empty
                                    keys_lst = list(output_dfs_filtered[0].keys())
                                    concatenated_dataframes = {key: [] for key in keys_lst}
                                
                                    # Loop through each dictionary in the list
                                    for d in output_dfs_filtered:
                                        for rp, df in d.items():
                                            if rp not in concatenated_dataframes: concatenated_dataframes[rp] = []  # Initialize an empty list for this return period
                                            # Append each sub-DataFrame to the corresponding return period list
                                            concatenated_dataframes[rp].append(df)
                                
                                    # Concatenate DataFrames per return period and output
                                    for rp, dfs in concatenated_dataframes.items():
                                        concatenated_dataframes[rp] = pd.concat(dfs, ignore_index=True)
                                        save_path = pathway_dict['output_path'] / 'damage' / country_code / hazard_type / f'{country_code}_{admin2}_{hazard_type}_ls{rp}_trig{trig_rp}_{sub_system}.parquet'
                                        (save_path.parent).mkdir(parents=True, exist_ok=True)
                                        concatenated_dataframes[rp].to_parquet(save_path, index=False)
                            damage_functions.create_damage_csv_without_exposure(collect_output, hazard_type, pathway_dict, country_code, sub_system) 
                        else:
                            #create parquet with damages at sub_system level
                            if output_dfs:
                                damages_df = pd.concat(output_dfs, ignore_index=True)
                                save_path = pathway_dict['output_path'] / 'damage' / country_code / hazard_type / f'{country_code}_{admin2}_{hazard_type}_{hazard_name}_{sub_system}.parquet'
                                (save_path.parent).mkdir(parents=True, exist_ok=True)
                                damages_df.to_parquet(save_path, index=False)

                                del damages_df

                            #create csv file with damages at national level
                            damage_functions.create_damage_csv_without_exposure(collect_output, hazard_type, pathway_dict, country_code, sub_system) #check whether this line should be moved to the left (i.e., Excel overwriting is the case now??)
        
    print(f'run completed for the following admin2: {admin2}')
    return admin2