"""
Global Multiple-Hazard Critical Infrastructure Risk Analysis

This script contains the code used to generate Excel files with results aggregated at the subnational level. These include infrastructure exposure by infrastructure type, 
hazard, and return period, as well as long-term average exposure (Expected Annual Exposure). The script also calculates Expected Annual Damage by infrastructure sector, including uncertainty ranges.

Author: Sadhana Nirandjan
  
@Author: Sadhana Nirandjan  - Institute for Environmental studies, VU University Amsterdam
"""

import os
import re
import copy
import ast
import warnings
from pathlib import Path
from collections import defaultdict

import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import matplotlib.pyplot as plt
import contextily as cx
import xlsxwriter

from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
########################################################################################################################
################          define paths          #############################################################
########################################################################################################################

project_root = Path("/path/to/gmhcira") # Adjust these paths to match your local setup
output_root = Path("/path/to/gmhcira_outputs") # Adjust these paths to match your local setup

# Input data
admin_path = project_root / "data" / "gadm" / "gadm_410-levels.gpkg"
vuln_path = project_root / "data" / "Vulnerability"

# Intermediate/output data
damage_data_path = output_root / "damage"
extracts_data_path = output_root / "extracts"
output_risk_path = output_root / "risk_with_1in475cutoff" 
summary_risk_path = output_root / "risk_summaries"

# Optional figures directory
figures_path = output_root / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

########################################################################################################################
################         functions          #############################################################
########################################################################################################################

def calculate_risk(road_segment, damages_dict):
    damages_lst = [damages_dict[rp][damages_dict[rp]['osm_id'] == road_segment]['Partial destruction (0.5)'].iloc[0] for rp in [*damages_dict]]
    asset_dam_df = pd.DataFrame([1/rp for rp in [*damages_dict]]+[1,1e-10],damages_lst+[0, max(damages_lst)]).reset_index()
    asset_dam_df.columns = ['damage','prob']
    asset_dam_df = asset_dam_df.sort_values('prob',ascending=True).reset_index(drop=True)
    return np.trapezoid(asset_dam_df.damage.values,asset_dam_df.prob.values) 

def calculate_risk_rp_vectorized(row):
    damages_lst = row.values
    rps = row.index
    if isinstance((row.index)[0], str): rps = [int(s) for con_rp in rps for s in re.findall(r'\d+', con_rp)]
    prob_values = np.array([1/rp for rp in rps] + [1, 1e-10]) 
    damage_values = np.append(damages_lst, [0, max(damages_lst)])
    sorted_indices = np.argsort(prob_values)
    prob_values = prob_values[sorted_indices]
    damage_values = damage_values[sorted_indices]
    return np.trapz(damage_values, prob_values)

def calculate_risk_vectorized(row):
    design_standard = 475 #everything at or below this value will be removed, and willa assume 0 damage at this standard
    damages_lst = row.values
    rps = row.index
    if isinstance((row.index)[0], str): rps = [int(s) for con_rp in rps for s in re.findall(r'\d+', con_rp)]
    maximum_damage = max(damages_lst)

    # Filter the rps and damages_lst based on the design_standard
    filtered_rps_damages = [(rp, damage) for rp, damage in zip(rps, damages_lst) if rp > design_standard]
    if filtered_rps_damages:
        rps_filtered, damages_filtered = zip(*filtered_rps_damages)
    else:
        rps_filtered, damages_filtered = [], []
    rps = list(rps_filtered)
    damages_lst = list(damages_filtered)
    
    #calculate risk
    if design_standard == 0:
        prob_values = np.array([1/rp for rp in rps] + [1, 1e-10]) #design standard of rp x
    else: 
        prob_values = np.array([1/rp for rp in rps] + [1/design_standard, 1e-10]) #design standard of rp x
    if not damages_lst:
        damage_values = np.append(damages_lst, [0, maximum_damage])
    else:
        damage_values = np.append(damages_lst, [0, max(damages_lst)])
    sorted_indices = np.argsort(prob_values)
    prob_values = prob_values[sorted_indices]
    damage_values = damage_values[sorted_indices]
    return np.trapezoid(damage_values, prob_values)

def calculate_eae_vectorized(row, overlay_prefix='overlay_', design_standard=0):
    # Extract all overlay columns dynamically
    overlay_cols = [col for col in row.index if col.startswith(overlay_prefix)]

    # Extract return periods from column names (e.g., 'overlay_10' → 10)
    rps = [int(re.findall(rf'{overlay_prefix}(\d+)', col)[0]) for col in overlay_cols]
    exposure_lst = [row[col] if pd.notna(row[col]) else 0 for col in overlay_cols]

    # Filter the rps and exposure_lst based on the design_standard
    filtered_rps_exposure = [(rp, exposure) for rp, exposure in zip(rps, exposure_lst) if rp > design_standard]
    if filtered_rps_exposure:
        rps_filtered, exposure_filtered = zip(*filtered_rps_exposure)
    else:
        rps_filtered, exposure_filtered = [], []

    rps = list(rps_filtered)
    exposure_lst = list(exposure_filtered)

    # Probability values + tail extension
    if design_standard == 0:
        prob_values = np.array([1/rp for rp in rps] + [1, 1e-10])
    else:
        prob_values = np.array([1/rp for rp in rps] + [1/design_standard, 1e-10])

    exposure_values = np.append(exposure_lst, [0, max(exposure_lst)])

    # Sort for trapezoid rule
    sorted_indices = np.argsort(prob_values)
    prob_values = prob_values[sorted_indices]
    exposure_values = exposure_values[sorted_indices]

    # Compute EAE using trapezoidal integration
    return np.trapezoid(exposure_values, prob_values)

def get_province(road_segment,subnational):
    try:
        column_names = subnational.columns
        if 'GID_3' in column_names:
            return subnational.loc[road_segment.geometry.intersects(subnational.geometry)].GID_3.values[0]
        elif 'GID_2' in column_names:
            return subnational.loc[road_segment.geometry.intersects(subnational.geometry)].GID_2.values[0]
    except:
        return None

def get_model_details(haz_damage_data_path):  
    if 'gem' in str(haz_damage_data_path.name):
        haz_rp_lst = [475, 2475]
        haz_model = 'gem'
    
    elif 'gar' in str(haz_damage_data_path.name):
        haz_rp_lst = [250, 475, 975, 1500, 2475]
        haz_model = 'gar'
    
    elif 'giri' in str(haz_damage_data_path.name) or 'earthquake' in str(haz_damage_data_path.name):
        haz_rp_lst = [250, 475, 975, 1500, 2475]
        haz_model = 'giri'

    elif 'fluvial' in str(haz_damage_data_path.name) or 'pluvial' in str(haz_damage_data_path.name) or 'coastal' in str(haz_damage_data_path.name) :
        haz_rp_lst = [5, 10, 20, 50, 100, 200, 500, 1000]
        haz_model = 'fathom_v3'    

    elif 'windstorm' in str(haz_damage_data_path.name):
        haz_rp_lst = [10, 20, 50, 100, 200, 1000, 2000, 5000, 10000]        
        haz_model = 'STORM'  

    return haz_rp_lst, haz_model

def risk_ead_per_curve(haz_damage_data_path, sub_system, hazard_type, assump_curves, admin2=False):
    haz_rp_lst,haz_model = get_model_details(haz_damage_data_path)
    
    #create df with all unique ID numbers, geometry and column ead
    ead_df = pd.DataFrame(columns=['osm_id']+['ead_{}'.format(curve_id) for curve_id in assump_curves] 
                              +['overlay_{}'.format(rp_trig) for rp_trig in haz_rp_lst])

    #create damages_dictionary containing damages for different return periods for sub system
    damages_dict = {key: pd.DataFrame() for key in haz_rp_lst}

    for rp in haz_rp_lst:
        damage_data_path_list = haz_damage_data_path.iterdir()
        if haz_model == 'gem': 
            data_path = [path for path in damage_data_path_list if '_{}_{}.parquet'.format(rp,sub_system) in str(path)]
        elif haz_model == 'gar':
            data_path = [path for path in damage_data_path_list if 'pga{}_{}.parquet'.format(rp,sub_system) in str(path)]
        elif haz_model == 'giri':
            data_path = [path for path in damage_data_path_list if '_{}y_{}.parquet'.format(rp,sub_system) in str(path)]
        elif haz_model == 'fathom_v3' or haz_model == 'fathom_v2':
            data_path = [path for path in damage_data_path_list if '_1in{}_{}.parquet'.format(rp,sub_system) in str(path)]
        elif haz_model == 'aqueduct':
            data_path = [path for path in damage_data_path_list if '_rp{}_0_{}.parquet'.format(str(rp).zfill(4),sub_system) in str(path)]
        elif haz_model == 'STORM':
            data_path = [path for path in damage_data_path_list if 'STORM_FIXED_RETURN_PERIODS_constant_{}_YR_RP_{}.parquet'.format(rp,sub_system) in str(path)]

        if admin2 != False:
            if len(data_path) != 0:
                data_path = [path for path in data_path if '_{}_'.format(admin2) in str(path)]
            else:
                print('No data for {} {} for admin2 area: {}'.format(sub_system, rp, admin2)) # this is the case if there are no assets of this sub system in an area
        
        if len(data_path) != 0:
            df = pd.read_parquet(data_path[0])
            damages_dict[rp] = pd.concat([damages_dict[rp], df], ignore_index=True)  #create dictionary with the return period 
            for curve_id in assump_curves: damages_dict[rp] = damages_dict[rp].rename(columns={curve_id: f"{curve_id}_{rp}"}) # Rename the curve_id column in the damages_dict[rp] DataFrame
        else:
            print('No data for {} {}'.format(sub_system, rp)) # this is the case if there are no assets of this sub system in an area

    if all(df.empty for df in damages_dict.values()) == True:
        return ead_df

    else:
        print('Time to calculate the EAD')
        #catch cases where no damages occur for high frequency events
        damages_dict =  handle_incomplete_damage_dict(haz_rp_lst, damages_dict, assump_curves)
    
        #calculate EAD
        temp_df = damages_dict[max(haz_rp_lst)][['osm_id']]
        for curve_id in assump_curves:
            temp_df_curve = temp_df[['osm_id']]
            for rp in haz_rp_lst: 
                if f"{curve_id}_{rp}" in damages_dict[rp].columns:
                    temp_df_curve = temp_df_curve.merge(damages_dict[rp][['osm_id', f"{curve_id}_{rp}"]], on='osm_id', how='left') #put all return periods in a single dataframe for curve ID
                else:
                    damages_dict[rp][f"{curve_id}_{rp}"] = pd.NA
                    temp_df_curve = temp_df_curve.merge(damages_dict[rp][['osm_id', f"{curve_id}_{rp}"]], on='osm_id', how='left') #put all return periods in a single dataframe for curve ID
    
            if temp_df_curve['{}_{}'.format(curve_id, max(haz_rp_lst))].isna().any():
                if hazard_type not in ['fluvial', 'pluvial', 'coastal', 'earthquake', 'windstorm']:
                    print('Dataframe contains osm_ids with nans for highest return period. Please check output')
                temp_df_curve = temp_df_curve.dropna(subset=['{}_{}'.format(curve_id, max(haz_rp_lst))])
            
            temp_df_curve.set_index('osm_id', inplace=True)
            temp_df_curve.columns = temp_df_curve.columns.str.replace(f"{curve_id}_", '', regex=False)
            temp_df_curve = temp_df_curve.fillna(0)
            temp_df.loc[:, ['ead_{}'.format(curve_id)]] = temp_df.apply(lambda row: calculate_risk_vectorized(temp_df_curve.loc[row['osm_id']]) if row['osm_id'] in temp_df_curve.index else np.nan, axis=1)
        ead_df = pd.concat([ead_df, temp_df], ignore_index=True)  
    
        #fill in overlay columns
        for rp in haz_rp_lst:
            overlay_dict = damages_dict[rp].set_index('osm_id')['overlay'].to_dict()
            ead_df['overlay_{}'.format(rp)] = ead_df['overlay_{}'.format(rp)].combine_first(ead_df['osm_id'].map(overlay_dict))
            
        return ead_df

def read_vuln_assump(vul_data, hazard_type, sub_system, infra_type_lst, database_id_curves=False):
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
    
    # Load assumptions file containing curve - maxdam combinations per infrastructure type
    if hazard_type in ['pluvial','fluvial','coastal']: 
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
            if sub_system in 'road':
                assump_curves = ['E7.1', 'E7.6', 'E7.7', 'E7.8', 'E7.9', 'E7.10', 'E7.11', 'E7.12', 'E7.13', 'E7.14' ]
            elif sub_system in 'rail':
                assump_curves = ['E8.11', 'E8.16','E8.17','E8.18','E8.19','E8.20','E8.21','E8.22','E8.23','E8.24']        
        elif hazard_type in ['pluvial','fluvial','coastal']: 
            if sub_system in 'road':
                assump_curves = ['F7.4','F7.5','F7.6','F7.7','F7.8','F7.9']
            elif sub_system in 'rail':
                assump_curves = ['F8.1', 'F8.2', 'F8.3', 'F8.4', 'F8.5', 'F8.6', 'F8.7'] 
        elif hazard_type in ['landslide_eq', 'landslide_rf']:
            assump_curves = [None]
        elif hazard_type == 'windstorm':
            if sub_system in ['road', 'rail']:
                assump_curves = ['W7.2']
    else:
        assump_curves = []
        for infra_type in infra_type_lst: 
            #get assumptions from database
            assumptions['Infrastructure type'] = assumptions['Infrastructure type'].str.lower()
            if "_" in infra_type: infra_type = infra_type.replace('_', ' ')
            assump_infra_type = assumptions[assumptions['Infrastructure type'] == infra_type]
            if assump_infra_type['Vulnerability ID number'].item() == 'No ID number, partial destruction is assumed':
                assump_curves_type = [None] #code evt uitbreiden, dat het onderscheid maakt tussen infrastructuur types waar wel/geen id nummer voor is gegeven
            else:    
                assump_curves_type = ast.literal_eval(assump_infra_type['Vulnerability ID number'].item())
            assump_curves.append(assump_curves_type)

        assump_curves = list(set(item for sublist in assump_curves for item in sublist))
    
    return assump_curves

def handle_incomplete_damage_dict(haz_rp_lst, damages_dict, assump_curves):
    """
    Handles cases where damage data for some return periods (RP) in `damages_dict` are incomplete or missing.
    
    Arguments:
        haz_rp_lst: List of hazard return periods (RPs) to process.
        damages_dict: Dictionary where keys are return periods, and values are dataframes containing damage data.
        assump_curves: List of curve identifiers for damage assumptions to process in the data.
    
    Returns:
        Updated `damages_dict` with filled data for missing or incomplete return periods.
    """

    for rp in haz_rp_lst:
        if damages_dict[rp].empty: # Check if the DataFrame for the current return period is empty
            df = copy.deepcopy(damages_dict[max(haz_rp_lst)])
            for curve_id in assump_curves:  # Iterate over the assumed curves to rename columns and set default values
                if f"{curve_id}_{max(haz_rp_lst)}" in df:
                    df = df.rename(columns={f"{curve_id}_{max(haz_rp_lst)}": f"{curve_id}_{rp}"}) # Check if the column for the maximum RP exists in the DataFrame
                    df[f"{curve_id}_{rp}"].values[:] = 0
                elif f"{curve_id}" in df:
                    df[f"{curve_id}"].values[:] = 0
            df['overlay'].values[:] = 0
            damages_dict[rp] = df

    return damages_dict

def get_landslide_return_periods(hazard_type):
    if hazard_type == 'landslide_eq':
        haz_trig_rp_lst = [475]
        landslide_rp_lst = [2.5, 10.0, 20.0, 100.0, 200.0, 1000.0]
    elif hazard_type == 'landslide_rf':
        haz_trig_rp_lst = [5, 25, 200, 1000]
        landslide_rp_lst = [5.0, 7.0, 10.0, 20.0, 33.0, 50.0, 100.0]

    return haz_trig_rp_lst, landslide_rp_lst

def copy_overlay(ead_df, damages_dict, landslide_rp_lst, lowest_non_empty_rp, highest_non_empty_rp, rp_trig):
    rps_in_range = sorted([rp for rp in landslide_rp_lst if lowest_non_empty_rp <= rp <= highest_non_empty_rp]) # filter the relevant rps 
    overlay_dict = {}
    for rp in rps_in_range:
        if not damages_dict[rp].empty:
            current_overlay_dict = damages_dict[rp].set_index('osm_id')['Overlay'].to_dict()
            overlay_dict.update(current_overlay_dict) # Update the overlay_dict with values from current_overlay_dict
    
    ead_df['overlay_{}'.format(rp_trig)] = ead_df['osm_id'].map(overlay_dict)

    return ead_df

def copy_numberoflandslides(ead_df, damages_dict, landslide_rp_lst, lowest_non_empty_rp, highest_non_empty_rp, rp_trig):
    rps_in_range = sorted([rp for rp in landslide_rp_lst if lowest_non_empty_rp <= rp <= highest_non_empty_rp]) # filter the relevant rps 
    overlay_dict = {}
    for rp in rps_in_range:
        if not damages_dict[rp].empty:
            current_overlay_dict = damages_dict[rp].set_index('osm_id')['number of landslides'].to_dict()
            overlay_dict.update(current_overlay_dict) # Update the overlay_dict with values from current_overlay_dict
    
    ead_df['number_landslides_{}'.format(rp_trig)] = ead_df['osm_id'].map(overlay_dict)

    return ead_df

def handle_zero_values(ead_df, haz_trig_rp_lst):
    """
    Replaces zero values in higher return periods with the value of the previous lower return period,
    if the lower return period has a non-zero value.

    Parameters:
    ead_df (pd.DataFrame): DataFrame containing the 'ead' columns.
    haz_trig_rp_lst (list): List of return periods corresponding to the 'ead' columns.

    Returns:
    pd.DataFrame: Updated DataFrame with zero values replaced.
    """

    # perform for risks
    values = ead_df[['ead_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst]].fillna(0).values # Extract the relevant columns and convert to numpy array for vectorized operations
    # Iterate over the columns, starting from the second column
    for i in range(1, values.shape[1]):
        # Replace zeros with the previous column's value
        values[:, i] = np.where(values[:, i] == 0, values[:, i-1], values[:, i])
    ead_df[['ead_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst]] = values # Convert the numpy array back to a DataFrame and update the original DataFrame

    # perform for overlay
    values = ead_df[['overlay_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst]].fillna(0).values # Extract the relevant columns and convert to numpy array for vectorized operations
    # Iterate over the columns, starting from the second column
    for i in range(1, values.shape[1]):
        # Replace zeros with the previous column's value
        values[:, i] = np.where(values[:, i] == 0, values[:, i-1], values[:, i])
    ead_df[['overlay_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst]] = values # Convert the numpy array back to a DataFrame and update the original DataFrame

    # perform for landslides
    values = ead_df[['number_landslides_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst]].fillna(0).values # Extract the relevant columns and convert to numpy array for vectorized operations
    # Iterate over the columns, starting from the second column
    for i in range(1, values.shape[1]):
        # Replace zeros with the previous column's value
        values[:, i] = np.where(values[:, i] == 0, values[:, i-1], values[:, i])
    ead_df[['number_landslides_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst]] = values  # Convert the numpy array back to a DataFrame and update the original DataFrame
    
    return ead_df

def ls_risk_ead_per_curve(haz_damage_data_path, sub_system, hazard_type, assump_curves, admin2=False):
    
    haz_trig_rp_lst, landslide_rp_lst = get_landslide_return_periods(hazard_type)
    
    #create empty df 
    ead_df = pd.DataFrame(columns=['osm_id']+['ead_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst] 
                              +['overlay_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst] +['number_landslides_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst])

    #read damage data into dictionary per hazard trigger return period 
    for rp_trig in haz_trig_rp_lst:
        damage_data_path_list = haz_damage_data_path.iterdir()
        rp_trig_path_list = [path for path in damage_data_path_list if '_trig{}.0_{}'.format(rp_trig, sub_system) in str(path)]
        if admin2 != False:
            if len(rp_trig_path_list) != 0:
                rp_trig_path_list = [path for path in rp_trig_path_list if '_{}_'.format(admin2) in str(path)]
        damages_dict = {key: pd.DataFrame() for key in landslide_rp_lst}
        for data_path in rp_trig_path_list:
            df = pd.read_parquet(data_path)
            damages_dict[df['return_period_landslide'].unique()[0]] = pd.concat([damages_dict[df['return_period_landslide'].unique()[0]], df], ignore_index=True) #create dictionary with the return period 
        
        non_empty_rps = [key for key, df in damages_dict.items() if not df.empty]
        if non_empty_rps:
            #modify dictionaries 
            lowest_non_empty_rp = min(non_empty_rps)
            damages_dict = {key: df for key, df in damages_dict.items() if key >= lowest_non_empty_rp} # Step 1: Remove all keys above this return period
            highest_non_empty_rp = max(non_empty_rps)
            damages_dict = {key: df for key, df in damages_dict.items() if key <= highest_non_empty_rp} # Step 2: Remove all keys below this return period
            
            # Calculate EAD per OSM road segment for rainfall event
            combined_df = pd.concat([damages_dict[rp] for rp in damages_dict.keys()]) #merge dataframes into one
            ead_df = pd.merge(ead_df, damages_dict[highest_non_empty_rp][['osm_id']], on=['osm_id'], how='outer')
    
            pivoted_damages = combined_df.pivot_table(index='osm_id', columns='return_period_landslide', values='Partial destruction (0.5)', fill_value=0)        
            ead_df['ead_{}'.format(rp_trig)] = ead_df.apply(lambda row: calculate_risk_rp_vectorized(pivoted_damages.loc[row['osm_id']]) if row['osm_id'] in pivoted_damages.index else 0, axis=1)
    
            #fill in overlay columns and number of landslides column
            ead_df = copy_overlay(ead_df, damages_dict, landslide_rp_lst, lowest_non_empty_rp, highest_non_empty_rp, rp_trig) # will take the highest overlay for hazard trigger return period
            ead_df = copy_numberoflandslides(ead_df, damages_dict, landslide_rp_lst, lowest_non_empty_rp, highest_non_empty_rp, rp_trig) # will take the highest number of landslides for hazard trigger return period

    if not ead_df.empty:
        # Calculate EAD for landslides
        ead_df = handle_zero_values(ead_df, haz_trig_rp_lst) # Handle 0 values for higher return periods with damages for the lower return periods
        temp_df = (ead_df.filter(['osm_id']+['ead_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst], axis=1)).set_index('osm_id') #create df with only ead columns
        ead_df['ead'] = ead_df.apply(lambda row: calculate_risk_vectorized(temp_df.loc[row['osm_id']]), axis=1)
    else:
        ead_df['ead'] = 0
        
    return ead_df

def get_return_periods(hazard_type):  
    if hazard_type == 'earthquake':
        haz_rp_lst = [250, 475, 975, 1500, 2475]

    elif hazard_type in ['pluvial','fluvial','coastal']:
        haz_rp_lst = [5, 10, 20, 50, 100, 200, 500, 1000]

    elif hazard_type in ['windstorm']:
        haz_rp_lst = [10, 20, 50, 100, 200, 1000, 2000, 5000, 10000]  
    
    elif hazard_type in ['landslide_rf']:
        haz_rp_lst = [5, 25, 200, 1000]

    elif hazard_type in ['landslide_eq']:
        haz_rp_lst = [475]

    return haz_rp_lst

def convert_value(x):
    # If x is a numpy array, convert it to a list.
    if isinstance(x, np.ndarray):
        return x.tolist()
    # If x is of NAType (pandas._libs.missing.NAType), return an empty list.
    if type(x).__name__ == 'NAType':
        return []
    return x

def partone_postprocessing_admin2(cis_dict, vuln_path, hazard_type, damage_data_path, country, subnational_df, admin2_lst):
    for ci_system in cis_dict: 
        for sub_system in cis_dict[ci_system]:
            try: 
                ead_df_list = []
                infra_type_lst = cis_dict[ci_system][sub_system]
                assump_curves = read_vuln_assump(vuln_path, hazard_type, sub_system, infra_type_lst, database_id_curves=True)              
                haz_damage_data_path = damage_data_path / country / hazard_type # adjust this to your folder
                
                if haz_damage_data_path.exists():
                    for admin2 in admin2_lst:
                        print(admin2)
                        try:
                            #calculate EAD 
                            if hazard_type in ['fluvial', 'pluvial', 'coastal', 'earthquake', 'windstorm']:
                                ead_df_admin2 = risk_ead_per_curve(haz_damage_data_path, sub_system, hazard_type, assump_curves, admin2)
                            elif hazard_type in ['landslide_eq', 'landslide_rf']: 
                                ead_df_admin2 = ls_risk_ead_per_curve(haz_damage_data_path, sub_system, hazard_type, assump_curves, admin2)

                            if hazard_type in ['fluvial', 'pluvial', 'coastal', 'earthquake', 'windstorm']:
                                curve_names = ['ead_{}'.format(curve) for curve in assump_curves]
                            elif hazard_type in ['landslide_eq', 'landslide_rf']:
                                curve_names = ['ead']
                            for curve_name in curve_names:
                                ead_df_admin2['{}_lower'.format(curve_name)] = ead_df_admin2[curve_name] * 0.75
                                ead_df_admin2['{}_upper'.format(curve_name)] = ead_df_admin2[curve_name] * 1.25
                            
                            if hazard_type in ['fluvial', 'pluvial', 'coastal', 'earthquake', 'windstorm']:
                                curve_names = curve_names + ['ead_{}_lower'.format(curve) for curve in assump_curves] + ['ead_{}_upper'.format(curve) for curve in assump_curves]
                            elif hazard_type in ['landslide_eq', 'landslide_rf']:
                                curve_names = curve_names + ['ead_lower'] + ['ead_upper']
                            
                            ead_df_admin2['ead_min'] = ead_df_admin2[curve_names].min(axis=1) # Calculate the min across the specified columns
                            ead_df_admin2['ead_Q1'] = ead_df_admin2[curve_names].quantile(0.25, axis=1) # Calculate lower quartile across the specified columns
                            ead_df_admin2['ead_Q2'] = ead_df_admin2[curve_names].median(axis=1) # Calculate the median across the specified columns
                            ead_df_admin2['ead_Q3'] = ead_df_admin2[curve_names].quantile(0.75, axis=1) # Calculate the across the specified columns
                            ead_df_admin2['ead_max'] = ead_df_admin2[curve_names].max(axis=1) # Calculate the max across the specified columns
                            ead_df_admin2 = ead_df_admin2.drop(columns=curve_names) #drop columns

                            #Calculate the EAE
                            ead_df_admin2['eae'] = ead_df_admin2.apply(calculate_eae_vectorized, axis=1)
            
                            # Export the GeoDataFrame with geomtries and asset types
                            geom_df = gpd.read_parquet(extracts_data_path / country / '{}'.format(sub_system))
                            ead_df_admin2 = ead_df_admin2.merge(geom_df[['osm_id', 'asset', 'geometry']], on='osm_id', how='left')
                            
                            output_file_path = output_risk_path / country / hazard_type / 'assetlevel_{}_{}_{}_{}.parquet'.format(country, admin2, hazard_type, sub_system)

                            
                            (output_file_path.parent).mkdir(parents=True, exist_ok=True)
                            ead_df_admin2 = gpd.GeoDataFrame(ead_df_admin2).set_crs(4326)
                            ead_df_admin2.to_parquet(output_file_path)

                            #add admin2 level column and add admin2 level df to list
                            ead_df_admin2 = gpd.GeoDataFrame(ead_df_admin2).to_crs(3857)
                            iso_subnational_df = subnational_df[subnational_df['GID_0'] == country]
                            ead_df_admin2['GID_2'] = admin2
                            ead_df_list.append(ead_df_admin2)

                        except Exception as error:
                            print(f"An error occurred for {country} {hazard_type} {ci_system} - {admin2}:", error) # An error occurred: name 'x' is not defined  

                    #concat admin2 level dfs before further postprocessing
                    ead_df = pd.concat(ead_df_list, ignore_index=True)

                    #add overlay and eae per infra type
                    selected_columns = [col for col in ead_df.columns if col.startswith('overlay_')] 
                    for infra_type in infra_type_lst:
                        ead_df_infra_type = ead_df[ead_df['asset'] == infra_type]

                        if not ead_df_infra_type.empty:
                            #output overlay per infra_type
                            adjusted_columns = [f'{infra_type}_{col}' for col in selected_columns]
                            ead_df_infra_type.columns = [f'{infra_type}_{col}' if col.startswith('overlay_') else col for col in ead_df_infra_type.columns]
                            iso_subnational_df = iso_subnational_df.merge(ead_df_infra_type[['GID_2'] + adjusted_columns].groupby('GID_2').sum(),left_on='GID_2',right_index=True, how='left').reset_index(drop=True)

                            #output a column with list of osm_ids per infra type
                            infratype_osm_ids_gid2 = ead_df_infra_type.groupby('GID_2')['osm_id'].apply(list).to_dict()
                            osm_ids_df = pd.DataFrame(list(infratype_osm_ids_gid2 .items()), columns=['GID_2', 'osm_ids'])
                            osm_ids_df.rename(columns={'osm_ids': '{}_osm_ids'.format(infra_type)}, inplace=True)
                            iso_subnational_df = iso_subnational_df.merge(osm_ids_df, on='GID_2', how='left')
                        
                            # output columns with count of exposed infra type per rp
                            infra_extension = next((u for u in unit_dict[ci_system][sub_system] if u.startswith(infra_type)), None)
                            if '_m2' in infra_extension:
                                # print('also add count for {}'.format(infra_type))
                                overlay_cols = [col for col in ead_df_infra_type.columns if '_overlay_' in col] # Create a list of column names that have '_overlay_' in them
                                for col in overlay_cols:
                                    ead_df_infra_type_count = ead_df_infra_type[[col, 'GID_2']][ead_df_infra_type[col].notna()]
                                    if not ead_df_infra_type_count.empty:
                                        #output a column with list of osm_ids per infra type
                                        infratype_counts_gid2 = ead_df_infra_type_count['GID_2'].value_counts().to_dict()
                                        counts_df = pd.DataFrame(list(infratype_counts_gid2.items()), columns=['GID_2', f'{col}_count'])
                                        #counts_df.rename(columns={'osm_ids': '{}_osm_ids'.format(infra_type)}, inplace=True)
                                        iso_subnational_df = iso_subnational_df.merge(counts_df, on='GID_2', how='left')
                                    else:
                                        iso_subnational_df[f'{col}_count'] = np.nan
                                        
                            # #output eae per infra_type
                            ead_df_infra_type = ead_df_infra_type.rename(columns={'eae': f'eae_{infra_type}'})
                            #iso_subnational_df = iso_subnational_df.merge(ead_df_infra_type[['GID_2',f'eae_{infra_type}']].groupby('GID_2').sum(),left_on='GID_2',right_index=True)
                            iso_subnational_df = iso_subnational_df.merge(ead_df_infra_type[['GID_2', f'eae_{infra_type}']].groupby('GID_2').sum(), left_on='GID_2',right_index=True,how='left')    

                            #also add eae_count for polygons
                            has_m2 = any(f"{infra_type}_m2" in units for subsystems in unit_dict.values() for units in subsystems.values())
                            if has_m2:
                                # print(f'also add eae_count for {infra_type}')
                                pattern_infra = re.compile(rf'^{infra_type}_overlay_(\d+)_count$')

                                infra_cols = [c for c in iso_subnational_df.columns if pattern_infra.search(c)]
                                rp_to_cols_infra = defaultdict(list)
                                for c in infra_cols:
                                    rp = int(pattern_infra.search(c).group(1))
                                    rp_to_cols_infra[rp].append(c)
                                
                                temp_infra = pd.DataFrame(
                                    {f'overlay_{rp}': iso_subnational_df[cols].sum(axis=1)
                                     for rp, cols in sorted(rp_to_cols_infra.items())},
                                    index=iso_subnational_df.index
                                )
                                
                                iso_subnational_df[f'eae_{infra_type}_count'] = temp_infra.apply(calculate_eae_vectorized, axis=1)
                    
                    # add ead per ci_system
                    selected_columns = [col for col in ead_df.columns if col.startswith('ead_') or col.startswith('number_landslides')]
                    iso_subnational_df = iso_subnational_df.merge(ead_df[['GID_2'] + selected_columns].groupby('GID_2').sum(),left_on='GID_2',right_index=True).reset_index(drop=True)

                    #adjust dataset
                    iso_subnational_df = iso_subnational_df.drop(columns = ['NL_NAME_1', 'VARNAME_2', 'NL_NAME_2', 'TYPE_2', 'ENGTYPE_2', 'CC_2', 'HASC_2']) # drop irrelevant columns
                    if hazard_type in ['landslide_eq', 'landslide_rf']:
                        haz_trig_rp_lst =  get_landslide_return_periods(hazard_type)[0]
                        iso_subnational_df = iso_subnational_df.drop(columns = ['ead_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst])
                    
                    #output dataset
                    gdf = gpd.GeoDataFrame(iso_subnational_df, geometry='geometry')
                    gdf = gdf.set_crs(3857, inplace=True)           
                    output_file_path = output_risk_path / country / hazard_type / '{}_{}_{}'.format(country, hazard_type, sub_system) # Define the output file path
                    (output_file_path.parent).mkdir(parents=True, exist_ok=True)
                    gdf.to_parquet(output_file_path)  # Export the GeoDataFrame to a shapefile
                    # all sub systems in one gdf instead of multiple > also add subsystem to column
                    print("Calculation of risk completed for {}, {} - {}".format(country, sub_system, hazard_type))            

            except Exception as error:
                print(f"An error occurred for {country} {hazard_type} {ci_system}:", error) # An error occurred: name 'x' is not defined  

########################################################################################################################
################          Settings        #############################################################
########################################################################################################################

glob_info = gpd.read_parquet(Path('/scistor/ivm/') / 'snn490'/ 'Projects' / 'gmhcira' / 'data' / 'gadm' / 'gadm_410_simplified_admin0_income') 
glob_info = glob_info.sort_values(by='area_km2', ascending=True)
iso_lst =  list(glob_info.GID_0)

hazard_types = ['fluvial', 'pluvial', 'coastal', 'earthquake', 'landslide_rf', 'landslide_eq', 'windstorm']

cis_dict = {
    "energy": {"power": ["transmission_line","distribution_line","cable","plant","substation",
                        "power_tower","power_pole"]},
    "transportation": {"road":  ["motorway", "trunk", "primary", "secondary", "tertiary", "track", "road", "residential"],
                        "air": ["airport", "runway", "terminal"],
                        "rail": ["railway"]
                        },
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


unit_dict = {
    "energy": {"power": ["transmission_line_km","distribution_line_km","cable_km","plant_m2","substation_m2",
                        "power_tower_count","power_pole_count"]},
    "transportation": {"road":  ["motorway_km", "trunk_km", "primary_km", "secondary_km", "tertiary_km", "track_km", "road_km", "residential_km"],
                        "air": ["airport_m2", "runway_km", "terminal_m2"],
                        "rail": ["railway_km"]
                        },
    "water": {"water_supply": ["water_tower_count", "water_well_count", "reservoir_covered_m2",
                                "water_treatment_plant_m2", "water_storage_tank_count"]},
    "waste": {"waste_solid": ["waste_transfer_station_m2"],
            "waste_water": ["wastewater_treatment_plant_m2"]},
    "telecommunication": {"telecom": ["communication_tower_count", "mast_count"]},
    "healthcare": {"healthcare": ["clinic_m2", "doctors_m2", "hospital_m2", "dentist_m2", "pharmacy_m2", 
                        "physiotherapist_m2", "alternative_m2", "laboratory_m2", "optometrist_m2", "rehabilitation_m2", 
                        "blood_donation_m2", "birthing_center_m2"]},
    "education": {"education": ["college_m2", "kindergarten_m2", "library_m2", "school_m2", "university_m2"]}
    }

########################################################################################################################
################          Step 1: Calculate EADs per curve and aggegrate to admin 2 level        #############################################################
########################################################################################################################

subnational_df = gpd.read_parquet(project_root / 'data' / 'gadm' / 'gadm_410_simplified_admin0_income') #GADM countries
subnational_df = subnational_df.to_crs(3857)
print(iso_lst)
print("Protection standards used in this run: 1-in-475 years for earthquakes and 1-in-10 years for all other hazards")

for hazard_type in hazard_types:
    for country in iso_lst: 
        print('Time for postprocessing of {}'.format(country))
        if hazard_type in ['coastal'] and country in ['AUS', 'BRA', 'CHL', 'CHN', 'FJI', 'GRL', 'IDN', 'NZL'] + ['CAN', 'USA', 'RUS']: 
            admin2_lst = subnational_df[subnational_df['GID_0'] == country]['GID_2'].tolist()
            partone_postprocessing_admin2(cis_dict, vuln_path, hazard_type, damage_data_path, country, subnational_df, admin2_lst)
            continue
        elif hazard_type in ['pluvial'] and country in ['AFG', 'ARG', 'BOL', 'BRA', 'BWA', 'CHL', 'COD', 'COG', 'DZA', 'EGY', 'ETH', 'FIN', 'FJI', 'GBR', 'IDN', 'IRN', 'IRQ', 'KAZ', 'MDG', 'MEX', 'MRT', 'NER', 'NGA', 'NZL', 'PAK', 'PRY', 'SDN', 'SOM', 'SSD', 'SWE', 'TCD', 'THA', 'TZA', 'UZB', 'ZMB'] + ['SAU', 'AGO', 'MLI', 'LBY', 'CHN', 'AUS', 'USA', 'CAN', 'RUS', 'IND', 'MNG', 'ZAF', 'VEN', 'UKR', 'TUR', 'TKM', 'PER', 'NAM', 'COL', 'MMR', 'MAR', 'KEN', 'CMR', 'MOZ']: 
            admin2_lst = subnational_df[subnational_df['GID_0'] == country]['GID_2'].tolist()
            partone_postprocessing_admin2(cis_dict, vuln_path, hazard_type, damage_data_path, country, subnational_df, admin2_lst)
            continue
        elif hazard_type in ['fluvial'] and country in ['ARG', 'AUS', 'BRA', 'CHL', 'CHN', 'COD', 'IDN', 'IND', 'KAZ', 'MLI', 'MRT', 'NZL', 'SDN'] + ['CAN', 'USA', 'RUS']: 
            admin2_lst = subnational_df[subnational_df['GID_0'] == country]['GID_2'].tolist()
            partone_postprocessing_admin2(cis_dict, vuln_path, hazard_type, damage_data_path, country, subnational_df, admin2_lst)
            continue
        elif hazard_type in ['landslide_rf'] and country in ['CAN', 'CHN', 'IND', 'RUS', 'USA']:
            admin2_lst = subnational_df[subnational_df['GID_0'] == country]['GID_2'].tolist()
            partone_postprocessing_admin2(cis_dict, vuln_path, hazard_type, damage_data_path, country, subnational_df, admin2_lst)
            continue
        elif hazard_type in ['landslide_eq'] and country in ['CAN', 'CHN', 'RUS', 'TUR', 'USA']:
            admin2_lst = subnational_df[subnational_df['GID_0'] == country]['GID_2'].tolist()
            partone_postprocessing_admin2(cis_dict, vuln_path, hazard_type, damage_data_path, country, subnational_df, admin2_lst)
            continue
        elif hazard_type in ['earthquake'] and country in ['USA', 'CAN', 'RUS']:
            admin2_lst = subnational_df[subnational_df['GID_0'] == country]['GID_2'].tolist()
            partone_postprocessing_admin2(cis_dict, vuln_path, hazard_type, damage_data_path, country, subnational_df, admin2_lst)
            continue

        for ci_system in cis_dict: 
            for sub_system in cis_dict[ci_system]:
                try: 
                    infra_type_lst = cis_dict[ci_system][sub_system]
                    assump_curves = read_vuln_assump(vuln_path, hazard_type, sub_system, infra_type_lst, database_id_curves=True)
                    haz_damage_data_path = damage_data_path / country / hazard_type 

                    if haz_damage_data_path.exists():
                        #calculate EAD 
                        if hazard_type in ['fluvial', 'pluvial', 'coastal', 'earthquake', 'windstorm']:
                            ead_df = risk_ead_per_curve(haz_damage_data_path, sub_system, hazard_type, assump_curves)
                        elif hazard_type in ['landslide_eq', 'landslide_rf']: 
                            ead_df = ls_risk_ead_per_curve(haz_damage_data_path, sub_system, hazard_type, assump_curves)

                        if hazard_type in ['fluvial', 'pluvial', 'coastal', 'earthquake', 'windstorm']:
                            curve_names = ['ead_{}'.format(curve) for curve in assump_curves]
                        elif hazard_type in ['landslide_eq', 'landslide_rf']:
                            curve_names = ['ead']
                        for curve_name in curve_names:
                            ead_df['{}_lower'.format(curve_name)] = ead_df[curve_name] * 0.75
                            ead_df['{}_upper'.format(curve_name)] = ead_df[curve_name] * 1.25
                        
                        if hazard_type in ['fluvial', 'pluvial', 'coastal', 'earthquake', 'windstorm']:
                            curve_names = curve_names + ['ead_{}_lower'.format(curve) for curve in assump_curves] + ['ead_{}_upper'.format(curve) for curve in assump_curves]
                        elif hazard_type in ['landslide_eq', 'landslide_rf']:
                            curve_names = curve_names + ['ead_lower'] + ['ead_upper']
                        
                        ead_df['ead_min'] = ead_df[curve_names].min(axis=1) # Calculate the min across the specified columns
                        ead_df['ead_Q1'] = ead_df[curve_names].quantile(0.25, axis=1) # Calculate lower quartile across the specified columns
                        ead_df['ead_Q2'] = ead_df[curve_names].median(axis=1) # Calculate the median across the specified columns
                        ead_df['ead_Q3'] = ead_df[curve_names].quantile(0.75, axis=1) # Calculate the across the specified columns
                        ead_df['ead_max'] = ead_df[curve_names].max(axis=1) # Calculate the max across the specified columns
                        ead_df = ead_df.drop(columns=curve_names) #drop columns

                        #Calculate the EAE
                        ead_df['eae'] = ead_df.apply(calculate_eae_vectorized, axis=1)
        
                        # Export the GeoDataFrame with geomtries and asset types
                        geom_df = gpd.read_parquet(extracts_data_path / country / '{}'.format(sub_system))
                        ead_df = ead_df.merge(geom_df[['osm_id', 'asset', 'geometry']], on='osm_id', how='left')
                        
                        output_file_path = output_risk_path / country / hazard_type / 'assetlevel_{}_{}_{}'.format(country, hazard_type, sub_system)

                        (output_file_path.parent).mkdir(parents=True, exist_ok=True)
                        ead_df = gpd.GeoDataFrame(ead_df).set_crs(4326)
                        ead_df.to_parquet(output_file_path)
                        
                        #aggregate at admin 2 level
                        ead_df = gpd.GeoDataFrame(ead_df).to_crs(3857)
                        iso_subnational_df = subnational_df[subnational_df['GID_0'] == country]
                        ead_df['GID_2'] = ead_df.apply(lambda road_segment: get_province(road_segment, iso_subnational_df), axis=1)

                        #add overlay and eae per infra type
                        selected_columns = [col for col in ead_df.columns if col.startswith('overlay_')] 
                        for infra_type in infra_type_lst:
                            ead_df_infra_type = ead_df[ead_df['asset'] == infra_type]

                            if not ead_df_infra_type.empty:
                                #output overlay per infra_type
                                adjusted_columns = [f'{infra_type}_{col}' for col in selected_columns]
                                ead_df_infra_type.columns = [f'{infra_type}_{col}' if col.startswith('overlay_') else col for col in ead_df_infra_type.columns]
                                iso_subnational_df = iso_subnational_df.merge(ead_df_infra_type[['GID_2'] + adjusted_columns].groupby('GID_2').sum(),left_on='GID_2',right_index=True, how='left').reset_index(drop=True)

                                #output a column with list of osm_ids per infra type
                                infratype_osm_ids_gid2 = ead_df_infra_type.groupby('GID_2')['osm_id'].apply(list).to_dict()
                                osm_ids_df = pd.DataFrame(list(infratype_osm_ids_gid2 .items()), columns=['GID_2', 'osm_ids'])
                                osm_ids_df.rename(columns={'osm_ids': '{}_osm_ids'.format(infra_type)}, inplace=True)
                                iso_subnational_df = iso_subnational_df.merge(osm_ids_df, on='GID_2', how='left')
                            
                                # output columns with count of exposed infra type per rp
                                infra_extension = next((u for u in unit_dict[ci_system][sub_system] if u.startswith(infra_type)), None)
                                if '_m2' in infra_extension:
                                    # print('also add count for {}'.format(infra_type))
                                    overlay_cols = [col for col in ead_df_infra_type.columns if '_overlay_' in col] # Create a list of column names that have '_overlay_' in them
                                    for col in overlay_cols:
                                        ead_df_infra_type_count = ead_df_infra_type[[col, 'GID_2']][ead_df_infra_type[col].notna()]
                                        if not ead_df_infra_type_count.empty:
                                            #output a column with list of osm_ids per infra type
                                            infratype_counts_gid2 = ead_df_infra_type_count['GID_2'].value_counts().to_dict()
                                            counts_df = pd.DataFrame(list(infratype_counts_gid2.items()), columns=['GID_2', f'{col}_count'])
                                            #counts_df.rename(columns={'osm_ids': '{}_osm_ids'.format(infra_type)}, inplace=True)
                                            iso_subnational_df = iso_subnational_df.merge(counts_df, on='GID_2', how='left')
                                        else:
                                            iso_subnational_df[f'{col}_count'] = np.nan
                                            
                                # #output eae per infra_type
                                ead_df_infra_type = ead_df_infra_type.rename(columns={'eae': f'eae_{infra_type}'})
                                iso_subnational_df = iso_subnational_df.merge(ead_df_infra_type[['GID_2', f'eae_{infra_type}']].groupby('GID_2').sum(), left_on='GID_2',right_index=True,how='left')    

                                #also add eae_count for polygons
                                has_m2 = any(f"{infra_type}_m2" in units for subsystems in unit_dict.values() for units in subsystems.values())
                                if has_m2:
                                    pattern_infra = re.compile(rf'^{infra_type}_overlay_(\d+)_count$')
    
                                    infra_cols = [c for c in iso_subnational_df.columns if pattern_infra.search(c)]
                                    rp_to_cols_infra = defaultdict(list)
                                    for c in infra_cols:
                                        rp = int(pattern_infra.search(c).group(1))
                                        rp_to_cols_infra[rp].append(c)
                                    
                                    temp_infra = pd.DataFrame({f'overlay_{rp}': iso_subnational_df[cols].sum(axis=1)
                                         for rp, cols in sorted(rp_to_cols_infra.items())},
                                        index=iso_subnational_df.index)
                                    
                                    iso_subnational_df[f'eae_{infra_type}_count'] = temp_infra.apply(calculate_eae_vectorized, axis=1)
                        
                        # add ead per ci_system
                        selected_columns = [col for col in ead_df.columns if col.startswith('ead_') or col.startswith('number_landslides')]
                        iso_subnational_df = iso_subnational_df.merge(ead_df[['GID_2'] + selected_columns].groupby('GID_2').sum(),left_on='GID_2',right_index=True).reset_index(drop=True)

                        #adjust dataset
                        iso_subnational_df = iso_subnational_df.drop(columns = ['NL_NAME_1', 'VARNAME_2', 'NL_NAME_2', 'TYPE_2', 'ENGTYPE_2', 'CC_2', 'HASC_2']) # drop irrelevant columns
                        if hazard_type in ['landslide_eq', 'landslide_rf']:
                            haz_trig_rp_lst =  get_landslide_return_periods(hazard_type)[0]
                            iso_subnational_df = iso_subnational_df.drop(columns = ['ead_{}'.format(rp_trig) for rp_trig in haz_trig_rp_lst])
                        
                        #output dataset
                        gdf = gpd.GeoDataFrame(iso_subnational_df, geometry='geometry')
                        gdf = gdf.set_crs(3857, inplace=True)
                        output_file_path = output_risk_path / country / hazard_type / '{}_{}_{}'.format(country, hazard_type, sub_system) # Define the output file path
                        (output_file_path.parent).mkdir(parents=True, exist_ok=True)
                        gdf.to_parquet(output_file_path)  # Export the GeoDataFrame to a shapefile
                        # all sub systems in one gdf instead of multiple 
                        print("Calculation of risk completed for {}, {} - {}".format(country, sub_system, hazard_type))

                except Exception as error:
                    print(f"An error occurred for {country} {hazard_type} {ci_system}:", error) # An error occurred: name 'x' is not defined                  
                        
########################################################################################################################
################          Step 2: calculate csv files        #############################################################
########################################################################################################################

risk_dict = {}
exposure_dict = {} 
multi_exposure_dict = {}

#for hazard_type in hazard_types: risk_dict[hazard_type] = pd.DataFrame() 
for hazard_type in hazard_types:
    global_df = subnational_df.drop(columns = ['NL_NAME_1', 'VARNAME_2', 'NL_NAME_2', 'TYPE_2', 'ENGTYPE_2', 'CC_2', 'HASC_2', 'geometry'])
    global_df = global_df.set_index('GID_2')
    exposure_df = subnational_df.drop(columns = ['NL_NAME_1', 'VARNAME_2', 'NL_NAME_2', 'TYPE_2', 'ENGTYPE_2', 'CC_2', 'HASC_2', 'geometry'])
    exposure_df = exposure_df.set_index('GID_2')
    haz_rp_lst = get_return_periods(hazard_type)

    #prepare the global files
    exposure_system_dict = {}
    multi_exposure_system_dict = {}
    columns_to_add = [] # list for multi-hazard tab
    for ci_system in cis_dict:
        for sub_system in cis_dict[ci_system]:
            risk_columns = ['_min', '_Q1', '_Q2', '_Q3', '_max']
            risk_columns = [sub_system + item for item in risk_columns]
            for column in risk_columns: global_df[column] = pd.NA  # or use `None` for a general empty value
            columns_to_add.extend(risk_columns)   

        #prepare global exposure file
        haz_rp_names = ['{}_{}'.format(infra_type, rp) 
                        for sub_system in cis_dict[ci_system] 
                        for infra_type in cis_dict[ci_system][sub_system]
                        for rp in haz_rp_lst
                       ]
        
        count_names = ['{}_{}_count'.format(infra_type, rp)
            for sub_system in cis_dict[ci_system]
            for infra_type in cis_dict[ci_system][sub_system]
            for rp in haz_rp_lst
            if '_km2' in next((u for u in unit_dict[ci_system][sub_system] if u.startswith(infra_type)), '')]

        eae_names = ['eae_{}'.format(infra_type)
            for sub_system in cis_dict[ci_system]
            for infra_type in cis_dict[ci_system][sub_system]]       
        
        new_columns_df = pd.DataFrame({column: pd.NA for column in (haz_rp_names+count_names+eae_names)}, index=exposure_df.index)
        exposure_system_dict[ci_system] = pd.concat([exposure_df, new_columns_df], axis=1)  

        #Prepare global multi-exposure file
        osm_id_names = ['{}_osm_ids'.format(infra_type) 
                        for sub_system in cis_dict[ci_system] 
                        for infra_type in cis_dict[ci_system][sub_system]]

        new_columns_df = pd.DataFrame({column: pd.NA for column in osm_id_names}, index=exposure_df.index)
        multi_exposure_system_dict[ci_system] = pd.concat([exposure_df, new_columns_df], axis=1)  

    exposure_dict[hazard_type] = exposure_system_dict
    multi_exposure_dict[hazard_type] = multi_exposure_system_dict

    #update the global files
    for country in iso_lst:  
        if (output_risk_path.parent / 'risk_with_1in10cutoff' / country / hazard_type).exists():
            for ci_system in cis_dict:              
                for sub_system in cis_dict[ci_system]:
                    if hazard_type in ['earthquake']: #and sub_system in ['power']:
                        output_file_path = output_risk_path.parent / 'risk_with_1in475cutoff' / country / hazard_type / '{}_{}_{}'.format(country, hazard_type, sub_system) # Define the output file path
                    else: 
                        output_file_path = output_risk_path.parent / 'risk_with_1in10cutoff' / country / hazard_type / '{}_{}_{}'.format(country, hazard_type, sub_system) # Define the output file path
                    if output_file_path.exists():
                        ead_df = gpd.read_parquet(output_file_path) # read ead data
                        ead_df = ead_df.rename(columns={
                            col: col.replace('overlay_', '').replace('ead', sub_system) 
                            for col in ead_df.columns if col.startswith('ead_') or 'overlay_' in col})
                        
                        #risk
                        risk_columns = ['_min', '_Q1', '_Q2', '_Q3', '_max']
                        risk_columns = [sub_system + item for item in risk_columns]
                        global_df.update(ead_df[['GID_2'] + risk_columns].set_index('GID_2'), overwrite=True) #update global df
        
                        #exposure    
                        for infra_type in cis_dict[ci_system][sub_system]:
                            haz_rp_names = ['{}_{}'.format(infra_type, rp) for rp in haz_rp_lst]
                            if all(column in ead_df.columns for column in haz_rp_names):
                                exposure_dict[hazard_type][ci_system].update(ead_df[['GID_2'] + haz_rp_names].set_index('GID_2'), overwrite=True) 
                            if '{}_osm_ids'.format(infra_type) in ead_df.columns:
                                multi_exposure_dict[hazard_type][ci_system].update(ead_df[['GID_2', '{}_osm_ids'.format(infra_type) ]].set_index('GID_2'), overwrite=True)

                            # add count for polygons
                            count_list = ['{}_{}_count'.format(infra_type, rp) for rp in haz_rp_lst] 
                            if all(column in ead_df.columns for column in count_list):
                                #print(f'Add overlay count columns for {infra_type}')
                                exposure_dict[hazard_type][ci_system].update(ead_df[['GID_2'] + count_list].set_index('GID_2'), overwrite=True) 

                            # add eae
                            if 'eae_{}'.format(infra_type) in ead_df.columns:
                                #print(f'Add eae columns for {infra_type}')
                                exposure_dict[hazard_type][ci_system].update(ead_df[['GID_2'] + ['eae_{}'.format(infra_type)]].set_index('GID_2'), overwrite=True) 

    #risk
    global_df.reset_index(inplace=True)
    index_col = global_df.pop('GID_2')
    global_df.insert(5, 'GID_2', index_col)
    # Generate the total columns by summing across the relevant columns for each quantile
    sub_systems_lst = [key for sub_dict in cis_dict.values() for key in sub_dict.keys()]
    global_df['total_min'] = global_df[[f'{sub_system}_min' for sub_system in sub_systems_lst]].sum(axis=1, min_count=1).fillna(pd.NA)
    global_df['total_Q1'] = global_df[[f'{sub_system}_Q1' for sub_system in sub_systems_lst]].sum(axis=1, min_count=1).fillna(pd.NA)
    global_df['total_Q2'] = global_df[[f'{sub_system}_Q2' for sub_system in sub_systems_lst]].sum(axis=1, min_count=1).fillna(pd.NA)
    global_df['total_Q3'] = global_df[[f'{sub_system}_Q3' for sub_system in sub_systems_lst]].sum(axis=1, min_count=1).fillna(pd.NA)
    global_df['total_max'] = global_df[[f'{sub_system}_max' for sub_system in sub_systems_lst]].sum(axis=1, min_count=1).fillna(pd.NA)
    risk_dict[hazard_type] = global_df #save this in dictionary

#calculate multi-hazard risk
risk_dict['multi-hazard'] = risk_dict[hazard_type][['GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NAME_2', 'GID_2']]
columns_to_add = columns_to_add + ['total_min', 'total_Q1', 'total_Q2', 'total_Q3', 'total_max']
for column in columns_to_add: risk_dict['multi-hazard'][column] = pd.NA  # or use `None` for a general empty value

# Iterate through the other hazard types and add their values
for hazard_type in risk_dict:
    if hazard_type != 'multi-hazard':  # Exclude 'multi_hazard' from the addition
        # Add specified columns using .add(), filling missing values with 0
        risk_dict['multi-hazard'][columns_to_add] = risk_dict['multi-hazard'][columns_to_add].add(
            risk_dict[hazard_type][columns_to_add].set_index(risk_dict['multi-hazard'].index),
            fill_value=0
        )

# Create an Excel writer object for risk
with pd.ExcelWriter(summary_risk_path / "global_infra_risk.xlsx", engine='xlsxwriter') as writer:
    # Iterate over each hazard type and corresponding global_df
    for hazard_type, hazard_df in risk_dict.items():
        # Transfer 2020 euros to 2020 USD 
        cols_to_multiply = [col for col in hazard_df.columns if col not in ['GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NAME_2', 'GID_2']] # Identify the columns to multiply
        hazard_df[cols_to_multiply] = round(hazard_df[cols_to_multiply] * round(1/0.877, 2)) # Multiply the selected columns by (1 / 0.877)
        
        # Write the DataFrame to the sheet named after the hazard type
        hazard_df.to_excel(writer, sheet_name=hazard_type, index=False)

for ci_system in cis_dict:
    with pd.ExcelWriter(summary_risk_path / f"global_{ci_system}_exposure.xlsx", engine='xlsxwriter') as writer:
        for hazard_type in hazard_types:
            if hazard_type in exposure_dict and ci_system in exposure_dict[hazard_type]:
                exposure_df = exposure_dict[hazard_type][ci_system].reset_index()
                # Rearrange GID_2 column if needed
                if 'GID_2' in exposure_df.columns:
                    index_col = exposure_df.pop('GID_2')
                    exposure_df.insert(5, 'GID_2', index_col)

                #adjust header > add count, km2 or km + convert units (now in m2 and m)
                infra_type_list_ci_system = [infra for subsystem in cis_dict[ci_system].values() for infra in subsystem]
                for infra_type in infra_type_list_ci_system:
                    extension = next(
                        (u.split('_')[-1] 
                         for sub_system in unit_dict[ci_system] 
                         for u in unit_dict[ci_system][sub_system] 
                         if u.startswith(infra_type)),
                        None)
                    #print(f"For infra_type '{infra_type}', the extension is: {extension}")
                
                    columns_of_interest = [col for col in exposure_df.columns if infra_type in col and '_count' not in col]
                    rename_mapping = {col: f"{col}_{extension}" for col in columns_of_interest}
                    exposure_df[columns_of_interest]
                    exposure_df.rename(columns=rename_mapping, inplace=True)
                    
                    relevant_columns = list(rename_mapping.values())
                    if extension == 'km':
                        exposure_df[relevant_columns] = round((exposure_df[relevant_columns] / 1000),2) 
                    elif extension == 'm2':
                        exposure_df[relevant_columns] = round((exposure_df[relevant_columns] / 1000),2)  
                
                # Write the DataFrame to a sheet named after the hazard type
                exposure_df.to_excel(writer, sheet_name=hazard_type, index=False)
                

        #for ci_system in cis_dict: 
        landslide_exposure_df = multi_exposure_dict[hazard_types[0]][ci_system][['GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NAME_2']]
        flood_exposure_df = multi_exposure_dict[hazard_types[0]][ci_system][['GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NAME_2']]
        multi_exposure_df = multi_exposure_dict[hazard_types[0]][ci_system][['GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NAME_2']]
        infra_types = ['{}'.format(infra_type) for sub_system in cis_dict[ci_system] for infra_type in cis_dict[ci_system][sub_system]]
        for infra_type in infra_types:
            temp_df = multi_exposure_df[[]]    
            for hazard_type in hazard_types:
                to_concat = multi_exposure_dict[hazard_type][ci_system][['{}_osm_ids'.format(infra_type)]].rename(columns={'{}_osm_ids'.format(infra_type): '{}_osm_ids_{}'.format(infra_type, hazard_type)})
                to_concat['{}_osm_ids_{}'.format(infra_type, hazard_type)] = to_concat['{}_osm_ids_{}'.format(infra_type, hazard_type)].apply(convert_value)
                
                temp_df = pd.concat([temp_df, to_concat], axis=1)
                
            temp_df['landslide_osm_ids'] = temp_df[[f"{infra_type}_osm_ids_{hazard}" for hazard in ['landslide_eq', 'landslide_rf']]].apply(lambda row: sum(row, []), axis=1)
            landslide_exposure_df['{}_exposure_count'.format(infra_type)] = temp_df['landslide_osm_ids'].apply(lambda x: len(set(x)))
            hazard_columns = [f'{infra_type}_osm_ids_{hazard}' for hazard in hazard_types]
            landslide_exposure_df['{}_hazard_count'.format(infra_type)] = temp_df[hazard_columns].apply(lambda row: sum(1 if isinstance(x, list) and len(x) > 0 else 0 for x in row), axis=1)

            temp_df['flood_osm_ids'] = temp_df[[f"{infra_type}_osm_ids_{hazard}" for hazard in ['pluvial', 'fluvial', 'coastal']]].apply(lambda row: sum(row, []), axis=1)
            flood_exposure_df['{}_exposure_count'.format(infra_type)] = temp_df['flood_osm_ids'].apply(lambda x: len(set(x)))
            hazard_columns = [f'{infra_type}_osm_ids_{hazard}' for hazard in hazard_types]
            flood_exposure_df['{}_hazard_count'.format(infra_type)] = temp_df[hazard_columns].apply(lambda row: sum(1 if isinstance(x, list) and len(x) > 0 else 0 for x in row), axis=1)
            
            temp_df['total_osm_ids'] = temp_df.apply(lambda row: sum(row, []), axis=1)
            multi_exposure_df['{}_exposure_count'.format(infra_type)] = temp_df['total_osm_ids'].apply(lambda x: len(set(x)))
            hazard_columns = [f'{infra_type}_osm_ids_{hazard}' for hazard in hazard_types]
            multi_exposure_df['{}_hazard_count'.format(infra_type)] = temp_df[hazard_columns].apply(lambda row: sum(1 if isinstance(x, list) and len(x) > 0 else 0 for x in row), axis=1)

        # Write the DataFrame to a sheet named after the hazard type
        flood_exposure_df = flood_exposure_df.reset_index()
        # Rearrange GID_2 column if needed
        if 'GID_2' in flood_exposure_df.columns:
            index_col = flood_exposure_df.pop('GID_2')
            flood_exposure_df.insert(5, 'GID_2', index_col)
        flood_exposure_df.replace(0, np.nan, inplace=True)
        flood_exposure_df.to_excel(writer, sheet_name='flood-exposure', index=False)
        
        # Write the DataFrame to a sheet named after the hazard type
        landslide_exposure_df = landslide_exposure_df.reset_index()
        # Rearrange GID_2 column if needed
        if 'GID_2' in landslide_exposure_df.columns:
            index_col = landslide_exposure_df.pop('GID_2')
            landslide_exposure_df.insert(5, 'GID_2', index_col)
        landslide_exposure_df.replace(0, np.nan, inplace=True)
        landslide_exposure_df.to_excel(writer, sheet_name='landslide-exposure', index=False)

        # Write the DataFrame to a sheet named after the hazard type
        multi_exposure_df = multi_exposure_df.reset_index()
        # Rearrange GID_2 column if needed
        if 'GID_2' in multi_exposure_df.columns:
            index_col = multi_exposure_df.pop('GID_2')
            multi_exposure_df.insert(5, 'GID_2', index_col)
        multi_exposure_df.replace(0, np.nan, inplace=True)
        multi_exposure_df.to_excel(writer, sheet_name='multi-exposure', index=False)