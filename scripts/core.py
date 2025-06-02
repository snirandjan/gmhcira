"""
Core code to run the global-scale risk assessment for multiple hazards and infrastructure 
  
@Author: Sadhana Nirandjan - Institute for Environmental studies, VU University Amsterdam
"""

import os,sys
import geopandas as gpd
from pathlib import Path
import risk_workflow_countries
import osm_extract

################################################################
                    ## set variables ##
################################################################
 
def set_variables():
    """Function to set the variables that are necessary as input to the model

    Returns:
        *cis_dict* (dictionary): overview of overarching infrastructure sub-systems as keys and a list with the associated sub-systems as values 
        *weight_assets* (dictionary): overview of the weighting of the assets per overarching infrastructure sub-system
        *weight_groups* (dictionary): overview of the weighting of the groups per overarching infrastructure sub-system
        *weight_subsystems* (dictionary): overview of the weighting of the sub-systems
    """    
    # Specify which subsystems and associated assets need to be analyzed 
    cis_dict = {
        "energy": {"power": ["transmission_line","distribution_line","cable","plant","substation",
                            "power_tower","power_pole"]},
        "transportation": {"road":  ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'road', 'track' ], 
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

    # natural hazard that needs to be analyzed
    hazard_types = ['pluvial'] #['fluvial', 'coastal', 'landslide_eq', 'landslide_rf', 'earthquake'] 

    # input specifications 
    eq_data = 'GIRI' #GIRI, GEM or GAR data
    flood_data = {'version': 'Fathom_v3', # options: 'Fathom_v2' or 'Fathom_v3' 
                    'year': 2020, # options: None (for Fathom_v2), '2020', '2030', '2050', '2080' (for Fathom_v3)
                    'scenario': None} #None (for Fathom_v2 and v3), 'SSP1_2.6', 'SSP2_4.5', 'SSP3_7.0', 'SSP5_8.5' (for Fathom_v3)
    landslide_rf_data = {'scenario': None} #None for current situation, 'SSP126', 'SSP585' 
    
    #Vuln. settings
    database_id_curves=True # True if curve ids are specified in an external file 
    database_maxdam=False # True if max dams are specified in an external file
    vuln_settings = [database_id_curves, database_maxdam]

    #overwrite existing files?
    overwrite = True 

    return [cis_dict, hazard_types, eq_data, flood_data, landslide_rf_data, vuln_settings, overwrite]


################################################################
                    ## Set pathways ##
################################################################
def set_paths():
    """
    Defines and returns a dictionary of standardized file paths required for risk assessment.

    This function sets the base, data, and project directories, including locations for 
    various hazard datasets (flood, earthquake, landslide, cyclone) and output storage. 
    The paths are hardcoded for a specific file system structure and used throughout 
    the risk assessment workflow to load inputs and save results.

    Returns:
        dict: A dictionary (`pathway_dict`) containing the following paths:
            - data_path: Path to general data resources (e.g., vulnerability, GADM, global info).
            - flood_data_path: Path to flood hazard data (Fathom).
            - eq_data_path: Path to earthquake hazard data (GIRI).
            - landslide_data_path: Path to landslide hazard and rainfall data.
            - cyclone_data_path: Path to tropical cyclone hazard data.
            - output_path: Path to the output directory for storing assessment results.

    Note:
        This function assumes a specific directory structure based on the user's environment 
        and may need to be adjusted for other systems.
    """
    
    # define your pathways
    base_path = Path('/home/snirandjan/')
    data_path = Path('/home/snirandjan/Projects/gmhcira/data') 
    project_space_path = Path('/projects/0/prjs1486/datasets')

    flood_data_path = project_space_path / 'fathomv3' # Flood data
    eq_data_path = project_space_path / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'earthquakes' / 'GIRI' # Earthquake data
    landslide_data_path = project_space_path / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'landslides' # Landslide data, should also contain a folder rainfall with the global normalized rainfall data
    cyclone_data_path = project_space_path / 'data_catalogue' / 'open_street_map' / 'global_hazards' / 'tropical_cyclones' # Cyclone data
    output_path = base_path / 'Outputs' / 'gmhcira' # Will contain the outputs

    pathway_dict = risk_workflow_countries.create_pathway_dict(data_path, flood_data_path, eq_data_path, landslide_data_path, cyclone_data_path, output_path)

    return pathway_dict


################################################################
                    ## Set pathways ##
################################################################

def run_risk_assessment(areas):
    """
    Runs a country-level risk assessment workflow for the specified ISO3 country codes.

    This function manages the execution of risk assessments for one or more countries
    by calling the appropriate processing function (`damage_calculation` or 
    `damage_calculation_admin2`) based on the country. It loads necessary paths 
    and configuration settings, then loops through each country code provided 
    and performs the assessment in sequence.

    Args:
        areas (list of str): List of ISO3 country codes (e.g., ['NLD', 'ETH', 'CHN']) 
            for which the risk assessment should be run.

    Notes:
        - Certain countries (e.g., 'NGA', 'NER', 'USA') use an alternative admin2-level
          processing method.
        - All required paths and input data are initialized internally via helper 
          functions (`set_paths` and `set_variables`).
        - Errors encountered during the processing of a country are caught and printed 
          to allow the remaining countries to continue processing.
    """

    # get paths
    pathway_dict = set_paths()

    # get settings
    cis_dict, hazard_types, eq_data, flood_data, landslide_rf_data, vuln_settings, overwrite = set_variables()[0:7]

    #Uncomment if you would like to extract the data relevant infrastructure data first
    # for area in areas:
    #     osm_extract.asset_extraction(area,
    #                                             pathway_dict,
    #                                             cis_dict,
    #                                             overwrite)

    for area in areas:
        try:   
            if area not in ['NGA', 'NER', 'UZB', 'ETH', 'ZMB', 'LBY', 'AFG', 'CHL', 'FJI', 'KIR', 'AUS', 'BRA', 'CAN', 'CHN', 'IDN', 'KAZ', 'RUS', 'USA']: 
                risk_workflow_countries.damage_calculation(area,
                                                        pathway_dict,
                                                        cis_dict,
                                                        hazard_types,
                                                        eq_data,
                                                        flood_data,
                                                        landslide_rf_data,
                                                        vuln_settings,
                                                        overwrite)
            else:
                risk_workflow_countries.damage_calculation_admin2(area,
                                                        pathway_dict,
                                                        cis_dict,
                                                        hazard_types,
                                                        eq_data,
                                                        flood_data,
                                                        landslide_rf_data,
                                                        vuln_settings,
                                                        overwrite)
        except Exception as error:
            print(f"An error occurred for {area}:", error) # An error occurred: name 'x' is not defined
        
if __name__ == '__main__':
    # receive ISO code, run one country
    if (len(sys.argv) > 1) & (len(sys.argv[1]) == 3):    
        areas = []
        areas.append(sys.argv[1])
        run_risk_assessment(areas)
    #receive multiple ISO-codes in a list, run specified countries
    elif '[' in sys.argv[1]:
        if ']' in sys.argv[1]:
            areas = sys.argv[1].strip('][').split('.') 
            run_risk_assessment(areas)
        else:
            print('FAILED: Please write list without space between list-items. Example: [NLD.LUX.BEL]')
    #receive global, run all countries in the world
    elif (len(sys.argv) > 1) & (sys.argv[1] == 'global'):    
        #glob_info = pd.read_excel(os.path.join('/scistor','ivm','snn490','Datasets','global_information_advanced.xlsx'))
        glob_info = gpd.read_parquet(Path('/home/snirandjan/Projects/gmhcira/data')/ 'gadm' / 'gadm_410_simplified_admin0_income') 
        glob_info = glob_info.sort_values(by='area_km2', ascending=True)
        areas = list(glob_info.GID_0)
        if len(areas) == 0:
            print('FAILED: Please check file with global information')
        else:
            print(areas)
            run_risk_assessment(areas)
    #receive income class, run all countries in income class
    elif (len(sys.argv) > 1) & (len(sys.argv[1]) > 3):    
        glob_info = gpd.read_parquet(Path('/home/snirandjan/Projects/gmhcira/data') / 'gadm' / 'gadm_410_simplified_admin0_income') 
        glob_info = glob_info.loc[glob_info['Income group']==sys.argv[1]]
        glob_info = glob_info.sort_values(by='area_km2', ascending=True)
        areas = list(glob_info.GID_0) 

        #uncomment if you would like to run code from certain country
        # if 'SYR' in areas: #low income
        #     areas = areas[areas.index('SYR'):]
        # elif 'BGD' in areas: #lower middle income
        #     areas = areas[areas.index('BGD'):]
        # elif 'ECU' in areas: #upper middle income SLV
        #     areas = areas[areas.index('ECU'):]
        # elif 'DNK' in areas: # high income SHN
        #     areas = areas[areas.index('DNK'):]

        if len(areas) == 0:
            print('FAILED: Please write the continents as follows: Low income, Lower middle income, Upper middle income, High income') 
        else:
            print(areas)
            run_risk_assessment(areas)
    else:
        print('FAILED: Either provide an ISO3 country name or a income group')