################################################################
                ## Load package and set path ##
################################################################
import os,sys
import pygeos
import pandas as pd
import geopandas as gpd
from pathlib import Path
#from pgpkg import Geopackage
from geofeather.pygeos import to_geofeather, from_geofeather
from itertools import repeat
from osgeo import gdal 
gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join("..", "osmconf.ini"))

#sys.path.append("C:\Projects\Coastal_Infrastructure\scripts")
import extract
import functions
from multiprocessing import Pool,cpu_count
                
#def run_all(goal_area = 'Netherlands', local_path = 'C:/Users/snn490/surfdrive'):
def run_all(areas, goal_area = 'DeuFraIta', local_path = os.path.join('/scistor','ivm','snn490')):
    """Function to manage and run the model (in parts). 

    Args:
        *areas* ([str]): list with areas (e.g. list of countries)
        *goal_area* (str, optional): area that will be analyzed. Defaults to "Global". 
        *local_path* ([str], optional): Local pathway. Defaults to os.path.join('/scistor','ivm','snn490').
    """    

    extract_infrastructure(local_path)
    #extent_health_infrastructure(local_path)
    #base_calculations(local_path) 
    #base_calculations_global(local_path) #if base calcs per area already exist
    #cisi_calculation(local_path,goal_area)

################################################################
                    ## set variables ##
################################################################
 
def set_variables():
    """Function to set the variables that are necessary as input to the model

    Returns:
        *infrastructure_systems* (dictionary): overview of overarching infrastructure sub-systems as keys and a list with the associated sub-systems as values 
        *weight_assets* (dictionary): overview of the weighting of the assets per overarching infrastructure sub-system
        *weight_groups* (dictionary): overview of the weighting of the groups per overarching infrastructure sub-system
        *weight_subsystems* (dictionary): overview of the weighting of the sub-systems
    """    
    # Specify which subsystems and associated asset groups need to be analyzed
    infrastructure_systems = {
                        #"energy":["power"], 
                        "transportation": ["airports"]#["roads", "airports","railways"],
                        #"water":["water_supply"],
                        #"waste":["waste_solid","waste_water"], 
                        #"telecommunication":["telecom"],
                        #"healthcare": ["health_point","health_polygon"] #["health"],
                        #"education":["education_facilities"]
                        }

    return [infrastructure_systems]


################################################################
                    ## Set pathways ##
################################################################

def set_paths(local_path = 'C:/Data/CISI',extract_data=False,extent_health_data=False):
    """Function to specify required pathways for inputs and outputs

    Args:
        *local_path* (str, optional): local path. Defaults to 'C:/Data/CISI'.
        *extract_data* (bool, optional): True if extraction part of model should be activated. Defaults to False.
        *base_calculation* (bool, optional): True if base calculations part of model should be activated. Defaults to False.
        *cisi_calculation* (bool, optional): True if CISI part of model should be activated. Defaults to False.

    Returns:
        *osm_data_path* (str): directory to osm data
        *fetched_infra_path* (str): directory to output location of the extracted infrastructure data 
        *country_shapes_path* (str): directory to dataset with administrative boundaries (e.g. of countries)
        *infra_base_path* (str): directory to output location of the rasterized infrastructure data 

    """ 
    # Set path to inputdata
    #osm_data_path = os.path.abspath(os.path.join(local_path,'Datasets','OpenStreetMap')) #path to map with pbf files from OSM 
    osm_data_path = os.path.abspath(os.path.join('/scistor','ivm','data_catalogue','open_street_map','country_osm')) #path to map with pbf files from OSM at cluster
    shapes_file = 'global_countries_advanced.geofeather'
    country_shapes_path = os.path.abspath(os.path.join(local_path,'Datasets','Administrative_boundaries', 'global_countries_buffer', shapes_file)) #shapefiles with buffer around country


    # Set path for outputs 
    base_path = os.path.abspath(os.path.join(local_path, 'Outputs', 'gmhcira')) #this path will contain folders in which 

    # path to save outputs - automatically made, not necessary to change output pathways
    fetched_infra_path = os.path.abspath(os.path.join(base_path,'Fetched_infrastructure')) #path to map with fetched infra-gpkg's 

    #Create folders for outputs (GPKGs and pngs)
    #Path(output_histogram_path).mkdir(parents=True, exist_ok=True)
    Path(fetched_infra_path).mkdir(parents=True, exist_ok=True)

    if extract_data:
        return [osm_data_path,fetched_infra_path,country_shapes_path]
    if extent_health_data:
        return [fetched_infra_path,base_path]


################################################################
 ## Step 1: Extract requested infrastructure from pbf-file  ##
################################################################

def extract_infrastructure_per_area(area,groups_list,osm_data_path,fetched_infra_path,country_shapes_path):
    """function to extract infrastrastructure for an area 

    Args:
        *area* (str): area to be analyzed
        *osm_data_path* (str): directory to osm data
        *fetched_infra_path* (str): directory to output location of the extracted infrastructure data 
        *country_shapes_path* (str): directory to dataset with administrative boundaries (e.g. of countries)
    """
    #try:
    #get shape data
    shape_countries = from_geofeather(country_shapes_path) #open as geofeather

    fetched_data_dict = {group: pd.DataFrame() for group in groups_list} #Create dictionary with asset groups as keys and df as value
    print("\033[1mTime to extract infrastructure data for area: {}\033[0m".format(area))
    for group in groups_list:
        print("Infrastructure belonging to the group '{}' will now be extracted for {}".format(group, area))
        data = '{}.osm.pbf'.format(area) #make directory to data
        if group == 'power':
            fetched_data_area = extract.merge_energy_datatypes1(os.path.join(osm_data_path, data)) #extract required data
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase charachters
        elif group == 'roads':
            fetched_data_area = extract.roads_all(os.path.join(osm_data_path, data)) #extract required data
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase charachters
                list_of_highway_assets_to_keep =["living_street", "motorway", "motorway_link", "motorway_junction", "primary","primary_link", "residential","road", "secondary", "secondary_link","tertiary","tertiary_link", "trunk", "trunk_link","unclassified","service",
                "pedestrian","bus_guideway","escape","raceway","cycleway","construction","bus_stop","crossing","mini_roundabout", "passing_place","rest_area","turning_circle","traffic_island","yes","emergency_bay",""]
                fetched_data_area = fetched_data_area.loc[fetched_data_area.asset.isin(list_of_highway_assets_to_keep)].reset_index()
                mapping_dict = {
                    "living_street" : "other", 
                    "motorway" : "motorway", 
                    "motorway_link" : "motorway",
                    "motorway_junction" : "motorway",
                    "primary" : "primary", 
                    "primary_link" : "primary", 
                    "residential" : "other",
                    "road" : "other", 
                    "secondary" : "secondary", 
                    "secondary_link" : "secondary", 
                    "tertiary" : "tertiary", 
                    "tertiary_link" : "tertiary", 
                    "trunk" : "trunk",
                    "trunk_link" : "trunk",
                    "unclassified" : "other", 
                    "service" : "other",
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
                } 
                fetched_data_area['asset'] = fetched_data_area.asset.apply(lambda x : mapping_dict[x])  #reclassification
        elif group == 'airports':
            fetched_data_area = extract.merge_airport_datatypes(os.path.join(osm_data_path, data))
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters
                #reclassify assets 
                #mapping_dict = {
                #    "aerodrome" : "airports",
                #}
                #fetched_data_area['asset'] = fetched_data_area.asset.apply(lambda x : mapping_dict[x])  #reclassification
                #fetched_data_area['geometry'] =pygeos.buffer(fetched_data_area.geometry,0) #avoid intersection
        elif group == 'railways':
            fetched_data_area = extract.railway_all(os.path.join(osm_data_path, data))
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters
                list_of_railway_assets_to_keep =['rail','tram','subway','construction','funicular','light_rail','narrow_gauge']
                fetched_data_area = fetched_data_area.loc[fetched_data_area.asset.isin(list_of_railway_assets_to_keep)].reset_index()
                #reclassify assets 
                mapping_dict = {
                    "rail" : "railway",
                    "tram" : "railway",
                    "subway" : "railway",
                    "construction" : "railway",
                    "funicular" : "railway",
                    "light_rail" : "railway",
                    "narrow_gauge" : "railway",
                    "monorail" : "railway",
                }
                fetched_data_area['asset'] = fetched_data_area.asset.apply(lambda x : mapping_dict[x])  #reclassification   
        elif group == 'ports':
            fetched_data_area  = extract.ports(os.path.join(osm_data_path, data))
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters
                fetched_data_area['geometry'] =pygeos.buffer(fetched_data_area.geometry,0) #avoid intersection
        elif group == 'water_supply':
            fetched_data_area  = extract.water_supply(os.path.join(osm_data_path, data))
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters
                fetched_data_area['geometry'] =pygeos.buffer(fetched_data_area.geometry,0) #avoid intersection
        elif group == 'waste_solid':
            fetched_data_area  = extract.waste_solid(os.path.join(osm_data_path, data))
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters  
                fetched_data_area['geometry'] =pygeos.buffer(fetched_data_area.geometry,0) #avoid intersection             
        elif group == 'waste_water':
            fetched_data_area  = extract.waste_water(os.path.join(osm_data_path, data))
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters
                #reclassify assets 
                mapping_dict = {
                    "wastewater_plant" : "wastewater_treatment_plant"
                }
                fetched_data_area['asset'] = fetched_data_area.asset.apply(lambda x : mapping_dict[x])  #reclassification                
                fetched_data_area['geometry'] =pygeos.buffer(fetched_data_area.geometry,0) #avoid intersection
        elif group == 'telecom':
            fetched_data_area  = extract.telecom(os.path.join(osm_data_path, data))
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters
                #reclassify assets 
                mapping_dict = {
                    "communications_tower" : "communication_tower", #big tower
                    "tower" : "communication_tower", #small tower
                    "mast" : "mast"
                }
                fetched_data_area['asset'] = fetched_data_area.asset.apply(lambda x : mapping_dict[x])  #reclassification
        elif group == 'health':
            fetched_data_area  = extract.social_infrastructure_combined(os.path.join(osm_data_path, data))
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters
                #reclassify assets 
                #mapping_dict = {
                #    "doctors" : "doctors",
                #    "clinic" : "clinic",
                #    "hospital" : "hospital",
                #    "dentist" : "dentist",
                #    "pharmacy" : "pharmacy",
                #    "physiotherapist" : "others",
                #    "alternative" : "others",
                #    "laboratory" : "others",
                #    "optometrist" : "others",
                #    "rehabilitation" : "others",
                #    "blood_donation" : "others",
                #    "birthing_center" : "others"
                #}
                #fetched_data_area['asset'] = fetched_data_area.asset.apply(lambda x : mapping_dict[x])  #reclassification

        elif group == 'health_point':
            fetched_data_area  = extract.social_infrastructure_point(os.path.join(osm_data_path, data))
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters
        elif group == 'health_polygon':
            fetched_data_area  = extract.social_infrastructure_polygon(os.path.join(osm_data_path, data))
            if 'asset' in fetched_data_area.columns:
                fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters
                fetched_data_area['geometry'] =pygeos.buffer(fetched_data_area.geometry,0) #avoid intersection

        elif group == 'education_facilities':
                fetched_data_area  = extract.education(os.path.join(osm_data_path, data))
                if 'asset' in fetched_data_area.columns:
                    fetched_data_area['asset'] = list(map(lambda x: x.lower(), fetched_data_area['asset'])) #make sure that asset column is in lowercase characters
                    fetched_data_area['geometry'] =pygeos.buffer(fetched_data_area.geometry,0) #avoid intersection
        else:
            print("WARNING: No extracting codes are written for the following area and group: {} {}".format(area, group))

        #get rid of random floating data
        country_shape = shape_countries[shape_countries['ISO_3digit'] == area]
        if country_shape.empty == False: #if ISO_3digit in shape_countries
            spat_tree = pygeos.STRtree(fetched_data_area.geometry)
            fetched_data_area = functions.clip_pygeos(fetched_data_area,country_shape.iloc[0],spat_tree)
        else:
            print("ISO_3digit code not specified in file containing shapefiles of country boundaries. Floating data will not be removed for area '{}'".format(area))
            
        #Save data in df
        fetched_data_dict[group] = fetched_data_dict[group].append(fetched_data_area, ignore_index=True)
        
    #if all df's are empty for area, then warning. Otherwise, make outputs
    if functions.check_dfs_empty(fetched_data_dict) == False: #df's contain data
        for group in groups_list:
            #export when df is not empty 
            if fetched_data_dict[group].empty == False:
                print("Extraction of requested infrastructure is complete for group '{}' in area '{}'. This data will now be exported as geofeather...".format(group, area))
                #Export fetched exposure data as geopackage
                temp_df = functions.transform_to_gpd(fetched_data_dict[group])
                temp_df.to_file(os.path.join(fetched_infra_path, '{}_{}.gpkg'.format(area,group)), layer=' ', driver="GPKG")
                #with Geopackage(os.path.join(fetched_infra_path, '{}_{}.gpkg'.format(area,group)), 'w') as out:
                #    out.add_layer(fetched_data_dict[group], name=' ', crs='EPSG:4326')
                to_geofeather(fetched_data_dict[group], os.path.join(fetched_infra_path, '{}_{}.feather'.format(area, group)), crs="EPSG:4326") #save as geofeather
            else:
                print("NOTIFICATION: Extraction for group '{}' for area '{}' resulted in an empty df. No output will be made...".format(group, area)) 
    else:
        print("WARNING: No infrastructure data is found in area '{}'. Please check if OSM-file is correct and whether it intersects with polygon of area (country_shape)".format(area))

    #except Exception as e:
    #    print('ERROR: {} for {}'.format(e, area))


def group_infrastructure_assets(infrastructure_systems):
    """function to obtain a list of the infrastructure groups that will be analyzed 

    Args:
        *infrastructure_systems*: dictionary, overview of overarching infrastructure sub-systems as keys and a list with the associated sub-systems as values 

    Returns:
        *groups_list*: list of the infrastructure groups that will be analyzed
    """   
    groups_list = []
    for ci_system in infrastructure_systems:
        for group in infrastructure_systems[ci_system]:
            groups_list.append(group)

    return groups_list

def extract_infrastructure(local_path):
    """function to extract infrastructure per area, parallel processing 

    Args:
        *local_path*: Local pathway. Defaults to os.path.join('/scistor','ivm','snn490').
    """
    # get paths
    osm_data_path,fetched_infra_path,country_shapes_path = set_paths(local_path,extract_data=True)

    # get settings
    infrastructure_systems = set_variables()[0]

    # get all asset groups in a list, so dictionary can be created with asset groups as keys and df as value
    groups_list = group_infrastructure_assets(infrastructure_systems)

    # turn dict into list to make sure we have all unique areas
    #listed_areas = list(areas.values())[0]

    # run the extract parallel per area
    print('Time to start extraction of requested assets for the following areas: {}'.format(areas))
    with Pool(cpu_count()-1) as pool: 
        pool.starmap(extract_infrastructure_per_area,zip(areas,
                                                        repeat(groups_list,len(areas)),
                                                        repeat(osm_data_path,len(areas)),
                                                        repeat(fetched_infra_path,len(areas)),
                                                        repeat(country_shapes_path,len(areas))),
                                                        chunksize=1) 

##########################################################################
 ## Step 2: Extent health infrastructure using point and polygon data  ##
##########################################################################
def extent_health_data_per_area(area,fetched_infra_path,base_path):
    """function to extent health infrastrastructure data for an area, by transforming point data into polygon data and append it to polygon dataset. 

    Args:
        *area* (str): area to be analyzed
        *fetched_infra_path* (str): directory to output location of the extracted infrastructure data 
        *base_path* (str): directory to base folder
    """
    
    #import data
    df_point = from_geofeather(os.path.join(fetched_infra_path, '{}_{}.feather'.format(area,'health_point'))) #open as geofeather 
    df_polygon = from_geofeather(os.path.join(fetched_infra_path, '{}_{}.feather'.format(area,'health_polygon'))) #open as geofeather 

    #remove duplicates
    df_point_filtered = compare_point_to_polygon(df_point, df_polygon) #remove duplicates from point data

    #area calculations for polygons
    df_polygon.insert(1, "area_degrees", "") #add assettype as column after first column for length calculations
    if not df_polygon.empty:
        df_polygon["area_degrees"] = pygeos.area(df_polygon.geometry) #calculate area in degrees per object and put in dataframe
    area_per_type = df_polygon.groupby(['asset'])['area_degrees'].mean() #get mean area per infrastructure type

    #assign mean area to infrastructure type point data
    df_point_filtered.insert(1, "area_degrees", "")#create area deree column df point_filtered data
    poly_types = area_per_type.index.tolist() #all polygon infra types to list
    point_types = df_point_filtered.asset.unique().tolist() #all point infra types to list

    for infra_type in poly_types:
        if infra_type in point_types:
            df_point_filtered.loc[df_point_filtered['asset'] == infra_type, 'area_degrees'] = area_per_type.loc[infra_type]

    #assign area to infrastructure types that are absent in polygon dataset
    missing_infra_types_lst = list(set(point_types) - set(poly_types)) #get infra types that are in point_types but not poly_types
    mean_value = area_per_type.drop(['hospital', 'clinic']).mean() #get mean area of health facilities neglecting area of hospitals

    for infra_type in missing_infra_types_lst:
        df_point_filtered.loc[df_point_filtered['asset'] == infra_type, 'area_degrees'] = mean_value

    #create polygon out of point data (see https://stackoverflow.com/questions/57507739/python-how-to-create-square-buffers-around-points-with-a-distance-in-meters)
    df_point_filtered = df_point_filtered.rename(columns={'geometry' : 'geometry_point'})
    df_point_filtered.insert(1, "geometry", "") #add geometry column to store polygon geometries
    df_point_filtered['radius_degrees'] = (df_point_filtered['area_degrees']**(1/2))/2

    for row in df_point_filtered.itertuples():
        df_point_filtered['geometry'].loc[row.Index] = pygeos.buffer(row.geometry_point, row.radius_degrees, cap_style='square')

    #combine health point and polygon df's
    df_point_filtered = df_point_filtered.drop(columns=['geometry_point', 'radius_degrees','area_degrees'])
    df_polygon = df_polygon.drop(columns=['area_degrees'])
    df_health = df_polygon.append(df_point_filtered).reset_index(drop=True)

    #print("Extraction of requested infrastructure is complete for group '{}' in area '{}'. This data will now be exported as geofeather...".format(group, area))
    to_geofeather(df_health, os.path.join(base_path, '{}_{}.feather'.format(area, 'health')), crs="EPSG:4326") #save as geofeather
    #Export fetched exposure data as geopackage
    temp_df = cisi_exposure.transform_to_gpd(df_health)
    temp_df.to_file(os.path.join(base_path, '{}_{}.gpkg'.format(area,'health')), layer=' ', driver="GPKG")
    #with Geopackage(os.path.join(fetched_infra_path, '{}_{}.gpkg'.format(area,group)), 'w') as out:
    #    out.add_layer(fetched_data_dict[group], name=' ', crs='EPSG:4326')

def compare_point_to_polygon(df_point, df_polygon):
    """
    Function that removes points with overlapping polygons if asset type is similar    
    Arguments:
        *df_point* : a geopandas GeoDataFrame with specified unique healthcare point data.
        *df_polygon* : a geopandas GeoDataFrame with specified unique healthcare polygons.       
    Returns:
        *GeoDataFrame* : a geopandas GeoDataFrame with unique healthcare assets (point data)
    """   

    #check for each polygon which points are overlaying with it 
    df_polygon['geometry'] = pygeos.buffer(df_polygon.geometry,0) #avoid intersection
    spat_tree = pygeos.STRtree(df_polygon.geometry) # https://pygeos.readthedocs.io/en/latest/strtree.html
    for point_row in df_point.itertuples():
        df_polygon_overlap = (df_polygon.loc[spat_tree.query(point_row.geometry,predicate='intersects').tolist()]).sort_index(ascending=True) #get point that overlaps with polygon
        if not df_polygon_overlap.empty:
            #if point_row.asset in df_polygon_overlap['asset'].tolist(): #check if infrastructure type is the same
            df_point = df_point.drop(point_row.Index)
    
    return df_point.reset_index(drop=True)

def extent_health_infrastructure(local_path):
    """function to extent health infrastructure per area, parallel processing 

    Args:
        *local_path*: Local pathway. Defaults to os.path.join('/scistor','ivm','snn490').
    """
    # get paths
    fetched_infra_path,base_path = set_paths(local_path,extent_health_data=True)

    # turn dict into list to make sure we have all unique areas
    #listed_areas = list(areas.values())[0]

    # run the extract parallel per area
    print('Time to start extending health infrastructure dataset for the following areas: {}'.format(areas))
    with Pool(cpu_count()-1) as pool: 
        pool.starmap(extent_health_data_per_area,zip(areas,
                                                        repeat(fetched_infra_path,len(areas)),
                                                        repeat(base_path,len(areas))),
                                                        chunksize=1) 

if __name__ == '__main__':
    #receive nothing, run area below
    if (len(sys.argv) == 1):    
        areas = ['Galveston_Bay']#['Zuid-Holland']
        run_all(areas)
    else:
        # receive ISO code, run one country
        if (len(sys.argv) > 1) & (len(sys.argv[1]) == 3):    
            areas = []
            areas.append(sys.argv[1])
            run_all(areas)
        #receive multiple ISO-codes in a list, run specified countries
        elif '[' in sys.argv[1]:
            if ']' in sys.argv[1]:
                areas = sys.argv[1].strip('][').split('.') 
                run_all(areas)
            else:
                print('FAILED: Please write list without space between list-items. Example: [NLD.LUX.BEL]')
        #receive global, run all countries in the world
        elif (len(sys.argv) > 1) & (sys.argv[1] == 'global'):    
            #glob_info = pd.read_excel(os.path.join('/scistor','ivm','snn490','Datasets','global_information_short.xlsx'))
            #glob_info = pd.read_excel(os.path.join(r'C:\Users\snn490\surfdrive\Datasets','global_information_test.xlsx'))
            glob_info = pd.read_excel(os.path.join('/scistor','ivm','snn490','Datasets','global_information_advanced.xlsx'))
            areas = list(glob_info.ISO_3digit) 
            if len(areas) == 0:
                print('FAILED: Please check file with global information')
            else:
                run_all(areas)
        #receive continent, run all countries in continent
        elif (len(sys.argv) > 1) & (len(sys.argv[1]) > 3):    
            #glob_info = pd.read_excel(os.path.join('/scistor','ivm','snn490','Datasets','global_information_short.xlsx'))
            #glob_info = pd.read_excel(os.path.join(r'C:\Users\snn490\surfdrive\Datasets','global_information_test.xlsx'))
            glob_info = pd.read_excel(os.path.join('/scistor','ivm','snn490','Datasets','global_information_advanced.xlsx'))
            glob_info = glob_info.loc[glob_info.continent==sys.argv[1]]
            areas = list(glob_info.ISO_3digit) 
            if len(areas) == 0:
                print('FAILED: Please write the continents as follows: Africa, Asia, Central-America, Europe, North-America,Oceania, South-America') 
            else:
                run_all(areas)
        else:
            print('FAILED: Either provide an ISO3 country name or a continent name')