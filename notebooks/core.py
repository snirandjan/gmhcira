"""DamageScanner - a directe damage assessment toolkit

Copyright (C) 2023 Elco Koks. All versions released under the MIT license.
"""

# Get all the needed modules
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path

from vector import VectorScanner,buildings,landuse,cis

class DamageScanner(object):
    """DamageScanner - a directe damage assessment toolkit	
    """
    def __init__(self,  
                 data_path, 
                 exposure_data,
                 hazard_data,
                 curves = None, 
                 maxdam = None):
        
        """Prepare the input for a damage assessment
        """

        # specify the basics
        self.data_path = data_path       
        self.osm = False
        self.default_curves = False
        self.default_maxdam = False
        
        # Specify paths to exposure data
        self.exposure_path =  Path(data_path / 'exposure' / exposure_data)

        if self.exposure_path.suffix in ['.tif','.tiff','.nc']:
            self.assessment_type = 'raster'
        
        elif self.exposure_path.suffix in ['.shp','.gpkg','.pbf','.geofeather','.geoparquet']:
            self.assessment_type = 'vector'
            
            if self.exposure_path.suffix == '.pbf':
                self.osm = True
        else:
            raise ImportError(": The exposure data should be a a shapefile, geopackage, geoparquet, osm.pbf, geotiff or netcdf file.")
           
        # Specify path to hazard data
        if not isinstance(hazard_data,list):
            self.hazard_path = Path(data_path / 'hazard' / hazard_data)
        else:
            self.hazard_path = [Path(data_path / 'hazard' / x) for x in hazard_data]
            
        # Collect vulnerability curves
        if curves is None:

            self.curves = 'https://zenodo.org/records/10203846/files/Table_D2_Multi-Hazard_Fragility_and_Vulnerability_Curves_V1.0.0.xlsx?download=1'

            raise ImportWarning("You have decided to choose the default set of curves. Make sure the exposure data aligns with these curves: https://zenodo.org/records/10203846")
            self.default_curves = True

        elif isinstance(curves, pd.DataFrame):
            self.curves = curves
        else:
            self.curves = Path(data_path / 'vulnerability' / curves)

        # Collect maxdam information
        if maxdam is None:
            self.maxdam = 'https://zenodo.org/records/10203846/files/Table_D3_Costs_V1.0.0.xlsx?download=1'

            raise ImportWarning("You have decided to choose the default set of maximum damages. Make sure the exposure data aligns with these maximum damages: https://zenodo.org/records/10203846")
            self.default_maxdam = True

        elif isinstance(maxdam,dict):
            self.maxdam = maxdam
        else:    
            self.maxdam = Path(data_path / 'vulnerability' / maxdam)

    # def exposure(self):


    # def vulnerability(self):
        
    def calculate(self,
                  hazard_type=None,
                  save_output=False,
                  **kwargs):
        """Damage assessment. Can be a specific hazard event, or a specific single hazard footprint, or a list of events/footprints.
        """

        if not hasattr(self, 'assessment_type'):
            raise ImportError('Please run .prepare() first to set up the assessment')

            
        if self.assessment_type == 'raster':
            
            return RasterScanner(
                            exposure_file = self.exposure_path,
                            hazard_file = self.hazard_path,
                            curve_path = self.curves,
                            maxdam_path = self.maxdam,
                            save=save_output)

        elif self.assessment_type == 'vector':

            
            if self.default_curves:    
                if hazard_type == 'flood':
                    sheet_name = 'F_Vuln_Depth'
                elif hazard_type == 'windstorm':
                    sheet_name = 'W_Vuln_V10m'
                
            # specificy essential data input characteristics
            if 'cell_size' in kwargs:
                self.cell_size = kwargs.get('cell_size')
            else:
                self.cell_size = 5
    
            if 'exp_crs' in kwargs:
                self.exp_crs = kwargs.get('exp_crs')
            else:
                self.exp_crs = 4326
    
            if 'haz_crs' in kwargs:
                self.haz_crs = kwargs.get('haz_crs')
            else:
                self.haz_crs = 4326
    
            if 'object_col' in kwargs:
                self.object_col = kwargs.get('object_col')
            else:
                self.object_col = 'landuse'
    
            if 'hazard_col' in kwargs:
                self.hazard_col = kwargs.get('hazard_col')
            else:
                self.hazard_col = 'inun_val'
    
            if 'lat_col' in kwargs:
                self.lat_col = kwargs.get('lat_col')
            else:
                self.lat_col = 'y'
    
            if 'lon_col' in kwargs:
                self.lon_col = kwargs.get('lon_col')
            else:
                self.lon_col = 'x'
    
            if 'centimers' in kwargs:
                self.centimers = kwargs.get('centimers')#0.01666,#5,
            else:
                self.centimers = False
            
            if self.osm:
                if 'buildings' in kwargs:
                    self.exposure_data = buildings(self)                  
                    self.exposure_data = self.exposure_data.rename({'building':'element_type'},axis=1)       
                    
                elif 'roads' in kwargs:
                    self.exposure_data = cis(self,infra_type='road')
                    self.exposure_data = self.exposure_data.rename({'highway':'element_type'},axis=1)       
                
                elif 'landuse' in kwargs:
                    self.exposure_data = landuse(self)
                    self.exposure_data = self.exposure_data.rename({'landuse':'element_type'},axis=1)       
                else:
                    raise RuntimeError('When using OSM data, you need to specify the object type (e.g. road, buildings, landuse)')
            else:
                self.exposure_data = self.exposure_path
            
            return VectorScanner(
                            exposure_file = self.exposure_data,
                            hazard_file = self.hazard_path,
                            curve_path = self.curves,
                            maxdam_path = self.maxdam,
                            cell_size = self.cell_size, #0.01666,#5,
                            exp_crs = self.exp_crs, #28992,
                            haz_crs = self.haz_crs, #4326,
                            object_col= self.object_col, #'landuse',
                            hazard_col= self.hazard_col,
                            lat_col = self.lat_col,
                            lon_col = self.lon_col,
                            centimeters= self.centimers, #False,
                            save=save_output)  

                
    def risk(self):
        """Risk assessment
        """
        pass

if __name__ == '__main__':

    #country = 'TJK'
    

    for country in ['TJK']:#''PAK','PNG']:
        data_path = Path('C:\\Users\\eks510\\OneDrive - Vrije Universiteit Amsterdam\\2_Projects\\ADB\\')
        vulnerability_path = data_path / 'Data' / 'vulnerability'
        flood_data = data_path / 'Data' / 'Flood_data_CDRI'
        
        
        all_tif_files = list(Path(flood_data).rglob('*.tif'))

        flood_depth = 'RD_30m_4326'
        pluvial = 'FLSW_U'
        fluvial = 'FLRF_U'

        flood_country_dict = {'TJK':'tajikistan',
                        'PAK':'pakistan',
                        'PNG':'papua-new-guinea'}
        


#        flood_depth_files = [x for x in all_tif_files if (flood_country_dict[country] in str(x)) & (flood_depth in str(x))]
        flood_depth_files = [x for x in all_tif_files if country in str(x)]

        maxdam = {'unclassified':300, 
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
        
        # collect vulnerability data
        all_files = list(Path(vulnerability_path).glob('*.xlsx'))
        vulnerability_file = [x for x in all_files if 'Curves' in str(all_files)][0]

        vul_df = pd.read_excel(vulnerability_file,sheet_name='F_Vuln_Depth')

        road_curves = vul_df[['ID number','F7.1','F7.2','F7.2a','F7.2b','F7.3','F7.4','F7.5','F7.6','F7.7','F7.8','F7.9','F7.10','F7.11','F7.12','F7.13']]
        road_curves = road_curves.iloc[4:125,:]
        road_curves.set_index('ID number',inplace=True)
        road_curves.index = road_curves.index.rename('Depth')  

        for flood_file in flood_depth_files:
            TJK = DamageScanner(data_path=data_path / 'Data',
                                exposure_data = f'{country}.osm.pbf',
                                hazard_data = flood_file,
                                curves = road_curves,
                                maxdam = maxdam
                                )
        
            output = TJK.calculate(roads = True,
                        cell_size=0.00027778,
                        exp_crs = 4326,
                        haz_crs = 4326,
                        centimeters=True           
)

            # return_period = str(flood_file).split('\\')[-1].split('_')[4]
            # flood_type = str(flood_file).split('\\')[-1].split('_')[2]

            return_period = str(flood_file).split('\\')[-1].split('_')[2].split('.')[0]
            flood_type = str(flood_file).split('\\')[-1].split('_')[1]

            save_path = data_path / 'Data' / 'flood_results_CDRI' / f"{country}_{flood_type}_{return_period}.parquet"
            output.to_parquet(save_path)
