"""
Global Multiple-Hazard Critical Infrastructure Risk Analysis

This script contains the code used to generate the global spatial figures presented in Figures 3, 4, and 5 of the manuscript, as well as Figure A1 in the Appendix.
  
@Author: Sadhana Nirandjan  - Institute for Environmental studies, VU University Amsterdam
"""

import re
import warnings
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

warnings.simplefilter(action='ignore', category=FutureWarning)

################################################################
                    ## set pathways ##
################################################################
project_root = Path("/path/to/gmhcira") # Adjust these paths to match your local setup
output_root = Path("/path/to/gmhcira_outputs") # Adjust these paths to match your local setup

# Input data
admin_path = project_root / "data" / "gadm" / "gadm_410-levels.gpkg"
vuln_path = project_root / "data" / "Vulnerability"

# Intermediate/output data
damage_data_path = output_root / "damage"
extracts_data_path = output_root / "extracts"
output_risk_path = output_root / "risk_summaries" # folder with Excel files

# Figures
figures_path = output_root / "figures"
figures_path.mkdir(parents=True, exist_ok=True)


################################################################
                    ## settings ##
################################################################
# Settings
hazard_types = ['fluvial', 'pluvial', 'coastal', 'earthquake', 'landslide_rf', 'landslide_eq', 'windstorm'] 

cis_dict = {
    "energy": {"power": ["transmission_line","distribution_line","cable","plant","substation",
                        "power_tower","power_pole"]},
    "transportation": {"road":  ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'residential', 'road', 'track' ], 
                        "air": ["airport", "runway", "terminal"],
                        "rail": ["railway"]}, 
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


################################################################
                    ## read risk data and admin boundaries for fig 4 and 5 ##
################################################################

# read risk data and administrative boundaries

# Read the Excel file into a dictionary
sheet_lst = hazard_types + ['multi-hazard']
sheets_dict = pd.read_excel(output_risk_path / "global_infra_risk.xlsx", sheet_name=None) # `sheet_name=None` reads all sheets into a dictionary of DataFrames
hazard_dict = {key: sheets_dict[key] for key in sheet_lst if key in sheets_dict} # Filter the sheets to match the hazard types

# Read subnational data
subnational_df = gpd.read_parquet(Path(admin_path.parent, "gadm_410_admin2_complete")) #GADM countries
for key in hazard_dict: 
    hazard_dict[key] = hazard_dict[key].merge(subnational_df[['GID_2', 'geometry']], on='GID_2', how='left') # Merge subnational_df with each DataFrame in the hazard_dict
    hazard_dict[key] = gpd.GeoDataFrame(hazard_dict[key], geometry='geometry').set_crs(4326, inplace=True)
print('risk numbers loaded')

gadm_countries = gpd.read_parquet(admin_path.parent / "gadm_410_simplified_admin0_income") 
print('admin 0 file loaded')


##################################
# color scheme
####################################

hazard_colors = {
    "fluvial": "#0F7FA8",        # deep river blue
    "pluvial": "#0AA6A0",        # blue-green teal
    "coastal": "#12CFA3",        # fresh sea green
    "earthquake": "#6F4A57",     # muted dark mauve
    "landslide_rf": "#FFD166",  # rainfall-triggered (yellow)
    "landslide_eq": "#F79A6C",  # EQ-triggered (orange-peach)
    "windstorms": "#E84A5F"     # strong warm red-pink
}

ci_sys_colors = {
    "energy": "#3F7FA0",            # slightly deeper muted blue
    "transportation": "#9CAFB7",    # cool grey-blue
    "telecommunication": "#CDB4DB", # soft lavender
    "water": "#EAD2AC",             # warm sand
    "waste": "#E6B89C",             # peach
    "health": "#FE938C",            # soft coral
    "education": "#B5838D"          # muted rose (anchor)
}

ci_subsys_colors = {
    # Energy
    "power": "#3F7FA0",

    # Transportation
    "road": "#7F96A1",
    "rail": "#9CAFB7",
    "air":  "#B5C6CE",

    # Water
    "water_supply": "#EAD2AC",

    # Waste
    "waste_solid": "#E6B89C",
    "waste_water": "#F1C9AE",

    # Telecommunication
    "telecom": "#CDB4DB",

    # Healthcare
    "healthcare": "#FE938C",

    # Education
    "education": "#B5838D"
}


################################################################
                    ## Create figure 3: global distribution of infrastructure exposure #
################################################################

#defining function filter for categories
def filter_assets(x):
    if x <= 100:
        return '1'
    if (x > 100 and x <= 1000):
        return '2'
    if (x > 1000  and x <= 10000):
        return '3'
    if (x > 10000  and x <= 100000):
        return '4'
    if (x > 100000  and x <= 1000000):
        return '5'

def filter_hazard(x):
    if (x > 0 and x <= 1):
        return 'A'
    if (x > 1 and x <= 2):
        return 'B'
    if (x > 2  and x <= 3):
        return 'C'
    if (x > 3  and x <= 4):
        return 'D'
    if x >= 5:
        return 'E'
    
# Inspiration for color codes: https://www.joshuastevens.net/cartography/make-a-bivariate-choropleth-map/
# Define the corner colors
top_left = "#E6DDC6" #'#f7fbff'    # Low x, Low y
top_right = '#64acbe' #'#08306b'   # Low x, High y
bottom_left = '#c85a5a' #'#f7fcb9' # High x, Low y
bottom_right = "#6B1F2B" #'#574249'  #'#00441b' # High x, High y

def interpolate_cmap(top_left, top_right, bottom_left, bottom_right, size=5):
    cmap_matrix = np.zeros((size, size, 3))
    
    for i in range(size):
        for j in range(size):
            row_top = np.array(plt.cm.colors.hex2color(top_left)) * (1 - j/(size-1)) + np.array(plt.cm.colors.hex2color(top_right)) * (j/(size-1))
            row_bottom = np.array(plt.cm.colors.hex2color(bottom_left)) * (1 - j/(size-1)) + np.array(plt.cm.colors.hex2color(bottom_right)) * (j/(size-1))
            cmap_matrix[i, j] = row_top * (1 - i/(size-1)) + row_bottom * (i/(size-1))
    return cmap_matrix

# Generate the colormap matrix
cmap_matrix = interpolate_cmap(top_left, top_right, bottom_left, bottom_right)

# Extract color codes and store in a list
all_colors_list = []
for i in range(cmap_matrix.shape[0]):
    for j in range(cmap_matrix.shape[1]):
        # Convert RGB to HEX
        color = plt.cm.colors.rgb2hex(cmap_matrix[i, j])
        all_colors_list.append(color)

# # Display the color codes
# for idx, color in enumerate(all_colors_list):
#     print(f'Category {idx+1}: {color}')

# # Display the color matrix
# plt.imshow(cmap_matrix, interpolation='nearest')
# plt.axis('off')
# plt.show()
    

##################################################
## Create multi-exposure figure for each CI system (appendix)
###############################

# read risk data and administrative boundaries
for ci_system in ['transportation', 'energy', 'waste', 'water', 'telecommunication', 'healthcare', 'education']: 
    # Read the Excel file into a dictionary
    exposure_sheet_lst = hazard_types + ['multi-exposure']
    exposure_sheets_dict = pd.read_excel(output_risk_path / f"global_{ci_system}_exposure.xlsx", sheet_name=None)
    exposure_dict = {key: exposure_sheets_dict[key] for key in exposure_sheet_lst if key in exposure_sheets_dict} # Filter the sheets to match the hazard types

    # Read subnational data
    for key in exposure_dict: 
        exposure_dict[key] = exposure_dict[key].merge(subnational_df[['GID_2', 'geometry']], on='GID_2', how='left') # Merge subnational_df with each DataFrame in the hazard_dict)
        exposure_dict[key] = gpd.GeoDataFrame(exposure_dict[key], geometry='geometry').set_crs(4326, inplace=True)

    # retrieve hazard exposure per admin
    hazard_collection = subnational_df[['GID_0', 'COUNTRY', 'GID_2']].copy()
    for hazard_type in hazard_types:
        temp_df = exposure_dict[hazard_type].copy()

        # find all exposure columns that start with 'eae'
        eae_cols = [c for c in temp_df.columns if c.startswith('eae')]
        if not eae_cols:
            # no eae columns -> everything is 0 for this hazard
            hazard_collection[hazard_type] = 0
            continue

        # flag = 1 if any eae_* > 0 on the row, else 0
        # (fill NaNs with 0 before comparison)
        has_any = temp_df[eae_cols].fillna(0).gt(0).any(axis=1).astype('int8')

        # index by GID_2 for quick alignment
        if 'GID_2' not in temp_df.columns:
            raise KeyError(f"'GID_2' not found in exposure_dict[{hazard_type}]")
        haz_series = pd.Series(has_any.values, index=temp_df['GID_2'])

        # map onto hazard_collection by GID_2; missing -> 0
        hazard_collection[hazard_type] = hazard_collection['GID_2'].map(haz_series).fillna(0).astype('int8')

    # (optional) add a column indicating presence of ANY hazard exposure
    hazard_cols = [h for h in hazard_types if h in hazard_collection.columns]
    hazard_collection['multiple_hazard'] = (hazard_collection[hazard_types].fillna(0).sum(axis=1).astype('int16'))

    # quick peek
    hazard_collection.head()


    # get a 'ci_system_exposure_count' and 'ci_system_hazard_count'
    # Work on a copy
    df = exposure_dict['multi-exposure'].copy()

    # 1) Detect columns
    exposure_cols = [c for c in df.columns if c.endswith('_exposure_count')]
    hazard_cols   = [c for c in df.columns if c.endswith('_hazard_count')]

    if not exposure_cols:
        raise ValueError("No '*_exposure_count' columns found in exposure_dict['multi-exposure'].")
    if not hazard_cols:
        raise ValueError("No '*_hazard_count' columns found in exposure_dict['multi-exposure'].")

    # 2) (Optional) coerce to numeric (in case any are strings)
    df[exposure_cols] = df[exposure_cols].apply(pd.to_numeric, errors='coerce')
    df[hazard_cols]   = df[hazard_cols].apply(pd.to_numeric, errors='coerce')

    # 3) ci_system_exposure_count = sum across all assets' exposure counts
    df[f'{ci_system}_exposure_count'] = df[exposure_cols].sum(axis=1, skipna=True)

    # 4) ci_system_hazard_count = max across all assets' hazard counts
    df[f'{ci_system}_hazard_count'] = df[hazard_cols].max(axis=1, skipna=True)

    # 5) (Optional) which asset dominates the hazard count?
    #    This gives you the asset name (prefix before '_hazard_count') that had the max.
    #    If there are ties or all-NaN, this picks the first max per row.
    if hazard_cols:
        # argmax over values
        vals = df[hazard_cols].to_numpy()
        # mask rows where all are NaN; handle safely
        all_nan = np.isnan(vals).all(axis=1)
        argmax_idx = np.nanargmax(np.where(np.isnan(vals), -np.inf, vals), axis=1)
        max_cols = np.array(hazard_cols)[argmax_idx]
        # extract asset name (prefix)
        get_asset = np.vectorize(lambda col: re.sub(r'_hazard_count$', '', col))
        df[f'{ci_system}_hazard_asset'] = get_asset(max_cols)
        df.loc[all_nan, f'{ci_system}_hazard_asset'] = pd.NA

    # 6) (Optional) cast counts to integers (nullable) if you prefer ints
    for col in [f'{ci_system}_exposure_count', f'{ci_system}_hazard_count']:
        df[col] = df[col].round().astype('Int64')

    # Put the result back if you like
    exposure_dict['multi-exposure'] = df


    # exposure_dict['multi-exposure'][f'{ci_system}_hazard_count'] == hazard_collection['multiple_hazard']
    # 1) Pull and align the two series on GID_2
    s_haz = (hazard_collection.set_index('GID_2')['multiple_hazard'].astype('Int64'))            
    s_trans = (exposure_dict['multi-exposure'].set_index('GID_2')[f'{ci_system}_hazard_count'].astype('Int64'))

    # 2) Combine into one frame (outer join to see any missing keys)
    cmp = pd.concat({'multiple_hazard': s_haz, f'{ci_system}_hazard_count': s_trans},axis=1)

    # 3) Normalize NaNs -> 0 for strict numeric compare (optional)
    cmp_filled = cmp.fillna(0).astype(int)

    # 4) Compare and summarize
    cmp_filled['equal'] = cmp_filled['multiple_hazard'] == cmp_filled[f'{ci_system}_hazard_count']
    mismatches = cmp_filled[~cmp_filled['equal']]

    print(f"Total rows compared: {len(cmp_filled)}")
    print(f"Matches: {cmp_filled['equal'].sum()}  |  Mismatches: {len(mismatches)}")

    # 5) Peek at mismatches
    if not mismatches.empty:
        display_cols = ['multiple_hazard', f'{ci_system}_hazard_count']
        print("\nFirst mismatches:")
        print(mismatches[display_cols].head(10))

    # 6) (Optional) assert equality to fail fast
    # assert mismatches.empty, "multiple_hazard != transportation_hazard_count for some GID_2"

    #too many mismatches, just replace the whole column with right one
    # Build a lookup from hazard_collection
    lookup = (hazard_collection.set_index('GID_2')['multiple_hazard'].astype('Int64'))

    # Replace the whole column in exposure_dict['multi-exposure'] by mapping GID_2
    df = exposure_dict['multi-exposure'].copy()
    df[f'{ci_system}_hazard_count'] = df['GID_2'].map(lookup).fillna(0).astype('Int64')

    # Save back
    exposure_dict['multi-exposure'] = df


    #applying the filter function to 'Salary' column 
    exposure_dict['multi-exposure'][f'{ci_system}_exposure_cat'] = (exposure_dict['multi-exposure'][f'{ci_system}_exposure_count']).apply(filter_assets)
    exposure_dict['multi-exposure'][f'{ci_system}_hazard_cat'] = exposure_dict['multi-exposure'][f'{ci_system}_hazard_count'].apply(filter_hazard)


    # Combine x and y codes to create Bi_Class
    exposure_dict['multi-exposure']['Bi_Class'] = exposure_dict['multi-exposure'][f'{ci_system}_exposure_cat'] + exposure_dict['multi-exposure'][f'{ci_system}_hazard_cat']

    exposure_dict['multi-exposure'][exposure_dict['multi-exposure']['Bi_Class'].isna()]

    ############
    # link color map to unique categories found in dataset
    ############

    # Define your categories
    categories = [f"{i}{chr(65+j)}" for i in range(1, 6) for j in range(5)]

    # Create a dictionary that maps each category to a color
    category_color_map = dict(zip(categories, all_colors_list))

    unique_categories = exposure_dict['multi-exposure']['Bi_Class'].unique() # Extract unique categories from the dataframe
    colors = [color for category, color in category_color_map.items() if category in unique_categories] # Filter category_color_map based on unique categories

    cmap = mcolors.ListedColormap(colors)

    # Extent (with padding)
    xmin, ymin, xmax, ymax = subnational_df.total_bounds
    pad_x = 0.02 * (xmax - xmin)
    pad_y = 0.02 * (ymax - ymin)
    xlim = (xmin - pad_x, xmax + pad_x)
    ylim = (ymin - pad_y, ymax + pad_y)

    mpl.rcParams['hatch.linewidth'] = 0.2   # default is ~1.0

    fig, ax = plt.subplots(figsize=(8,8))

    # Background national polygons (under everything)
    subnational_df.plot(ax=ax, facecolor ="#D9D9D9", edgecolor='none', zorder=0) #'#edede9'

    # Choropleth with hatched "no data" areas
    exposure_dict['multi-exposure'].plot(
        ax=ax,
        column='Bi_Class',
        cmap=cmap,
        categorical=True,
        legend=False,
        zorder=1,
        missing_kwds=dict(
            color="#D9D9D9", #'#F0F0F0',        # light fill behind hatch (or use (1,1,1,0) for transparent)
            edgecolor="#333333", #'#999999',    # hatch color follows edgecolor
            linewidth = 0,
            # hatch='///',
            # label='No data'
        )
    )

    # Admin-2 boundaries (thin white) & country outlines on top
    # exposure_dict['multi-exposure'].boundary.plot(ax=ax, color='white', linewidth=0.05, zorder=2)
    gadm_countries.boundary.plot(ax=ax, linewidth=0.3, edgecolor="#333333", zorder=3)

    # Extent & cosmetics
    xmin, ymin, xmax, ymax = gadm_countries.total_bounds
    pad_x = 0.02*(xmax-xmin); pad_y = 0.02*(ymax-ymin)
    ax.set_xlim(xmin-pad_x, xmax+pad_x)
    ax.set_ylim(ymin-pad_y, ymax+pad_y)
    ax.set_axis_off()
    plt.tight_layout()
    plt.axis('off') # we don't need axis with coordinates
    # ax.set_title('Bivariate Choropleth Middle Corridor Road Network')

    # Step 2: draw the legend

    # We're drawing a 3x3 "box" as 3 columns
    # The xmin and xmax arguments axvspan are defined to create equally sized small boxes

    img2 = fig # refer to the main figure
    ax2 = fig.add_axes([0.09, 0.37, 0.1, 0.1]) # add new axes to place the legend there
                                            # and specify its location 
    alpha = 1 # alpha argument to make it more/less transperent

    # Column 1
    ax2.axvspan(xmin=0, xmax=0.20, ymin=0, ymax=0.20, alpha=alpha, color=all_colors_list[0])
    ax2.axvspan(xmin=0, xmax=0.20, ymin=0.20, ymax=0.40, alpha=alpha, color=all_colors_list[1])
    ax2.axvspan(xmin=0, xmax=0.20, ymin=0.40, ymax=0.60, alpha=alpha, color=all_colors_list[2])
    ax2.axvspan(xmin=0, xmax=0.20, ymin=0.60, ymax=0.80, alpha=alpha, color=all_colors_list[3])
    ax2.axvspan(xmin=0, xmax=0.20, ymin=0.80, ymax=1, alpha=alpha, color=all_colors_list[4])

    # Column 2
    ax2.axvspan(xmin=0.2, xmax=0.40, ymin=0, ymax=0.20, alpha=alpha, color=all_colors_list[5])
    ax2.axvspan(xmin=0.2, xmax=0.40, ymin=0.20, ymax=0.40, alpha=alpha, color=all_colors_list[6])
    ax2.axvspan(xmin=0.2, xmax=0.40, ymin=0.40, ymax=0.60, alpha=alpha, color=all_colors_list[7])
    ax2.axvspan(xmin=0.2, xmax=0.40, ymin=0.60, ymax=0.80, alpha=alpha, color=all_colors_list[8])
    ax2.axvspan(xmin=0.2, xmax=0.40, ymin=0.80, ymax=1, alpha=alpha, color=all_colors_list[9])

    # Column 3
    ax2.axvspan(xmin=0.4, xmax=0.6, ymin=0, ymax=0.20, alpha=alpha, color=all_colors_list[10])
    ax2.axvspan(xmin=0.4, xmax=0.6, ymin=0.20, ymax=0.40, alpha=alpha, color=all_colors_list[11])
    ax2.axvspan(xmin=0.4, xmax=0.6, ymin=0.40, ymax=0.60, alpha=alpha, color=all_colors_list[12])
    ax2.axvspan(xmin=0.4, xmax=0.6, ymin=0.60, ymax=0.80, alpha=alpha, color=all_colors_list[13])
    ax2.axvspan(xmin=0.4, xmax=0.6, ymin=0.80, ymax=1, alpha=alpha, color=all_colors_list[14])

    # Column 4
    ax2.axvspan(xmin=0.6, xmax=0.8, ymin=0, ymax=0.20, alpha=alpha, color=all_colors_list[15])
    ax2.axvspan(xmin=0.6, xmax=0.8, ymin=0.20, ymax=0.40, alpha=alpha, color=all_colors_list[16])
    ax2.axvspan(xmin=0.6, xmax=0.8, ymin=0.40, ymax=0.60, alpha=alpha, color=all_colors_list[17])
    ax2.axvspan(xmin=0.6, xmax=0.8, ymin=0.60, ymax=0.80, alpha=alpha, color=all_colors_list[18])
    ax2.axvspan(xmin=0.6, xmax=0.8, ymin=0.80, ymax=1, alpha=alpha, color=all_colors_list[19])

    # Column 5
    ax2.axvspan(xmin=0.8, xmax=1.0, ymin=0, ymax=0.20, alpha=alpha, color=all_colors_list[20])
    ax2.axvspan(xmin=0.8, xmax=1.0, ymin=0.20, ymax=0.40, alpha=alpha, color=all_colors_list[21])
    ax2.axvspan(xmin=0.8, xmax=1.0, ymin=0.40, ymax=0.60, alpha=alpha, color=all_colors_list[22])
    ax2.axvspan(xmin=0.8, xmax=1.0, ymin=0.60, ymax=0.80, alpha=alpha, color=all_colors_list[23])
    ax2.axvspan(xmin=0.8, xmax=1.0, ymin=0.80, ymax=1, alpha=alpha, color=all_colors_list[24])


    # Step 3: annoate the legend
    ax2.tick_params(axis='both', which='both', length=0) # remove ticks from the big box
    ax2.axis('off'); # turn off its axis
    ax2.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)) # draw arrow for x 
    ax2.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)) # draw arrow for y 
    ax2.text(s='Exposed assets', x=-0.04, y=-0.18, fontsize=7,) # annotate x axis x=-0.2, y=-0.25,
    ax2.text(s='Hazards', x=-0.25, y=0.1, rotation=90, fontsize=7,); # annotate y axis

    plt.tight_layout()
    plt.savefig(figures_path / f"figure_3_{ci_system}.png", bbox_inches='tight', dpi=300)
    plt.close(fig) 


################################################################
                    ## Figure 3 global distribution of infrastructure exposure multi-exposure, multi-hazard##
################################################################

ci_systems_lst = ['transportation', 'energy', 'waste', 'water', 'telecommunication', 'healthcare', 'education']
systems_exposure_dict = {}

for ci_system in ci_systems_lst: #'transportation',
    # Read the Excel file into a dictionary
    exposure_sheet_lst = hazard_types + ['multi-exposure']
    exposure_sheets_dict = pd.read_excel(output_risk_path / f"global_{ci_system}_exposure.xlsx", sheet_name=None)
    exposure_dict = {key: exposure_sheets_dict[key] for key in exposure_sheet_lst if key in exposure_sheets_dict} # Filter the sheets to match the hazard types

    # Read subnational data
    for key in exposure_dict: 
        exposure_dict[key] = exposure_dict[key].merge(subnational_df[['GID_2', 'geometry']], on='GID_2', how='left') # Merge subnational_df with each DataFrame in the hazard_dict)
        exposure_dict[key] = gpd.GeoDataFrame(exposure_dict[key], geometry='geometry').set_crs(4326, inplace=True)

    # get a 'ci_system_exposure_count' 
    df = exposure_dict['multi-exposure'].copy()
    # 1) Detect columns
    exposure_cols = [c for c in df.columns if c.endswith('_exposure_count')]
    if not exposure_cols:
        raise ValueError("No '*_exposure_count' columns found in exposure_dict['multi-exposure'].")

    # 2) (Optional) coerce to numeric (in case any are strings)
    df[exposure_cols] = df[exposure_cols].apply(pd.to_numeric, errors='coerce')

    # 3) ci_system_exposure_count = sum across all assets' exposure counts
    df[f'{ci_system}_exposure_count'] = df[exposure_cols].sum(axis=1, skipna=True)

    # 4) (Optional) cast counts to integers (nullable) if you prefer ints
    for col in [f'{ci_system}_exposure_count']:
        df[col] = df[col].round().astype('Int64')

    # Put the result back 
    exposure_dict['multi-exposure'] = df

    systems_exposure_dict[ci_system] = exposure_dict    

# start from one system just to get the index / admin columns
summary_bivariate_df = systems_exposure_dict[ci_systems_lst[0]]['multi-exposure'][['GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'GID_2', 'NAME_2', 'geometry']].copy()
# merge exposure counts from each CI system
for ci_system in ci_systems_lst:
    exposure_col = f"{ci_system}_exposure_count"
    summary_bivariate_df = summary_bivariate_df.merge(systems_exposure_dict[ci_system]['multi-exposure'][['GID_0', 'GID_1', 'GID_2', exposure_col]], on=['GID_0', 'GID_1', 'GID_2'], how='left'    )
# replace NaNs with 0 (important before summing)
exposure_cols = [f"{ci_system}_exposure_count" for ci_system in ci_systems_lst]
summary_bivariate_df[exposure_cols] = summary_bivariate_df[exposure_cols].fillna(0)
# total CI exposure across all systems
summary_bivariate_df['CI_exposure_count'] = summary_bivariate_df[exposure_cols].sum(axis=1)

# check hazard exposure and create dct
hazard_collection_dct = {}

for ci_system in ci_systems_lst:
    # retrieve hazard exposure per admin
    hazard_collection = subnational_df[['GID_0', 'COUNTRY', 'GID_2']].copy()
    for hazard_type in hazard_types:
        temp_df = systems_exposure_dict[ci_system][hazard_type].copy()

        # find all exposure columns that start with 'eae'
        eae_cols = [c for c in temp_df.columns if c.startswith('eae')]
        if not eae_cols:
            # no eae columns -> everything is 0 for this hazard
            hazard_collection[hazard_type] = 0
            continue

        # flag = 1 if any eae_* > 0 on the row, else 0
        # (fill NaNs with 0 before comparison)
        has_any = temp_df[eae_cols].fillna(0).gt(0).any(axis=1).astype('int8')

        # index by GID_2 for quick alignment
        if 'GID_2' not in temp_df.columns:
            raise KeyError(f"'GID_2' not found in systems_exposure_dict[{ci_system}][{hazard_type}]")
        haz_series = pd.Series(has_any.values, index=temp_df['GID_2'])

        # map onto hazard_collection by GID_2; missing -> 0
        hazard_collection[hazard_type] = hazard_collection['GID_2'].map(haz_series).fillna(0).astype('int8')

    # (optional) add a column indicating presence of ANY hazard exposure
    hazard_cols = [h for h in hazard_types if h in hazard_collection.columns]
    hazard_collection['multiple_hazard'] = (
        hazard_collection[hazard_types].fillna(0).sum(axis=1).astype('int16'))

    hazard_collection_dct[ci_system] = hazard_collection
    hazard_collection.head()

# put hazard_count results in summary_bivariate_df
# 1) initialize hazard indicator columns in the summary df
for hazard_type in hazard_types: summary_bivariate_df[f"{hazard_type}_exposure_count"] = 0

# 2) compute hazard indicator per hazard across CI systems
for hazard_type in hazard_types:
    temp_df = pd.DataFrame(index=summary_bivariate_df.index)  # temp_df holds per-system hazard exposure columns for this hazard
    for ci_system in ci_systems_lst:
        col = f"{ci_system}_{hazard_type}_exposure_count"
        s = hazard_collection_dct[ci_system][hazard_type] # pull the series (could be counts, booleans, etc.)
        temp_df[col] = pd.Series(s).reindex(summary_bivariate_df.index).fillna(0) # align to summary index, treat missing as 0

    temp_df["sum"] = temp_df.sum(axis=1) # sum across systems for this hazard
    temp_df[f"{hazard_type}_exposure_count"] = (temp_df["sum"] > 0).astype(int) # binary: 1 if any system has exposure > 0, else 0

    # write into summary
    summary_bivariate_df[f"{hazard_type}_exposure_count"] = temp_df[f"{hazard_type}_exposure_count"].values

# get a ci_system_hazard_count'
# 1) initialize hazard indicator columns in the summary df
for hazard_type in hazard_types: summary_bivariate_df[f"{ci_system}_hazard_count"] = 0

# 2) for each CI system: count how many hazards are present (>0) per admin unit
for ci_system in ci_systems_lst:
    temp_df = pd.DataFrame(index=summary_bivariate_df.index)
    for hazard_type in hazard_types:
        col = f"{hazard_type}_present"
        s = hazard_collection_dct[ci_system][hazard_type]
        temp_df[col] = (pd.Series(s).reindex(summary_bivariate_df.index).fillna(0))

    # convert to binary presence per hazard, then sum across hazards
    hazard_presence = (temp_df > 0).astype(int)
    summary_bivariate_df[f"{ci_system}_hazard_count"] = hazard_presence.sum(axis=1).values

# get final global hazard exposure across ci ssytems
hazard_cols = [f"{hazard_type}_exposure_count" for hazard_type in hazard_types]
summary_bivariate_df['global_hazard_count'] = (summary_bivariate_df[hazard_cols] > 0).sum(axis=1)


#applying the filter function to 'Salary' column 
summary_bivariate_df['CI_exposure_cat'] = (summary_bivariate_df['CI_exposure_count']).apply(filter_assets)
summary_bivariate_df['global_hazard_count_cat'] = summary_bivariate_df['global_hazard_count'].apply(filter_hazard)
# Combine x and y codes to create Bi_Class
summary_bivariate_df['Bi_Class'] = summary_bivariate_df['CI_exposure_cat'] + summary_bivariate_df['global_hazard_count_cat']

############
# link color map to unique categories found in dataset
############

# Define your categories
categories = [f"{i}{chr(65+j)}" for i in range(1, 6) for j in range(5)]

# Create a dictionary that maps each category to a color
category_color_map = dict(zip(categories, all_colors_list))

unique_categories = summary_bivariate_df['Bi_Class'].unique() # Extract unique categories from the dataframe
colors = [color for category, color in category_color_map.items() if category in unique_categories] # Filter category_color_map based on unique categories

cmap = mcolors.ListedColormap(colors)

# Extent (with padding)
xmin, ymin, xmax, ymax = subnational_df.total_bounds
pad_x = 0.02 * (xmax - xmin)
pad_y = 0.02 * (ymax - ymin)
xlim = (xmin - pad_x, xmax + pad_x)
ylim = (ymin - pad_y, ymax + pad_y)


mpl.rcParams['hatch.linewidth'] = 0.2   # default is ~1.0

fig, ax = plt.subplots(figsize=(8,8))

# Background national polygons (under everything)
subnational_df.plot(ax=ax, facecolor ="#D9D9D9", edgecolor='none', zorder=0) #'#edede9'

# Choropleth with hatched "no data" areas
summary_bivariate_df.plot(
    ax=ax,
    column='Bi_Class',
    cmap=cmap,
    categorical=True,
    legend=False,
    zorder=1,
    missing_kwds=dict(
        color="#D9D9D9", #'#F0F0F0',        # light fill behind hatch (or use (1,1,1,0) for transparent)
        edgecolor="#333333", #'#999999',    # hatch color follows edgecolor
        linewidth = 0,
        # hatch='///',
        # label='No data
    ))

# Admin-2 boundaries (thin white) & country outlines on top
gadm_countries.boundary.plot(ax=ax, linewidth=0.3, edgecolor="#333333", zorder=3)

# Extent & cosmetics
xmin, ymin, xmax, ymax = gadm_countries.total_bounds
pad_x = 0.02*(xmax-xmin); pad_y = 0.02*(ymax-ymin)
ax.set_xlim(xmin-pad_x, xmax+pad_x)
ax.set_ylim(ymin-pad_y, ymax+pad_y)
ax.set_axis_off()
plt.tight_layout()
plt.axis('off') # we don't need axis with coordinates
# ax.set_title('Bivariate Choropleth Middle Corridor Road Network')

# Step 2: draw the legend

# We're drawing a 3x3 "box" as 3 columns
# The xmin and xmax arguments axvspan are defined to create equally sized small boxes

img2 = fig # refer to the main figure
ax2 = fig.add_axes([0.09, 0.37, 0.1, 0.1]) # add new axes to place the legend there
                                        # and specify its location 
alpha = 1 # alpha argument to make it more/less transperent

# Column 1
ax2.axvspan(xmin=0, xmax=0.20, ymin=0, ymax=0.20, alpha=alpha, color=all_colors_list[0])
ax2.axvspan(xmin=0, xmax=0.20, ymin=0.20, ymax=0.40, alpha=alpha, color=all_colors_list[1])
ax2.axvspan(xmin=0, xmax=0.20, ymin=0.40, ymax=0.60, alpha=alpha, color=all_colors_list[2])
ax2.axvspan(xmin=0, xmax=0.20, ymin=0.60, ymax=0.80, alpha=alpha, color=all_colors_list[3])
ax2.axvspan(xmin=0, xmax=0.20, ymin=0.80, ymax=1, alpha=alpha, color=all_colors_list[4])

# Column 2
ax2.axvspan(xmin=0.2, xmax=0.40, ymin=0, ymax=0.20, alpha=alpha, color=all_colors_list[5])
ax2.axvspan(xmin=0.2, xmax=0.40, ymin=0.20, ymax=0.40, alpha=alpha, color=all_colors_list[6])
ax2.axvspan(xmin=0.2, xmax=0.40, ymin=0.40, ymax=0.60, alpha=alpha, color=all_colors_list[7])
ax2.axvspan(xmin=0.2, xmax=0.40, ymin=0.60, ymax=0.80, alpha=alpha, color=all_colors_list[8])
ax2.axvspan(xmin=0.2, xmax=0.40, ymin=0.80, ymax=1, alpha=alpha, color=all_colors_list[9])

# Column 3
ax2.axvspan(xmin=0.4, xmax=0.6, ymin=0, ymax=0.20, alpha=alpha, color=all_colors_list[10])
ax2.axvspan(xmin=0.4, xmax=0.6, ymin=0.20, ymax=0.40, alpha=alpha, color=all_colors_list[11])
ax2.axvspan(xmin=0.4, xmax=0.6, ymin=0.40, ymax=0.60, alpha=alpha, color=all_colors_list[12])
ax2.axvspan(xmin=0.4, xmax=0.6, ymin=0.60, ymax=0.80, alpha=alpha, color=all_colors_list[13])
ax2.axvspan(xmin=0.4, xmax=0.6, ymin=0.80, ymax=1, alpha=alpha, color=all_colors_list[14])

# Column 4
ax2.axvspan(xmin=0.6, xmax=0.8, ymin=0, ymax=0.20, alpha=alpha, color=all_colors_list[15])
ax2.axvspan(xmin=0.6, xmax=0.8, ymin=0.20, ymax=0.40, alpha=alpha, color=all_colors_list[16])
ax2.axvspan(xmin=0.6, xmax=0.8, ymin=0.40, ymax=0.60, alpha=alpha, color=all_colors_list[17])
ax2.axvspan(xmin=0.6, xmax=0.8, ymin=0.60, ymax=0.80, alpha=alpha, color=all_colors_list[18])
ax2.axvspan(xmin=0.6, xmax=0.8, ymin=0.80, ymax=1, alpha=alpha, color=all_colors_list[19])

# Column 5
ax2.axvspan(xmin=0.8, xmax=1.0, ymin=0, ymax=0.20, alpha=alpha, color=all_colors_list[20])
ax2.axvspan(xmin=0.8, xmax=1.0, ymin=0.20, ymax=0.40, alpha=alpha, color=all_colors_list[21])
ax2.axvspan(xmin=0.8, xmax=1.0, ymin=0.40, ymax=0.60, alpha=alpha, color=all_colors_list[22])
ax2.axvspan(xmin=0.8, xmax=1.0, ymin=0.60, ymax=0.80, alpha=alpha, color=all_colors_list[23])
ax2.axvspan(xmin=0.8, xmax=1.0, ymin=0.80, ymax=1, alpha=alpha, color=all_colors_list[24])

# --- Bin labels to display on the legend axes ---
# x_labels = ["≤100", "100–1k", "1k–10k", "10k–100k", "100k–300k"]
x_labels = ["100", "1k", "10k", "100k", "300k"]
y_labels = ["1", "2", "3", "4", "≥5"]
centers = [0.1, 0.3, 0.5, 0.7, 0.9] # centers of the 5 bins in your [0,1] legend box
adjusted_positions = [0.2, 0.4, 0.6, 0.8, 1.0]

# X-axis (assets) labels under the arrow
for cx, lab in zip(adjusted_positions, x_labels):
    ax2.text(cx, -0.06, lab, ha="center", va="top", rotation=90, fontsize=7)

# Y-axis (hazards) labels left of the arrow
for cy, lab in zip(centers, y_labels):
    ax2.text(-0.06, cy, lab, ha="right", va="center", fontsize=7)

# Step 3: annoate the legend
ax2.tick_params(axis='both', which='both', length=0) # remove ticks from the big box
ax2.axis('off'); # turn off its axis
ax2.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)) # draw arrow for x 
ax2.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)) # draw arrow for y 
# ax2.text(s='Exposed assets', x=-0.04, y=-0.18, fontsize=7,) # annotate x axis x=-0.2, y=-0.25,
# ax2.text(s='Hazards', x=-0.25, y=0.1, rotation=90, fontsize=7,); # annotate y axis
ax2.text(s='Exposed assets', x=-0.0, y=-0.5, fontsize=7,) # annotate x axis x=-0.2, y=-0.25,
ax2.text(s='Hazards', x=-0.35, y=0.1, rotation=90, fontsize=7,); # annotate y axis x=-0.5, y=0.4

plt.tight_layout()
plt.savefig(figures_path / f"figure_3_multihazard_multici.png", bbox_inches='tight', dpi=300)
plt.close(fig) 

print('Finished making figure 3 multi-level, multi-hazard')


################################################################
                    ## Create figure 4 global distribution of dominant hazard and infrastructure damage ##
################################################################

# check for each hazard which hazard is dominant in an area and create new df
dom_haz_df = hazard_dict['fluvial'][['GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NAME_2','GID_2', 'geometry']]
for hazard_type in hazard_types:
    temp_df = hazard_dict[hazard_type][['GID_2', 'total_Q2']]
    temp_df = temp_df.rename(columns={'total_Q2': f'{hazard_type}_Q2'})
    dom_haz_df = dom_haz_df.merge(temp_df, on='GID_2', how='left')

check_columns = [f'{hazard_type}_Q2' for hazard_type in hazard_types] # List of hazard columns

# Find the column with the maximum value for each row and assign the corresponding hazard type
dom_haz_df['dominant_hazard'] = dom_haz_df[check_columns].idxmax(axis=1)
dom_haz_df['dominant_hazard'] = dom_haz_df['dominant_hazard'].str.replace('_Q2', '', regex=False)

# check for which sub system is dominant in an area and create new df
dom_sys_df = hazard_dict['multi-hazard'][['GID_0', 'COUNTRY', 'GID_1', 'NAME_1', 'NAME_2','GID_2', 'geometry']]
sub_systems_lst = [key for category in cis_dict.values() for key in category.keys()]
check_columns = [f'{sub_system}_Q2' for sub_system in sub_systems_lst] # List of hazard columns

# Find the column with the maximum value for each row and assign the corresponding hazard type
dom_sys_df['dominant_sub_system'] = hazard_dict['multi-hazard'][check_columns].idxmax(axis=1)
dom_sys_df['dominant_sub_system'] = dom_sys_df['dominant_sub_system'].str.replace('_Q2', '', regex=False)


# palette options: https://coolors.co/palette/ef476f-f78c6b-ffd166-83d483-06d6a0-0cb0a9-118ab2-073b4c
# https://coolors.co/palette/734f5a-264653-2a9d8f-e9c46a-f4a261-e76f51-941c2f-c05761

# ---- Label maps ----
hazard_label_map = {
    "fluvial": "Fluvial flooding",
    "pluvial": "Pluvial flooding",
    "coastal": "Coastal flooding",
    "earthquake": "Earthquakes",
    "landslide_rf": "Rainfall-triggered\nlandslides",
    "landslide_eq": "Earthquake-triggered\nlandslides",
    "windstorms": "Tropical cyclones",
}

# Sub-system label map (nice, publication-ready)
subsys_label_map = {
    "power": "Power",
    "road": "Roads",
    "rail": "Railway",
    "air": "Air transport",
    "water_supply": "Water supply",
    "waste_solid": "Solid waste",
    "waste_water": "Wastewater",
    "telecom": "Telecommunications",
    "healthcare": "Healthcare",
    "education": "Education",
}


############
# plotting
#########

fig4 = plt.figure(constrained_layout=False, figsize=(18, 12))
gs = fig4.add_gridspec(2, 1, hspace=0.0)   # 👈 vertical spacing between rows

f4_ax1 = fig4.add_subplot(gs[0, 0])
f4_ax2 = fig4.add_subplot(gs[1, 0])

# --------------------------
# Panel A — Dominant hazard
# --------------------------
haz_order = list(hazard_colors.keys())
haz_colors = [hazard_colors[k] for k in haz_order]
haz_code_map = {k: i for i, k in enumerate(haz_order)}

dom_haz_df = dom_haz_df.copy()
dom_haz_df["dominant_hazard"] = dom_haz_df["dominant_hazard"].replace({"windstorm": "windstorms"})
dom_haz_df["dom_haz_code"] = dom_haz_df["dominant_hazard"].map(haz_code_map)

haz_cmap = mcolors.ListedColormap(haz_colors)

dom_haz_df.plot(column="dom_haz_code",
    ax=f4_ax1,
    cmap=haz_cmap,
    vmin=-0.5,
    vmax=len(haz_order) - 0.5,
    legend=False,)

gadm_countries.boundary.plot(ax=f4_ax1,
    linewidth=0.3,
    color="#333333",
    zorder=10)

# Manual legend in dict order (includes all hazards)
haz_handles = [Patch(
        facecolor=hazard_colors[k],
        edgecolor="black",
        linewidth=0.6,
        label=hazard_label_map.get(k, k),)
    for k in haz_order]

f4_ax1.legend(handles=haz_handles,
    title="Dominant hazard type", #Primary hazard type of EAD
    loc="center left",
    bbox_to_anchor=(0.97, 0.5),   # 👈 right side, vertically centered
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    facecolor='#fdfdfd',          # light grey
    edgecolor="#666666",          # grey border
)


# -------------------------------
# Panel B — Dominant sub-system
# -------------------------------
sub_order = list(ci_subsys_colors.keys())
sub_colors = [ci_subsys_colors[k] for k in sub_order]
sub_code_map = {k: i for i, k in enumerate(sub_order)}

dom_sys_df = dom_sys_df.copy()
dom_sys_df["dom_sub_code"] = dom_sys_df["dominant_sub_system"].map(sub_code_map)

sub_cmap = mcolors.ListedColormap(sub_colors)

dom_sys_df.plot(column="dom_sub_code",
    ax=f4_ax2,
    cmap=sub_cmap,
    vmin=-0.5,
    vmax=len(sub_order) - 0.5,
    legend=False,)

gadm_countries.boundary.plot(ax=f4_ax2,
    linewidth=0.3,
    color="#333333",
    zorder=10)


sub_handles = [Patch(facecolor=ci_subsys_colors[k],
        edgecolor="black",
        linewidth=0.6,
        label=subsys_label_map.get(k, k),)
    for k in sub_order]

f4_ax2.legend(handles=sub_handles,
    title="Dominant subsystem", #Primary affected subsystem
    loc="center left",
    bbox_to_anchor=(0.97, 0.5),   
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    facecolor='#fdfdfd',
    edgecolor="#666666",)


# Clean axes
f4_ax1.set_axis_off()
f4_ax2.set_axis_off()

# Panel labels
panel_labels = ['(a)', '(b)']
axes = [f4_ax1, f4_ax2]

# Per-panel (x, y) coordinates in axes space
label_positions = [
    (0.045, 0.95),  # (a) 
    (0.045, 0.95),  # (b)
]

for ax, label, (x, y) in zip(axes, panel_labels, label_positions):
    ax.text(x, y, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight='bold',
        va='top',
        ha='left')


# plt.tight_layout()
plt.savefig(figures_path / f"figure_4.png", bbox_inches='tight', dpi=300)

print('Finished making figure 4')

###############################################################
                    # Create figure 5: global distribution of infrastructure damage ##
###############################################################
# color scheme
custom_colors = ["#D9D9D9",
    "#FED0BB",  # very light peach
    "#FCB9B2",  # light pink
    "#D96A6F",  # mid rose
    "#B23A48",  # saturated red
    "#8C2F39",  # dark red
    "#461220"   # very dark wine
]

custom_colors = [
    "#D9D9D9",  # zero (grey)
    "#FAD4C0",  # light peach (less pink)
    "#F1B6A6",  # soft coral (replaces light pink)
    "#C96A63",  # muted brick rose
    "#A33A3A",  # neutral deep red
    "#7A2F2F",  # dark red-brown
    "#451214"   # very dark red-brown
]

hazard_label_map = { "fluvial": "Fluvial flooding", "pluvial": "Pluvial flooding", "coastal": "Coastal flooding", "earthquake": "Earthquakes", "landslide_rf": "Rainfall-triggered landslides", "landslide_eq": "Earthquake-triggered landslides", "windstorm": "Tropical cyclones", "multi-hazard": "Multiple-hazard"}

country_col = "GID_0"   
risk_cols = ["total_min", "total_Q1", "total_Q2", "total_Q3", "total_max"]

# #############
# #option 
# #1 disyplay at national level
# ########
# print('Start making figure 5 option 1')
# hazard_dict_country = {}
# gadm_geom = gadm_countries[["GID_0", "geometry"]]

# for hazard, df in hazard_dict.items():
#     hazard_dict_country[hazard] = gpd.GeoDataFrame(df.groupby("GID_0", as_index=False)[risk_cols]
#         .sum().merge(gadm_geom, on="GID_0", how="left", validate="one_to_one"),
#         geometry="geometry", crs=gadm_countries.crs)

# # ---- Figure layout ----
# fig5 = plt.figure(constrained_layout=False, figsize=(18, 14))
# gs = fig5.add_gridspec(4, 2)
# axes = [fig5.add_subplot(gs[i, j]) for i in range(4) for j in range(2)]

# # ---- Global bins (USD) shared across ALL subplots ----
# bins = np.array([0, 1e-9, 1e7, 1e8, 1e9, 1e10, 1e11, 3e11])  # USD
# bin_labels = ["0",
#     ">0–10 million",
#     "10–100 million",
#     "100 million-1 billion",
#     "1–10 billion",
#     "10–100 billion",
#     ">100 billion"]

# # ---- Discrete colormap + categorical norm ----
# n_bins = len(bins) - 1
# discrete_cmap = ListedColormap(custom_colors)
# norm = mcolors.BoundaryNorm(boundaries=bins, ncolors=n_bins, clip=True)

# # ---- Plot maps (no per-panel legends) ----
# for ax, (hazard_type, gdf) in zip(axes, hazard_dict_country.items()):

#     # Plot directly in USD (no conversion)
#     gdf.plot(column= "total_Q2",
#         ax=ax,
#         cmap=discrete_cmap,
#         norm=norm,
#         legend=False,
#         linewidth=0.2,
#         edgecolor="none",
#         zorder=1)

#     # Country boundaries on top
#     gadm_countries.boundary.plot(ax=ax,
#         linewidth=0.3,
#         edgecolor="#333333",
#         zorder=10)

#     label = hazard_label_map.get(hazard_type, hazard_type)
#     ax.set_title(label, fontsize=12, pad=-500)
#     # ax.text(        0.02, 1.02, hazard_type,
#     #     transform=ax.transAxes,
#     #     ha="left",
#     #     va="bottom",
#     #     fontsize=12,
#     #     zorder=20)
    
#     ax.set_axis_off()

# # ---- Panel labels ----
# panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
# for ax, lab in zip(axes, panel_labels):
#     ax.text(0.02, 1.0, lab, # 0.02, 0.98 this was good without hazard type
#         transform=ax.transAxes,
#         ha="left", va="top",
#         fontsize=12,
#         fontweight='bold',
#         color="#333333",
#         zorder=20)

# # ---- Discrete legend (instead of a colorbar) ----
# legend_handles = [
#     mpatches.Patch(facecolor=discrete_cmap(i),
#         edgecolor="black",
#         linewidth=0.5,
#         label=lab) for i, lab in enumerate(bin_labels)]

# fig5.legend(handles=legend_handles,
#     title="Expected Annual Damages (USD)",
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.03),
#     ncol=7,
#     frameon=True,
#     fancybox=True,
#     shadow=True,
#     framealpha=0.95,
#     facecolor='#fdfdfd',
#     edgecolor="#666666")

# # ---- Spacing ----
# fig5.subplots_adjust(bottom=0.12, wspace=0.02, hspace=0.0)
 
# plt.tight_layout()
# plt.savefig(figures_path / f"figure_5a.png", bbox_inches='tight', dpi=300)
# plt.close(fig5) 

# print('Finished making figure 5 option 1')


# ##############
# #option
# #2 only affected areas
# ###########
print('Start making figure 5 option 2')

# Country base geometries (full countries)
base_countries = gadm_countries[[country_col, "geometry"]].copy()
base_countries = gpd.GeoDataFrame(base_countries, geometry="geometry", crs=gadm_countries.crs)

hazard_dict_country = {}
for hazard, gdf in hazard_dict.items():
    gdf = gdf.copy()
    # Keep only affected admin areas
    affected = gdf[(gdf[risk_cols].fillna(0) > 0).any(axis=1)].copy()
    if affected.empty:
        # ---- No affected areas: return full country geometries with zero risk ----
        out = base_countries.copy()
        for c in risk_cols:
            out[c] = 0.0
        hazard_dict_country[hazard] = out
    else:
        # ---- Affected areas exist: dissolve footprint and aggregate risk ----
        agg = affected.groupby(country_col, as_index=False)[risk_cols].sum()
        geom = affected.dissolve(by=country_col, as_index=False)[[country_col, "geometry"]]
        out = geom.merge(agg, on=country_col, how="left")
        out["geometry"] = out.geometry.buffer(0) # Optional: repair geometries after dissolve
        out = out[out.geometry.notna() & ~out.geometry.is_empty].copy() # Safety cleanup
        out = gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs or base_countries.crs) # Preserve CRS from the input/admin layer (important!)
        hazard_dict_country[hazard] = out

# ---- Figure layout ----
fig5 = plt.figure(constrained_layout=False, figsize=(18, 14))
gs = fig5.add_gridspec(4, 2)
axes = [fig5.add_subplot(gs[i, j]) for i in range(4) for j in range(2)]

# ---- Global bins (USD) shared across ALL subplots ----
bins = np.array([0, 1e-9, 1e7, 1e8, 1e9, 1e10, 1e11, 3e11])  # USD
bin_labels = ["0",
    ">0–10 million",
    "10–100 million",
    "100 million–1 billion",
    "1–10 billion",
    "10–100 billion",
    ">100 billion"]

# ---- Discrete colormap + categorical norm ----
n_bins = len(bins) - 1
discrete_cmap = ListedColormap(custom_colors)
norm = mcolors.BoundaryNorm(boundaries=bins, ncolors=n_bins, clip=True)

# ---- Plot maps (no per-panel legends) ----
for ax, (hazard_type, gdf) in zip(axes, hazard_dict_country.items()):

    # Plot directly in USD (no conversion)
    gdf.plot(column="total_Q2",
        ax=ax,
        cmap=discrete_cmap,
        norm=norm,
        legend=False,
        linewidth=0.2,
        edgecolor="none",
        zorder=1)

    # Grey background in back
    gadm_countries.plot(ax=ax,
        linewidth=0.3,   
        edgecolor="#333333",
        facecolor="#D9D9D9",
        zorder=0)
    subnational_df.plot(
        ax=ax,
        facecolor="none",   # or a color
        edgecolor="#F7F7F7",
        linewidth=0.01, #0.05, 
        zorder=8)
    # Country boundaries on top
    gadm_countries.boundary.plot(ax=ax,
        linewidth=0.3,
        edgecolor="#333333",
        zorder=10)

    label = hazard_label_map.get(hazard_type, hazard_type)
    ax.set_title(label, fontsize=12, pad=-500)

    ax.set_axis_off()

# ---- Panel labels ----
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
for ax, lab in zip(axes, panel_labels):
    ax.text(0.02, 1.05, lab, # 0.02, 0.98 this was good without hazard type
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=12,
        fontweight='bold',
        color="#333333",
        zorder=20)

# ---- Discrete legend (instead of a colorbar) ----
legend_handles = [
    mpatches.Patch(facecolor=discrete_cmap(i),
        edgecolor="black",
        linewidth=0.5,
        label=lab) for i, lab in enumerate(bin_labels[1:], start=1)]

fig5.legend(handles=legend_handles,
    title="Expected Annual Damages (USD)",
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),
    ncol=6,
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.95,
    facecolor='#fdfdfd',
    edgecolor="#666666")

# ---- Spacing ----
fig5.subplots_adjust(bottom=0.12, wspace=0.02, hspace=0)

plt.tight_layout()
plt.savefig(figures_path / f"figure_5b_with_adminbound.png", bbox_inches='tight', dpi=300)

print('Finished making figure 5 option 2')
