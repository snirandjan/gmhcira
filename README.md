# Global multiple hazard critical infrastructure risk analysis (GMHCIRA)
Python implementation of the Global Multiple-Hazard Critical Infrastructure Risk Analysis (GMHCIRA). This repository provides the code to:
- extract critical infrastructure assets from OpenStreetMap (OSM) data;
- calculate damages per return period for each hazard;
- calculate risk at the asset level;
- generate summary files containing exposure and risk estimates aggregated to the subnational level at the global scale.

The repository also provides Jupyter Notebooks and additional code to reproduce the figures and supplementary material presented in Nirandjan et al. (2026).

## Data used in Nirandjan et al. (2026)
- Critical infrastructure data are derived from OpenStreetMap (OSM) and can be freely downloaded from https://planet.openstreetmap.org/. The planet file used in Nirandjan et al. (2026) was downloaded on October 28, 2024. However, the latest release of the planet.osm.pbf file can also be used to run the code.
- The tropical cyclone hazard dataset is publicly available at: https://data.4tu.nl/datasets/0ea98bdd-5772-4da8-ae97-99735e891aff/4
- The earthquake hazard dataset and susceptibility maps for earthquake-triggered and rainfall-triggered landslides are publicly available at: https://giri.unepgrid.ch/map
- Global flood hazard data from the Fathom Global Flood Map 3.1 are used with permission from Fathom.
- The open-access vulnerability database (V1.1.0), including vulnerability curves and asset values, presented by Nirandjan et al. (2024), is available at: https://zenodo.org/records/10203846
- The 2024 World Bank income classification of countries is retrieved from: https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups
- GDP data are retrieved from the World Bank: https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
- GADM administrative boundaries (levels 0 and 2, version 4.1) are retrieved from: https://gadm.org/data.html

## Python requirements

Recommended option is to use a [miniconda](https://conda.io/miniconda.html)
environment to work in for this project, relying on conda to handle some of the
trickier library dependencies.

```bash

# Add conda-forge channel for extra packages
conda config --add channels conda-forge

# Create a conda environment for the project and install packages
conda env create -f environment.yml
conda activate py311
```
**Requirements:** [NumPy](http://www.numpy.org/), [pandas](https://pandas.pydata.org/), [geopandas](http://geopandas.org/), [matplotlib](https://matplotlib.org/)

## How to cite
If you use GMHCIRA in your work, please cite the corresponding paper:

Nirandjan, S., Verschuur, J., Wing, O.E.J., de Moel, H., Ward, P.J. Aerts, J.C.J.H. & Koks, E.E. A global-scale assessment of the multiple-hazard risk to critical infrastructure. Manuscript under review at _Environmental Research Letters_.


    @article{Nirandjan_GMHCIRA,
      title={A global-scale assessment of the multiple-hazard risk to critical infrastructure},
      author={Nirandjan, S. and Verschuur, J. and Wing, O.E.J. and de Moel, H. and Ward, P.J. and Aerts, J.C.J.H. and Koks, E.E.},
      journal={Environmental Research Letters},
      volume={},
      number={},
      pages={},
      year={2026}
    }

      
The following DOI can be cited for this repository:
[]

### License
Copyright (C) 2026 Sadhana Nirandjan & Elco Koks. All versions released under the [MIT license](LICENSE).
