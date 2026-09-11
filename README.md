# Global multiple hazard critical infrastructure risk analysis


Python implementation of global multiple hazard critical infrastructure risk analysis





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
If you use GMHCITRA in your work, please cite the corresponding paper:

Nirandjan, S., Verschuur, J., Wing, O.E.J., de Moel, H., Ward, P.J. Aerts, J.C.J.H. & Koks, E.E. A global-scale assessment of the multiple-hazard risk to critical infrastructure. Manuscript under review at _Environmental Research Letters_.


    @article{Nirandjan2022_CISI,
      title={A spatially-explicit harmonized global dataset of critical infrastructure},
      author={Nirandjan, S., Koks, E.E., Ward, P.J. and Aerts, J.C.J.H.},
      journal={Scientific Data},
      volume={9},
      number={150},
      pages={13},
      year={2022}
    }

      
The following DOI can be cited for this repository:
[]

### License
Copyright (C) 2026 Sadhana Nirandjan & Elco Koks. All versions released under the [MIT license](LICENSE).
