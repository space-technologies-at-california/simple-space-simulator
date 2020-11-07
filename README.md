# simple-space-simulator

[![Build Status](https://travis-ci.org/space-technologies-at-california/simple-space-simulator.svg?branch=main)](https://travis-ci.org/space-technologies-at-california/simple-space-simulator)

This is the main repository for the Simple Space Simulator (or S-Cubed) python package.

To get started run `conda env create -f env.yml`. After creating the environment run 
`conda activate simple_space_simulator` and run `python3 setup.py install`.

## Documentation
This package follows the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) docstring guidelines.

## Linting 
This project uses flake8 which is installed in the conda environment. To
check for code issues run `flake8 path/to/code/to/check.py --max-line-length 110`. To automatically 
reformat code run `autopep8 --in-place --recursive --max-line-length 110 .` which is also installed automatically into you conda environment.
The build will not succeed unless code has no flake8 errors.