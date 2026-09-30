"""
A script which converts data from MATLAB (.mat) format files into CSVs.

This script primarily exists to convert the data from the Todescato paper's 
GitHub repository to CSV files which can be used in this project.
"""

import os
import pandas as pd
import scipy.io
import numpy as np

# Read in Colorado weather dataset
colorado_mat = scipy.io.loadmat("./Code/Todescato-Code/data/datasets/colorado.mat")

# Extract data from .mat file
colorado_locations = pd.DataFrame(colorado_mat["stationsLocations"]).rename(columns = {0:'Longitude', 1:'Latitude'})

colorado_ids = colorado_mat["stationsId"][0].split('\n')[:-1]
colorado_ids = pd.DataFrame({"ID": colorado_ids})

colorado_measurements = pd.DataFrame(colorado_mat["measurementsPpt"])

# Merge and save dataframe
colorado_full = pd.concat([colorado_locations, colorado_ids, colorado_measurements], axis = 1)
colorado_full.to_csv("Code/data/colorado.csv", index = False, header = True)