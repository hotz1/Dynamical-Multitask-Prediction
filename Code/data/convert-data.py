"""
A script which converts data from MATLAB (.mat) format files into CSVs.

This script exists to convert the data from the Todescato paper's GitHub repository 
to locally-stored CSV files which are usable with Python for this project.
"""

import pandas as pd
import scipy.io
import os
import glob

# Read in Colorado weather dataset
colorado_in_dir = "./Todescato-Code/data/datasets"
colorado_out_dir = "./Code/data/datasets"
os.makedirs(colorado_out_dir, exist_ok = True)

colorado_mat = scipy.io.loadmat(os.path.join(colorado_in_dir, "colorado.mat"))

# Extract data from .mat file
colorado_locations = pd.DataFrame(colorado_mat["stationsLocations"]).rename(columns = {0:'Longitude', 1:'Latitude'})
colorado_ids = colorado_mat["stationsId"][0].split('\n')[:-1]
colorado_ids = pd.DataFrame({"ID": colorado_ids})
colorado_measurements = pd.DataFrame(colorado_mat["measurementsPpt"])

# Merge and save dataframe
colorado_full = pd.concat([colorado_locations, colorado_ids, colorado_measurements], axis = 1)
colorado_full.to_csv(os.path.join(colorado_out_dir, "colorado.csv"), index = False, header = True)

# Read in precomputed approximations for the Gaussian time kernel
gtk_in_dir = "./Todescato-Code/data/gaussian_time_kernel_approximations"
gtk_out_dir = "./Code/data/gaussian_time_kernel_approximations"
os.makedirs(gtk_out_dir, exist_ok = True)

gtk_files = glob.glob(os.path.join(gtk_in_dir, "*.mat"))
for gtk in gtk_files:
    # Read in the data
    gtk_mat = scipy.io.loadmat(gtk)
    gtk_df = pd.DataFrame({"num": gtk_mat["num"].ravel(),
                           "den": gtk_mat["den"].ravel()})
    gtk_df.index.name = "coefficient"
    
    # Save locally as CSV
    gtk_specs = os.path.splitext(os.path.basename(gtk))[0] + ".csv"
    gtk_df.to_csv(os.path.join(gtk_out_dir, gtk_specs), index = True, header = True)
