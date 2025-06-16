# Standard libraries
import datetime
import math
import pathlib
import re
import requests
import sqlite3
import sys

# Installed libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import spiceypy
import tqdm

# We use the standard libraries gzip and shutil to unzip the file
import gzip
import shutil

# Auxiliary module that contains the apparent magnitude
from auxiliary import photometry

# Let's set a dark background
plt.style.use('dark_background')

# Set a default font size for better readability
plt.rcParams.update({'font.size': 14})

# Define the directory and name of the Granvik data. The data are originally stored as a gzip
# file and need to be unzipped
data_dir = pathlib.Path("raw_data/")
data_dir.mkdir(parents=True, exist_ok=True)
granvik_neo_model_file_zip = pathlib.Path("Granvik+_2018_Icarus.dat.gz")
granvik_neo_model_file = pathlib.Path("Granvik+_2018_Icarus.dat")

data_filepath = data_dir / granvik_neo_model_file
download_filepath = data_dir / granvik_neo_model_file_zip
print(f"Our Granvik et al. (2018) file path: {data_filepath}")

# Download the Granvik Model
dl_link = "https://www.mv.helsinki.fi" \
          "/home/mgranvik/data/Granvik+_2018_Icarus/Granvik+_2018_Icarus.dat.gz"
response = requests.get(dl_link)
download_file_path = pathlib.Path(download_filepath)
with download_file_path.open(mode="wb+") as file_obj:
    file_obj.write(response.content)

with gzip.open(download_file_path, 'rb') as f_in:
    with open(data_filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

granvik_model_df = pd.read_csv(data_filepath, names=["SemiMajorAxis_AU",
                                                     "Ecc_",
                                                     "Incl_deg",
                                                     "LongAscNode_deg",
                                                     "ArgP_deg",
                                                     "MeanAnom_deg",
                                                     "H"],
                               delim_whitespace=True)

print(granvik_model_df.head(5))

# Enrich the dataframe

# Load SPICE kernels
spiceypy.furnsh("./kernels/spk/de432s.bsp")
spiceypy.furnsh("./kernels/lsk/naif0012.tls")
spiceypy.furnsh("./kernels/pck/gm_de431.tpc")

# Get the G*M value of the Sun
_, gm_sun_pre = spiceypy.bodvcd(bodyid=10, item='GM', maxn=1)
gm_sun = gm_sun_pre[0]

# Add the perihelion and convert the angular values to deg
granvik_model_df.loc[:, "Perihelion_AU"] = (1.0 - granvik_model_df["Ecc_"]) \
                                               * granvik_model_df["SemiMajorAxis_AU"]

granvik_model_df.loc[:, "Incl_rad"] = np.radians(granvik_model_df["Incl_deg"])
granvik_model_df.loc[:, "LongAscNode_rad"] = np.radians(granvik_model_df["LongAscNode_deg"])
granvik_model_df.loc[:, "ArgP_rad"] = np.radians(granvik_model_df["ArgP_deg"])

# Convert AU to km
granvik_model_df.loc[:, "Perihelion_km"] = \
    granvik_model_df["Perihelion_AU"].apply(lambda x: spiceypy.convrt(x, "AU", "km"))
granvik_model_df.loc[:, "SemiMajorAxis_km"] = \
    granvik_model_df["SemiMajorAxis_AU"].apply(lambda x: spiceypy.convrt(x, "AU", "km"))

# Generate generic positions


# First, we need to define a function that converts true anomaly values to mean anomalies (since
# SPICE does not handle true anomalies
def true2mean(true_anom, ecc):
    atan2_x1 = -1.0 * np.sqrt(1.0 - ecc ** 2.0) * np.sin(true_anom)
    atan2_x2 = -1.0 * ecc - np.cos(true_anom)

    mean_anom = np.arctan2(atan2_x1, atan2_x2) \
                + np.pi \
                - ecc * ((np.sqrt(1.0 - ecc ** 2.0) * np.sin(true_anom))
                         / (1.0 + ecc * np.cos(true_anom)))

    return mean_anom


# Let's create an example plot with the true anomaly distribution
# Sample orbit:
sample_rp = spiceypy.convrt(0.5, "AU", "km")
sample_ecc = 0.7
sample_incl = 0.0
sample_lnode = 0.0
sample_argp = 0.0
sample_et = spiceypy.utc2et("2000-001T12:00:00")

# First we create a "complete" orbit that is used to explain our method:
mean_anomaly = np.radians(np.arange(0, 360, 0.1))

# List that will store the positions in 2D (X-Y plane)
sample_trajectory = []

for k in tqdm.tqdm(mean_anomaly):
    # Compute the state vector and store in positional values in X-Y direction, converted in AU
    temp_state = spiceypy.conics([sample_rp, sample_ecc, sample_incl, sample_lnode, sample_argp,
                                  k, sample_et, gm_sun], sample_et)
    sample_trajectory.append([spiceypy.convrt(temp_state[0], "km", "AU"),
                              spiceypy.convrt(temp_state[1], "km", "AU")])

sample_trajectory = np.array(sample_trajectory)

# Now we create 36 NEO positions using the mean anomaly. Basically ... we need the same code again!
mean_anomaly = np.radians(np.arange(0, 360, 10))

# List that will store the positions in 2D (X-Y plane)
mean_anom_10deg_trajectory = []

for k in tqdm.tqdm(mean_anomaly):
    # Compute the state vector and store in positional values in X-Y direction, converted in AU
    temp_state = spiceypy.conics([sample_rp, sample_ecc, sample_incl, sample_lnode, sample_argp,
                                  k, sample_et, gm_sun], sample_et)
    mean_anom_10deg_trajectory.append([spiceypy.convrt(temp_state[0], "km", "AU"),
                                       spiceypy.convrt(temp_state[1], "km", "AU")])

mean_anom_10deg_trajectory = np.array(mean_anom_10deg_trajectory)

# And finally a list with the true anomaly values!
# Now we create 36 NEO positions using the mean anomaly. Basically ... we need the same code again!
true_anomaly = np.radians(np.arange(0, 360, 10))
mean_from_true_anom = [true2mean(n, sample_ecc) for n in true_anomaly]

# List that will store the positions in 2D (X-Y plane)
true_anom_10deg_trajectory = []

for k in tqdm.tqdm(mean_from_true_anom):
    # Compute the state vector and store in positional values in X-Y direction, converted in AU
    temp_state = spiceypy.conics([sample_rp, sample_ecc, sample_incl, sample_lnode, sample_argp,
                                  k, sample_et, gm_sun], sample_et)
    true_anom_10deg_trajectory.append([spiceypy.convrt(temp_state[0], "km", "AU"),
                                       spiceypy.convrt(temp_state[1], "km", "AU")])

true_anom_10deg_trajectory = np.array(true_anom_10deg_trajectory)

# We plot now the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True)

ax1.set_title("Mean Anomaly Distances")
ax1.scatter(0, 0, color="yellow", s=25)
ax1.plot(sample_trajectory[:, 0],
        sample_trajectory[:, 1],
        alpha=0.8,
        color="white",
        linestyle="dotted")
ax1.plot(mean_anom_10deg_trajectory[:, 0],
        mean_anom_10deg_trajectory[:, 1],
        alpha=0.8,
        color="tab:green",
        linestyle="None",
        marker="^")
ax1.axis('equal')

ax2.set_title("True Anomaly Distances")
ax2.scatter(0, 0, color="yellow", s=25)
ax2.plot(sample_trajectory[:, 0],
        sample_trajectory[:, 1],
        alpha=0.8,
        color="white",
        linestyle="dotted")
ax2.plot(true_anom_10deg_trajectory[:, 0],
        true_anom_10deg_trajectory[:, 1],
        alpha=1,
        color="tab:orange",
        linestyle="None",
        marker="x")
ax2.axis('equal')

plt.tight_layout()

ax2.set_xlabel("Eclip. x in AU")
ax1.set_ylabel("Eclip. y in AU")
ax2.set_ylabel("Eclip. y in AU")

plt.savefig(f'./images/12/anomaly_distances.png', dpi=300)

# Create more NEOs

# First, let's add an epoch:
granvik_model_df.loc[:, "epoch_et"] = spiceypy.utc2et("2000-001T12:00:00")

# We remove the Mean Anomaly first and ...
granvik_model_df.drop(columns="MeanAnom_deg", inplace=True)

# ... replace it with an array
mean_anomaly_deg_array = np.arange(0, 360, 60)
granvik_model_df.loc[:, "MeanAnom_deg"] = [mean_anomaly_deg_array for i in granvik_model_df.index]

# Explode the list
granvik_model_df = granvik_model_df.explode("MeanAnom_deg").copy()
granvik_model_df.loc[:, "MeanAnom_rad"] = \
    granvik_model_df["MeanAnom_deg"].apply(lambda x: np.radians(x))
granvik_model_df.reset_index(drop=True, inplace=True)

print(granvik_model_df.head(5))

# Store the resulting dataframe
data_dir = pathlib.Path("results/Granvik")
data_dir.mkdir(parents=True, exist_ok=True)
dataframe_filepath = data_dir / "enriched_granvik_model.parquet"

# Save
granvik_model_df.to_parquet(dataframe_filepath)