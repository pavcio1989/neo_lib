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
from matplotlib import ticker
import numpy as np
import pandas as pd
import sklearn.neighbors
import spiceypy
import tqdm

# Auxiliary module that contains the apparent magnitude
from auxiliary import photometry

# Let's set a dark background
plt.style.use('dark_background')

# Set a default font size for better readability
plt.rcParams.update({'font.size': 14})

# Load SPICE kernels
spiceypy.furnsh("./kernels/spk/de432s.bsp")
spiceypy.furnsh("./kernels/lsk/naif0012.tls")
spiceypy.furnsh("./kernels/pck/gm_de431.tpc")

# Get the G*M value of the Sun
_, gm_sun_pre = spiceypy.bodvcd(bodyid=10, item='GM', maxn=1)
gm_sun = gm_sun_pre[0]

# Load the Granvik model
data_dir = pathlib.Path("results/Granvik")
data_dir.mkdir(parents=True, exist_ok=True)
dataframe_filepath = data_dir / "enriched_granvik_model_w_appmag.parquet"

granvik_model_df = pd.read_parquet(dataframe_filepath)

# As shown in previous notebooks, we need to transform the longitude values to plot them in a
# matplotlib map projection
granvik_model_df.loc[:, "earth2neo_eclip_long_4plot"] = (
    granvik_model_df["earth2neo_eclip_long"].apply(lambda x: -1*((x % np.pi) - np.pi) if x > np.pi else -1*x))

# Filter NEO model

# In this cell, you can filter the data as you wish!
filtered_granvik_model_df = granvik_model_df[granvik_model_df[f"ang_dist_neo2sun_deg"] >= 45.0].copy()
filtered_granvik_model_df = filtered_granvik_model_df[(filtered_granvik_model_df[f"app_mag"] <= 25.0)].copy()

# Please note: We obtain the latitude and THEN the longitude. Why? Take a look at the KDE cell.
neo_positions_coord = filtered_granvik_model_df[["earth2neo_eclip_lat",
                                                 "earth2neo_eclip_long_4plot"]].values

print(f"Number of de-enriched NEOs: {len(neo_positions_coord) / 6}")

# Get th ET and ...
time_et = granvik_model_df["epoch_et"].iloc[0]

# ... determine the positional vector of the Sun as seen from Earth and compute the corresponding
# sky coordinates
earth2sun_position_vec = spiceypy.spkgps(targ=10,
                                         et=time_et,
                                         ref="ECLIPJ2000",
                                         obs=399)[0]
_, sun_ecl_long, sun_ecl_lat = spiceypy.recrad(earth2sun_position_vec)

# Convert the values to determine the "Opposition Direction"
sun_opp_ecl_long = (sun_ecl_long + np.pi) % (2.0 * np.pi)
sun_opp_ecl_lat = -1.0 * sun_ecl_lat

# We need to transform the longitude values for matplotlib
sun_ecl_long_4plot = \
    -1*((sun_ecl_long % np.pi) - np.pi) if sun_ecl_long > np.pi else -1*sun_ecl_long
sun_opp_ecl_long_4plot = \
    -1*((sun_opp_ecl_long % np.pi) - np.pi) if sun_opp_ecl_long > np.pi else -1*sun_opp_ecl_long

# Set a figure
plt.figure(figsize=(12, 8))

# Apply the aitoff projection and activate the grid
plt.subplot(projection="aitoff")
plt.grid(True)

# Add the NEOs
plt.scatter(neo_positions_coord[:, 1],
            neo_positions_coord[:, 0],
            marker=".",
            s=1,
            alpha=0.2,
            color="white")

# Add the Sun
plt.plot(sun_ecl_long_4plot,
         sun_ecl_lat,
         color="yellow",
         marker="o",
         markersize=15,
         alpha=0.5)

# Add the Opposition point
plt.scatter(sun_opp_ecl_long_4plot,
            sun_opp_ecl_lat,
            color="orange",
            marker="s",
            s=200,
            alpha=0.8,
            edgecolor='black')

plt.xlabel("Ecl. long. in deg.")
plt.ylabel("Ecl. lat. in deg.")

# Replace the standard x ticks (longitude) with the ecliptic coordinates
plt.xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0,
                             30, 60, 90, 120, 150]),
           labels=['150°', '120°', '90°', '60°', '30°', '0°',
                   '330°', '300°', '270°', '240°', '210°'])

plt.savefig(f'./images/13/sky_coordinates_scatterplot.png', dpi=300)

# Kernel density estimator

# We apply a broad kernel with a size of 5 degrees to smooth the resulting distribution
kde = sklearn.neighbors.KernelDensity(bandwidth=np.radians(5),
                                      metric="haversine",
                                      kernel="exponential",
                                      algorithm="ball_tree")
kde.fit(neo_positions_coord)

# Compute the PDF in a long-lat mesh grid
sample_lat, sample_long = np.meshgrid(np.linspace(-0.5*np.pi, 0.5*np.pi, 100),
                                      np.linspace(-np.pi, np.pi, 100))
latlong = np.vstack([sample_lat.ravel(),
                     sample_long.ravel()]).T
sky_pdf = np.exp(kde.score_samples(latlong))
sky_pdf = sky_pdf.reshape(sample_lat.shape)

# We plot now the final figure of the sky-based PDF

# First we compute the resulting NEO density in 1/deg^2. Use the PDF, consider the number of all
# NEOs (len(neo_positions_coord)). This leads to the number of NEOs per Steradian. We apply now
# (180/pi)^2 to convert the result to NEOs/deg^2. However, since we enriched the NEO model in script
# number 12 by a factor of 6 we need to divide it by 6 to obtain the final density
neo_density = sky_pdf * len(neo_positions_coord)/(180/math.pi)**2 / 6

# Set a figure
plt.figure(figsize=(12, 8))

# Apply the aitoff projection and activate the grid
plt.subplot(projection="aitoff")
plt.grid(True)

# Create a color contour plot (filled)
CS = plt.contourf(sample_long,
                  sample_lat,
                  neo_density,
                  levels=np.linspace(0, np.max(neo_density), 100),
                  cmap=plt.cm.nipy_spectral)
cbar = plt.colorbar(CS, shrink=0.6)
tick_locator = ticker.MaxNLocator(nbins=5)
cbar.locator = tick_locator
cbar.update_ticks()
cbar.set_label("Nr. of NEOs per sq. deg.")

# Add the Sun
plt.plot(sun_ecl_long_4plot,
         sun_ecl_lat,
         color="yellow",
         marker="o",
         markersize=15,
         alpha=0.5)

# Add the Opposition point
plt.scatter(sun_opp_ecl_long_4plot,
            sun_opp_ecl_lat,
            color="black",
            marker="s",
            s=200,
            alpha=0.5,
            edgecolor='white')

# Replace the standard x ticks (longitude) with the ecliptic coordinates
plt.xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0,
                             30, 60, 90, 120, 150]),
           labels=['150°', '120°', '90°', '60°', '30°', '0°',
                   '330°', '300°', '270°', '240°', '210°'])

plt.savefig(f'./images/13/sky_based_kde.png', dpi=300)
