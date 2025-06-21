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
import sklearn
import spiceypy
import tqdm

# Auxiliary module that contains the apparent magnitude
from auxiliary import photometry

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
dataframe_filepath = data_dir / "enriched_granvik_model.parquet"

granvik_model_df = pd.read_parquet(dataframe_filepath)

# Compute the Apparent Magnitude & Corresponding Irradiance

# Get the Epoch in ET to compute the state vectors of ...
init_et = granvik_model_df.iloc[0]["epoch_et"]

# ... the Earth as seen from the Sun
sun2earth_position_vec = spiceypy.spkgps(targ=399,
                                         et=init_et,
                                         ref="ECLIPJ2000",
                                         obs=10)[0]

# Invert the Sun -> Earth vector to get the Earth -> Sun vector
earth2sun_position_vec = -1.0 * sun2earth_position_vec

# TQDM has a nice Pandas implementation that allows us to display the progress of "apply" functions
tqdm.tqdm.pandas()

# Compute the position vector of each NEO as seen from the Sun
granvik_model_df.loc[:, "sun2neo_position_vec"] = \
    granvik_model_df.progress_apply(lambda x: spiceypy.conics(elts=[x["Perihelion_km"],
                                                                    x["Ecc_"],
                                                                    x["Incl_rad"],
                                                                    x["LongAscNode_rad"],
                                                                    x["ArgP_rad"],
                                                                    x["MeanAnom_rad"],
                                                                    x["epoch_et"],
                                                                    gm_sun],
                                                              et=init_et)[:3],
                                    axis=1)

# To compute the apparent magnitude we need to re-compute the positional vectors and convert it to
# AU
granvik_model_df.loc[:, "neo2earth_position_vec"] = \
    granvik_model_df["sun2neo_position_vec"].progress_apply(lambda x: sun2earth_position_vec - x)
granvik_model_df.loc[:, "neo2sun_position_vec"] = \
    granvik_model_df["sun2neo_position_vec"].progress_apply(lambda x: -1.0 * x)

granvik_model_df.loc[:, "neo2earth_position_vec_AU"] = \
    granvik_model_df["neo2earth_position_vec"].progress_apply(lambda x: [spiceypy.convrt(k, "km", "AU") for k in x])
granvik_model_df.loc[:, "neo2sun_position_vec_AU"] = \
    granvik_model_df["neo2sun_position_vec"].progress_apply(lambda x: [spiceypy.convrt(k, "km", "AU") for k in x])

# Compute the apparent magnitude of each NEO (for the set epoch)
granvik_model_df.loc[:, "app_mag"] = \
    granvik_model_df.progress_apply(lambda x:
                                    photometry.hg_app_mag(abs_mag=x["H"],
                                                          vec_obj2obs=x["neo2earth_position_vec_AU"],
                                                          vec_obj2ill=x["neo2sun_position_vec_AU"],
                                                          slope_g=0.15),
                                    axis=1)

# Convert the apparent magnitude to the corresponding irradiance given in W/m^2
granvik_model_df.loc[:, "irradiance"] = \
    granvik_model_df.progress_apply(lambda x:
                                    photometry.appmag2irr(app_mag=x["app_mag"]),
                                    axis=1)

# We compute now the NEO's coordinates in a similar way using the dataframe and the apply function.
granvik_model_df.loc[:, "earth2neo_position_vec_AU"] = \
    granvik_model_df["neo2earth_position_vec_AU"].progress_apply(lambda x: -1.0 * np.array(x))

granvik_model_df.loc[:, "earth2neo_recrad"] = \
    granvik_model_df["earth2neo_position_vec_AU"].progress_apply(lambda x: spiceypy.recrad(x))

granvik_model_df.loc[:, "earth2neo_dist_AU"] = \
    granvik_model_df["earth2neo_recrad"].progress_apply(lambda x: x[0])

granvik_model_df.loc[:, "earth2neo_eclip_long"] = \
    granvik_model_df["earth2neo_recrad"].progress_apply(lambda x: x[1])

granvik_model_df.loc[:, "earth2neo_eclip_lat"] = \
    granvik_model_df["earth2neo_recrad"].progress_apply(lambda x: x[2])

# Add now the angular distance between the NEO and the Sun as seen from Earth
granvik_model_df.loc[:, "ang_dist_neo2sun_deg"] = \
    granvik_model_df["earth2neo_position_vec_AU"] \
        .progress_apply(lambda x: np.degrees(spiceypy.vsep(x, earth2sun_position_vec)))

# Clean-Up
granvik_model_df.drop(columns=["neo2earth_position_vec",
                               "neo2sun_position_vec",
                               "neo2earth_position_vec_AU",
                               "neo2sun_position_vec_AU",
                               "earth2neo_position_vec_AU",
                               "earth2neo_recrad"],
                      inplace=True)

# Store the final dataframe as a parquet file
dataframe_filepath = data_dir / "enriched_granvik_model_w_irradiance.parquet"

granvik_model_df.to_parquet(dataframe_filepath)
