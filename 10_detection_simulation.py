# Standard libraries
import datetime
import math
import pathlib
import sqlite3
import sys

# Installed libraries
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import spiceypy
import tqdm

# Auxiliary module that contains the apparent magnitude
from auxiliary import photometry

# Accessing the NEO database
database_dir = pathlib.Path("./databases/neos/")
database_file = pathlib.Path("neodys.db")
database_filepath = database_dir / database_file

# Establish a connection to the database and set a cursor
neodys_db_con = sqlite3.connect(database_filepath)
neodys_db_cur = neodys_db_con.cursor()

# Get all information from the DB. Since the DB is rather small, this won't cause any issues!
neo_df = pd.read_sql("SELECT * FROM main", neodys_db_con)

# Close the database.
neodys_db_con.close()

# Load SPICE kernels
spiceypy.furnsh("./kernels/spk/de432s.bsp")
spiceypy.furnsh("./kernels/lsk/naif0012.tls")
spiceypy.furnsh("./kernels/pck/gm_de431.tpc")

# Get the G*M value of the Sun
_, gm_sun_pre = spiceypy.bodvcd(bodyid=10, item='GM', maxn=1)
gm_sun = gm_sun_pre[0]

# For our computations we need to convert some values from AU to km and from deg to rad
neo_df.loc[:, "Perihel_km"] = neo_df["Perihel_AU"].apply(lambda x: spiceypy.convrt(x, "AU", "km"))
neo_df.loc[:, "Incl_rad"] = neo_df["Incl_deg"].apply(lambda x: math.radians(x))
neo_df.loc[:, "LongAscNode_rad"] = neo_df["LongAscNode_deg"].apply(lambda x: math.radians(x))
neo_df.loc[:, "ArgP_rad"] = neo_df["ArgP_deg"].apply(lambda x: math.radians(x))
neo_df.loc[:, "MeanAnom_rad"] = neo_df["MeanAnom_deg"].apply(lambda x: math.radians(x))
neo_df.loc[:, "Epoch_JD"] = neo_df["Epoch_MJD"].apply(lambda x: x + 2400000.5)
neo_df.loc[:, "Epoch_et"] = neo_df["Epoch_JD"].apply(lambda x: spiceypy.utc2et(str(x) + " JD"))

# Only non-negative values of perihelium are accepted
neo_df = neo_df[neo_df["Perihel_km"] > 0]

# Some simulation parameters (better way: creating a config file for maximum flexibility)
init_time_utc = "2010-01-01T00:00:00"
init_time_et = spiceypy.utc2et(init_time_utc)

# "Circle" around the opposition that is the detection area (in degrees)
opp_range = 15.0

# Minimum detection threshold (in magnitude)
mag_detec = 22.0

# Dataframe that stores the results
detected_neo_df = pd.DataFrame([])

# Simulation steps in hours
obs_steps = 24.0

# Observation range in hours
obs_range = 2.5 * 365.0 * 24.0

# Simulation loop. For better efficiency this can be done asynchronously
for time_step_h in tqdm.tqdm(np.arange(0, obs_range, 24.0)):

    # Computation time
    _time_et = init_time_et + (time_step_h * 3600.0)
    neo_df.loc[:, "et_of_detection"] = _time_et

    # Position vector of the Earth as seen from the Sun
    sun2earth_position_vec = spiceypy.spkgps(targ=399,
                                             et=_time_et,
                                             ref="ECLIPJ2000",
                                             obs=10)[0]
    earth2sun_position_vec = -1.0 * sun2earth_position_vec

    # Compute the position vector of each NEO as seen from the Sun
    neo_df.loc[:, "sun2neo_position_vec"] = \
        neo_df.apply(lambda x: spiceypy.conics(elts=[x["Perihel_km"],
                                                     x["Ecc_"],
                                                     x["Incl_rad"],
                                                     x["LongAscNode_rad"],
                                                     x["ArgP_rad"],
                                                     x["MeanAnom_rad"],
                                                     x["Epoch_et"],
                                                     gm_sun],
                                               et=_time_et)[:3],
                     axis=1)

    # To compute the apparent magnitude we need to re-compute the positional vectors and convert it
    # to AU
    neo_df.loc[:, "neo2earth_position_vec"] = \
        neo_df["sun2neo_position_vec"].apply(lambda x: sun2earth_position_vec - x)

    neo_df.loc[:, "neo2sun_position_vec"] = \
        neo_df["sun2neo_position_vec"].apply(lambda x: -1.0 * x)

    neo_df.loc[:, "neo2earth_position_vec_AU"] = \
        neo_df["neo2earth_position_vec"].apply(lambda x:
                                               [spiceypy.convrt(k, "km", "AU") for k in x])
    neo_df.loc[:, "neo2sun_position_vec_AU"] = \
        neo_df["neo2sun_position_vec"].apply(lambda x:
                                             [spiceypy.convrt(k, "km", "AU") for k in x])

    neo_df.loc[:, "earth2neo_position_vec_AU"] = \
        neo_df["neo2earth_position_vec_AU"].apply(lambda x: -1.0 * np.array(x))

    # Compute the apparent magnitude of each NEO
    neo_df.loc[:, "app_mag"] = \
        neo_df.apply(lambda x: photometry.hg_app_mag(abs_mag=x["AbsMag_"],
                                                     vec_obj2obs=x["neo2earth_position_vec_AU"],
                                                     vec_obj2ill=x["neo2sun_position_vec_AU"],
                                                     slope_g=x["SlopeParamG_"]), axis=1)

    # Compute the angular distance between NEO and opposition direction
    neo_df.loc[:, "ang_dist_neo2opp_deg"] = \
        neo_df["earth2neo_position_vec_AU"].apply(lambda x: np.degrees(spiceypy.vsep(x, -1.0 * earth2sun_position_vec)))

    # Get the detected NEOs
    detec_rows = neo_df.loc[(neo_df["app_mag"] <= mag_detec) & (neo_df["ang_dist_neo2opp_deg"] <= opp_range), :]

    # ... add them to the detected dataframe, remove them from the simulation
    detected_neo_df = pd.concat([detected_neo_df, detec_rows], ignore_index=True)
    neo_df.drop(detec_rows.index, inplace=True)

    # Dataframe empty? Quit
    if len(neo_df) == 0:
        break

# Convert the ET detection time to a human-readable format
detected_neo_df.loc[:, "utc_of_detection"] = \
    detected_neo_df["et_of_detection"].apply(lambda x: spiceypy.et2utc(x, "ISOC", 0))

detected_neo_df = detected_neo_df[['Name',
                                   'SemMajAxis_AU',
                                   'Ecc_',
                                   'Incl_deg',
                                   'LongAscNode_deg',
                                   'ArgP_deg',
                                   'AbsMag_',
                                   'SlopeParamG_',
                                   'Aphel_AU',
                                   'Perihel_AU',
                                   'NEOClass',
                                   'utc_of_detection']].copy()

# Store the results in a parquet file
pathlib.Path("results/simulation").mkdir(parents=True, exist_ok=True)
detected_neo_df.to_parquet(f"results/simulation/"
                           + f"UTC{init_time_utc.split("T")[0]}"
                           + f"_OppDist{opp_range}"
                           + f"_MagDetec{mag_detec}"
                           + f"_StepSize{obs_steps}"
                           + f"_HourObs{obs_range}.parquet")

# We also store the undetected NEOs (for the sake of completion)
undetected_neo_df = neo_df[['Name',
                                   'SemMajAxis_AU',
                                   'Ecc_',
                                   'Incl_deg',
                                   'LongAscNode_deg',
                                   'ArgP_deg',
                                   'AbsMag_',
                                   'SlopeParamG_',
                                   'Aphel_AU',
                                   'Perihel_AU',
                                   'NEOClass']].copy()

undetected_neo_df.to_parquet(f"results/simulation/"
                             + f"UNDETECTED_UTC{init_time_utc.split("T")[0]}"
                             + f"_OppDist{opp_range}"
                             + f"_MagDetec{mag_detec}"
                             + f"_StepSize{obs_steps}"
                             + f"_HourObs{obs_range}.parquet")
