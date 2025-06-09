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

# To enable interactivity, we need ipywidgets
import ipywidgets

# Append to root directory of this repository
sys.path.append("../")

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

print(neo_df.head(5))
print(f"No of NEOs: {neo_df.shape[0]}")

# Load SPICE kernels
spiceypy.furnsh("./kernels/spk/de432s.bsp")
spiceypy.furnsh("./kernels/lsk/naif0012.tls")
spiceypy.furnsh("./kernels/pck/gm_de431.tpc")

# Get the G*M value of the Sun
_, gm_sun_pre = spiceypy.bodvcd(bodyid=10, item='GM', maxn=1)
gm_sun = gm_sun_pre[0]

# Determine today's datetime
curr_time_et = spiceypy.utc2et(datetime.datetime.now().strftime("%Y-%m-%d"))

sun2earth_position_vec = spiceypy.spkgps(targ=399,
                                     et=curr_time_et,
                                     ref="ECLIPJ2000",
                                     obs=10)[0]
print(sun2earth_position_vec)

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
print(neo_df.head(5))

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
                                           et=curr_time_et)[:3],
                 axis=1)


# Let's take the position value of (433) Eros and compare it with NASA Horizons
print(neo_df.iloc[0]["sun2neo_position_vec"])

# To compute the apparent magnitude we need to re-compute the positional vectors and convert it to
# AU
neo_df.loc[:, "neo2earth_position_vec"] = \
    neo_df["sun2neo_position_vec"].apply(lambda x: sun2earth_position_vec - x)
neo_df.loc[:, "neo2sun_position_vec"] = \
    neo_df["sun2neo_position_vec"].apply(lambda x: -1.0 * x)

neo_df.loc[:, "neo2earth_position_vec_AU"] = \
    neo_df["neo2earth_position_vec"].apply(lambda x: [spiceypy.convrt(k, "km", "AU") for k in x])
neo_df.loc[:, "neo2sun_position_vec_AU"] = \
    neo_df["neo2sun_position_vec"].apply(lambda x: [spiceypy.convrt(k, "km", "AU") for k in x])

# Compute the apparent magnitude of each NEO for today!
neo_df.loc[:, "app_mag"] = \
    neo_df.apply(lambda x: photometry.hg_app_mag(abs_mag=x["AbsMag_"],
                                                 vec_obj2obs=x["neo2earth_position_vec_AU"],
                                                 vec_obj2ill=x["neo2sun_position_vec_AU"],
                                                 slope_g=x["SlopeParamG_"]), axis=1)

# Let's take a look. We can use e.g., Stellarium to get a "round about" feeling whether the
# app. mag. of Eros is correct
print(neo_df[["Name", "app_mag"]].head(5))

# Anyway ... we see that some app. mag. values appear to be outliers! The AbsMag_ does not appear
# to be realisitc (a simple placeholder?)
# https://newton.spacedys.com/neodys/index.php?pc=1.0
print(neo_df[["Name", "AbsMag_", "app_mag"]].sort_values(by="app_mag"))

# We set some NEOs we'd like to remove
neos_2_del = ["2020WM"]
neo_df = neo_df.loc[~neo_df["Name"].isin(neos_2_del)].copy()
neo_df.reset_index(drop=True, inplace=True)

# Now let's plot the distribution of the app. mag.
# Let's set a dark background
plt.style.use('dark_background')

# Set a default font size for better readability
plt.rcParams.update({'font.size': 14})

plt.figure(figsize=(12, 6))
counts, bins, _ = plt.hist(neo_df["app_mag"],
                           bins=np.arange(13.0, 36.0, 1.0),
                           color="yellow",
                           alpha=0.7)

plt.grid(linestyle="dashed", alpha=0.3)
plt.xlim(13, 35)
plt.xlabel("Apparent Magnitude")
plt.ylabel("Number of NEOs")
plt.ylim(0, 5500)

# Add the limiting magnitude of Pan-STARSS as a vertical line
# https://panstarrs.ifa.hawaii.edu/pswww/?page_id=34
plt.vlines(24, 0, 4000, linestyles="dashed", color="lightblue")
plt.text(24.5, 3800, "Pan-STARRS Lim. Mag.", color="lightblue")

plt.savefig(f'./images/07/apparent_magnitudes_histogram.png', dpi=300)

print(f"Number of NEOs Pan-STARRS could observe today: {int(sum(counts[:11]))}")
print(f"Number of NEOs Pan-STARRS could NOT observe today: {int(sum(counts[11:]))}")

# First a small reminder from session #8: https://www.youtube.com/watch?v=6GnzgzePYLg

# Use a dark background
plt.style.use('dark_background')

# Set a figure
plt.figure(figsize=(12, 8))

# Apply the aitoff projection and activate the grid
plt.subplot(projection="aitoff")
plt.grid(True)

# Set long. / lat. labels
plt.xlabel('Long. in deg')
plt.ylabel('Lat. in deg')

# Replace the standard x ticks (longitude) with the ecliptic coordinates
plt.xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0,
                             30, 60, 90, 120, 150]),
           labels=['150°', '120°', '90°', '60°', '30°', '0°',
                   '330°', '300°', '270°', '240°', '210°'])

plt.savefig(f'./images/08/ecliptic_coordinates.png', dpi=300)

# For orientation purposes we will plot the Sun and the corresponding opposition in an ecliptic
# coordinate system

# Compute the vector Earth -> Sun and compute the corresponding long and lat values
earth2sun_position_vec = -1.0 * sun2earth_position_vec
_, sun_ecl_long, sun_ecl_lat = spiceypy.recrad(earth2sun_position_vec)

# Convert the values to determine the "Opposition Direction"
sun_opp_ecl_long = (sun_ecl_long + np.pi) % (2.0 * np.pi)
sun_opp_ecl_lat = -1.0 * sun_ecl_lat

# We need to transform the longitude values for matplotlib
sun_ecl_long_4plot = \
    -1*((sun_ecl_long % np.pi) - np.pi) if sun_ecl_long > np.pi else -1*sun_ecl_long
sun_opp_ecl_long_4plot = \
    -1*((sun_opp_ecl_long % np.pi) - np.pi) if sun_opp_ecl_long > np.pi else -1*sun_opp_ecl_long

# Determine today's datetime
curr_time_utc = datetime.datetime.now().strftime("%Y-%m-%d")
curr_time_et = spiceypy.utc2et(curr_time_utc)

# Let's print the Sun's coordinates and compare it with values from Stellarium
print(f"The Sun's Ecliptic Longitude (at: {curr_time_utc}): "
      f"{round(np.degrees(sun_ecl_long), 2)} deg")
print(f"The Sun's Ecliptic Latitude (at: {curr_time_utc}): "
      f"{round(np.degrees(sun_ecl_lat), 2)} deg")

# Now we plot the sky plot with the Sun and the corresponding Opposition direction

# Use a dark background
plt.style.use('dark_background')

# Set a figure
plt.figure(figsize=(12, 8))

# Apply the aitoff projection and activate the grid
plt.subplot(projection="aitoff")
plt.grid(True)

# Set long. / lat. labels
plt.xlabel('Long. in deg')
plt.ylabel('Lat. in deg')

# Replace the standard x ticks (longitude) with the ecliptic coordinates
plt.xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0,
                             30, 60, 90, 120, 150]),
           labels=['150°', '120°', '90°', '60°', '30°', '0°',
                   '330°', '300°', '270°', '240°', '210°'])

# Add the Sun
plt.plot(sun_ecl_long_4plot,
         sun_ecl_lat,
         color="yellow",
         marker="o",
         markersize=15,
         alpha=0.5)

# Add the Opposition point
plt.plot(sun_opp_ecl_long_4plot,
         sun_opp_ecl_lat,
         color="teal",
         marker="s",
         markersize=10,
         alpha=0.8)

plt.savefig(f'./images/08/ecliptic_coordinates_with_sun.png', dpi=300)

# We compute now the NEO's coordinates in a similar way using the dataframe and the apply function.
neo_df.loc[:, "earth2neo_position_vec_AU"] = neo_df["neo2earth_position_vec_AU"].apply(lambda x: -1.0 * np.array(x))
neo_df.loc[:, "earth2neo_recrad"] = neo_df["earth2neo_position_vec_AU"].apply(lambda x: spiceypy.recrad(x))
neo_df.loc[:, "earth2neo_dist_AU"] = neo_df["earth2neo_recrad"].apply(lambda x: x[0])
neo_df.loc[:, "earth2neo_eclip_long"] = neo_df["earth2neo_recrad"].apply(lambda x: x[1])
neo_df.loc[:, "earth2neo_eclip_lat"] = neo_df["earth2neo_recrad"].apply(lambda x: x[2])

# Before we plot the data, we need to convert the longitude data into a
# matplotlib compatible format. We computed longitude values between 0 and
# 2*pi (360 degrees). matplotlib expects values between -pi and +pi. Further,
# sky maps count from 0 degrees longitude to the left. Thus we need also to
# invert the longitude values
neo_df.loc[:, "earth2neo_eclip_long_4plot_ecl"] = \
    neo_df["earth2neo_eclip_long"].apply(lambda x: -1*((x % np.pi) - np.pi) if x > np.pi else -1*x)

# Example from video:
# neo_sub_df = neo_df.loc[(neo_df["app_mag"] < 24) \
#                        & (neo_df["AbsMag_"] > 20) \
#                        & (neo_df["earth2neo_dist_AU"] < 0.25)].copy()
neo_sub_df = neo_df.loc[(neo_df["app_mag"] > 25)
                        & (neo_df["AbsMag_"] > 22)
                        & (neo_df["earth2neo_dist_AU"] < 1)].copy()

# Use a dark background
plt.style.use('dark_background')

# Set a figure
plt.figure(figsize=(12, 8))

# Apply the aitoff projection and activate the grid
plt.subplot(projection="aitoff")
plt.grid(True)

# Set long. / lat. labels
plt.xlabel('Long. in deg')
plt.ylabel('Lat. in deg')

plt.plot(neo_sub_df["earth2neo_eclip_long_4plot_ecl"],
         neo_sub_df["earth2neo_eclip_lat"],
         marker='.', linestyle='None', markersize=2, alpha=1, color="lightskyblue")

# Replace the standard x ticks (longitude) with the ecliptic coordinates
plt.xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0,30, 60, 90, 120, 150]),
           labels=['150°', '120°', '90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°', '210°'])

# Add the Sun
plt.plot(sun_ecl_long_4plot,
         sun_ecl_lat,
         color="yellow",
         marker="o",
         markersize=15,
         alpha=0.5)

# Add the Opposition point
plt.plot(sun_opp_ecl_long_4plot,
         sun_opp_ecl_lat,
         color="teal",
         marker="s",
         markersize=10,
         alpha=0.8)

plt.savefig(f'./images/08/ecliptic_coordinates_with_sun_and_neos.png', dpi=300)

# Using the SPICE functionvsep, we add a new row: the angular distance between the NEO(s) and the
# Sun.
#
# Background: NEO surveys, or "night time" telescopes (there are Sun-telescopes, so that's why I
# describe it so strangely), could theoretically operate during dusk and dawn. However, depending
# on the optics and camera system, even a bright sky could damage the instrument.
# Since we do not model a horizon, we take an angular distance between NEOs and the Sun as a rough
# "protection estimate"
neo_df.loc[:, "ang_dist_neo2sun_deg"] = \
    neo_df["earth2neo_position_vec_AU"].apply(lambda x:
                                              np.degrees(spiceypy.vsep(x, earth2sun_position_vec)))

# First we set up some nice interactive widgets

# This cell contains miscellaneous widget elements that are being used in our interactive plotting
# routine

# A date picker to e.g., compute the positions of the NEOs and the position of the Sun. However,
# here, it is disabled. Adding this functionality would be a nice "homework" for you!
date_picker_widget = ipywidgets.DatePicker(
    description='Date',
    diabled=True
)

# We add also a drop-down menu to select the NEO class
neo_class_widget = ipywidgets.Dropdown(
    options=['Amor', 'Apollo', 'Aten', 'Atira', 'Other', 'All'],
    description="NEO Class"
)

# 2 selection range slider to add filtering options for the apparent and absolute magnitude
app_mag_widget = ipywidgets.SelectionRangeSlider(
    options=range(5, 31),
    index=[0, 10],
    description='App. Mag'
)

abs_mag_widget = ipywidgets.SelectionRangeSlider(
    options=range(9, 35),
    index=[0, 10],
    description='Abs. Mag'
)

# Angular distance between Sun and NEO(s)
ang_dist_widget = ipywidgets.IntSlider(
    value=0,
    min=0,
    max=60,
    step=1,
    description="Sun-Dist",
)

# Set the dark mode and the font size and style
plt.style.use('dark_background')
plt.rc('font', family='serif', size=12)


# Set a function for the (interactive) plots
def plot_sky_map(date, neo_class, app_mag, abs_mag, ang_dist):
    # Set a figure
    plt.figure(figsize=(12, 8))

    # Apply the aitoff projection and activate the grid
    plt.subplot(projection="aitoff")
    plt.grid(True)

    # Set long. / lat. labels
    plt.xlabel('Long. in deg')
    plt.ylabel('Lat. in deg')

    # NEO Class filtering
    if neo_class == "All":
        _filtered_neo_df = neo_df.copy()
    else:
        _filtered_neo_df = neo_df.loc[neo_df["NEOClass"] == neo_class]

    # App. and Abs. Magnitude filtering
    _filtered_neo_df = _filtered_neo_df.loc[(_filtered_neo_df["app_mag"] >= app_mag[0])
                                            & (_filtered_neo_df["app_mag"] <= app_mag[1])]

    _filtered_neo_df = _filtered_neo_df.loc[(_filtered_neo_df["AbsMag_"] >= abs_mag[0])
                                            & (_filtered_neo_df["AbsMag_"] <= abs_mag[1])]

    # Angular distance filtering
    _filtered_neo_df = _filtered_neo_df.loc[_filtered_neo_df["ang_dist_neo2sun_deg"] > ang_dist]

    # Plotting the NEOs
    plt.plot(_filtered_neo_df["earth2neo_eclip_long_4plot_ecl"],
             _filtered_neo_df["earth2neo_eclip_lat"],
             marker='.',
             linestyle='None',
             markersize=2,
             alpha=1,
             color="lightskyblue")

    # Replace the standard x ticks (longitude) with the ecliptic coordinates
    plt.xticks(ticks=np.radians([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]),
               labels=['150°', '120°', '90°', '60°', '30°', '0°', '330°', '300°', '270°', '240°',
                       '210°'])

    # Add the Sun
    plt.plot(sun_ecl_long_4plot,
             sun_ecl_lat,
             color="yellow",
             marker="o",
             markersize=15,
             alpha=0.5)

    # Add the Opposition point
    plt.plot(sun_opp_ecl_long_4plot,
             sun_opp_ecl_lat,
             color="teal",
             marker="s",
             markersize=10,
             alpha=0.8)

    # Plot the total number of visible NEOs
    plt.title(f"#NEOs: {len(_filtered_neo_df)}", fontsize=12)

    plt.show()


# Create an interactive session!
# NOTE: It works only in Jupyter notebook
ipywidgets.interactive(plot_sky_map,
                       date=date_picker_widget,
                       neo_class=neo_class_widget,
                       app_mag=app_mag_widget,
                       abs_mag=abs_mag_widget,
                       ang_dist=ang_dist_widget)
