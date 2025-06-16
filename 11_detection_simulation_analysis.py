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

# Load the simulation results
detected_neo_df = pd.read_parquet("results/simulation/UTC2010-01-01_OppDist15.0_MagDetec22.0_StepSize24.0_HourObs21900.0.parquet")
undetected_neo_df = pd.read_parquet("results/simulation/UNDETECTED_UTC2010-01-01_OppDist15.0_MagDetec22.0_StepSize24.0_HourObs21900.0.parquet")

# Convert the detection time strings to a datetime object
detected_neo_df.loc[:, "utc_of_detection"] = detected_neo_df["utc_of_detection"].str[:10]
detected_neo_df.loc[:, "utc_of_detection"] = \
    detected_neo_df["utc_of_detection"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

# Compute the detection day (w.r.t. the initial observation datetime)
detected_neo_df.loc[:, "delta_day_wrt_init"] = \
    detected_neo_df["utc_of_detection"].apply(lambda x: (x - detected_neo_df["utc_of_detection"].iloc[0]).days)

# Let's set a dark background
plt.style.use('dark_background')

# Set a default font size for better readability
plt.rcParams.update({'font.size': 14})

# A cumulative plot that shows how many NEOs we have been detected in our opposition area over
# time
fig, ax = plt.subplots(figsize=(12, 8))

# Cumulative histogram in 30 days bins
ax.hist(detected_neo_df["delta_day_wrt_init"],
        bins=np.arange(0, np.max(detected_neo_df["delta_day_wrt_init"]) + 30, 30),
        cumulative=True,
        color="tab:orange",
        alpha=.7,
        rwidth=0.8)

# Formating and layout
ax.set_xlim(-30, np.max(detected_neo_df["delta_day_wrt_init"]) + 30)

ax.set_title("Detection history of all detected NEOs")
ax.set_xlabel("Observation range w.r.t. initial date time in days")
ax.set_ylabel("Number of detected NEOs")

ax.grid(linestyle="--", alpha=0.5)

plt.savefig(f'./images/11/all_detected_NEOs.png', dpi=300)

# Let's take a differentiated look at the size distribution of detected NEOs

# A second, overlapping histogram that show the larger NEOs (upper absolute magnitude limit of
# e.g., 18
absmag_limit = 18
temp_detected_neo_df = detected_neo_df.loc[detected_neo_df["AbsMag_"] <= absmag_limit].copy()

fig, ax = plt.subplots(figsize=(12, 8))

ax.hist(detected_neo_df["delta_day_wrt_init"],
        bins=np.arange(0, np.max(detected_neo_df["delta_day_wrt_init"]) + 30, 30),
        cumulative=True,
        color="tab:orange",
        alpha=.7,
        rwidth=0.8,
        label="All NEOs")

ax.hist(temp_detected_neo_df["delta_day_wrt_init"],
        bins=np.arange(0, np.max(temp_detected_neo_df["delta_day_wrt_init"]) + 30, 30),
        cumulative=True,
        color="purple",
        alpha=.9,
        rwidth=0.8,
        label=fr"NEOs $(H\leq{absmag_limit})$")

# Layout and formatting
ax.set_xlim(-30, np.max(detected_neo_df["delta_day_wrt_init"])+30)

ax.set_title("Detection history of NEOs")
ax.set_xlabel("Observation range w.r.t. initial date time in days")
ax.set_ylabel("Number of detected NEOs")

ax.legend(fancybox=True, loc="upper left")
ax.grid(linestyle="--", alpha=0.5)

plt.savefig(f'./images/11/all_detected_NEOs_labels.png', dpi=300)

# Here we take a look at the bias of certain elements. We compute the ratio between the
# detected "bins" and the undetected ones to see, if orbital elements, or magnitude are biased with
# our simulation setting(s)

# If you want to play around: please go ahead! The default parameters are:
elem = "AbsMag_"
bins_range = np.arange(20, 30 + 1, 1)
step_size = (bins_range[1]-bins_range[0])

# Use numpy to compute the detection ratio
hist_detec, _ = np.histogram(detected_neo_df[elem], bins=bins_range)
hist_undetec, bin_edges = np.histogram(undetected_neo_df[elem], bins=bins_range)

hist_ratio = hist_detec / hist_undetec
hist_ratio = np.nan_to_num(hist_ratio)

fig, ax = plt.subplots(figsize=(12, 8))

plt.bar(bin_edges[:-1] + step_size / 2, hist_ratio, width=step_size, color="tab:orange", alpha=0.7)
plt.xlim(bins_range[0], bins_range[-1])
plt.xlabel(f"{elem} in the corresponding dimension")
plt.ylabel("Detection / Undetection ratio")
plt.savefig(f'./images/11/detected_undetected_ratio.png', dpi=300)

# We can do the same thing also for the NEO classes!
detec_classes_df = detected_neo_df.groupby(by="NEOClass").count().reset_index()[["NEOClass", "Name"]]
detec_classes_df.rename(columns={"Name": "detec_count"}, inplace=True)

undetec_classes_df = undetected_neo_df.groupby(by="NEOClass").count().reset_index()[["NEOClass", "Name"]]
undetec_classes_df.rename(columns={"Name": "undetec_count"}, inplace=True)

merged_classes_df = pd.merge(detec_classes_df, undetec_classes_df, on="NEOClass")
merged_classes_df.loc[:, "detec_ratio"] = merged_classes_df["detec_count"] / merged_classes_df["undetec_count"]

print(merged_classes_df)

# Hmmm let's take a look at the distribution of the orbital elements

# And we only consider the smaller NEOs (H >= 25 mag)
absmag_limit = 25

# We display the detected and undetected NEOs
temp_detected_neo_df = detected_neo_df.loc[detected_neo_df["AbsMag_"] >= absmag_limit].copy()
temp_undetected_neo_df = undetected_neo_df.loc[undetected_neo_df["AbsMag_"] >= absmag_limit].copy()

# Additionally, our scatter plot will display the size of the NEOs by scaling the scatter points
temp_detec_l = (temp_detected_neo_df["AbsMag_"] - temp_detected_neo_df["AbsMag_"].min())
temp_detec_size = temp_detec_l * 1.0 / (temp_detec_l.max())

temp_undetec_l = (temp_undetected_neo_df["AbsMag_"] - temp_undetected_neo_df["AbsMag_"].min())
temp_undetec_size = temp_undetec_l * 1.0 / (temp_undetec_l.max())

# And not we plot the results
fig, ax = plt.subplots(figsize=(20, 8))

ax.scatter(temp_detected_neo_df["Perihel_AU"],
           temp_detected_neo_df["Ecc_"],
           s=temp_detec_size * 50,
           color="white",
           alpha=0.5,
           label="Detected NEOs")
ax.scatter(temp_undetected_neo_df["Perihel_AU"],
           temp_undetected_neo_df["Ecc_"],
           s=temp_undetec_size * 50,
           color="tab:red",
           alpha=0.5,
           label="Undetected NEOs")

# Layout and formatting
plt.xlabel("Perihel in AU")
plt.ylabel("Eccentricity")
plt.legend(fancybox=True, loc="lower left")
plt.xlim(0.2, 1.2)

plt.savefig(f'./images/11/orbital_elements_distribution.png', dpi=300)
