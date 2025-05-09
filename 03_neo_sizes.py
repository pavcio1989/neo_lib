# Import modules
import pathlib
import sqlite3
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

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

print(neo_df.head(10))


# Now we create a function to compute the size of the NEOs
def comp_neo_diameter(abs_mag: float, albedo: float = 0.15):
    """
    Function to compute the diameter of NEOs based on their absolute magnitude and albedo. If no
    albedo is provided, a default value of 0.15 is assumed.

    The result is provided in km.

    Parameters
    ----------
    abs_mag : float
        The NEO's absolute magnitude.
    albedo : float, default = 0.15
        The NEO's albedo.
    """

    # Compute the diameter in km
    neo_diam_km = ((10.0 ** (-0.2 * abs_mag)) / (math.sqrt(albedo))) * 1329.0

    return neo_diam_km


# Let's convert the absolute magnitude to the corresponding sizes
neo_diam_array = np.array([round(comp_neo_diameter(k),2) for k in neo_df["AbsMag_"]])

# Print some statistics
print(f"Known minimum NEO diameter: {np.min(neo_diam_array)} km")
print(f"Known maximum NEO diameter: {np.max(neo_diam_array)} km")
print(f"Mean NEO diameter: {np.mean(neo_diam_array)} km")
print(f"Median NEO diameter: {np.median(neo_diam_array)} km")

# Some values appear to be ... weird. We check now the corresponding absolute magnitude
print(neo_df.sort_values(by="AbsMag_")[["Name", "AbsMag_"]].head(5))
print(neo_df.sort_values(by="AbsMag_")[["Name", "AbsMag_"]].tail(5))

# Define a histogram bins array
bins_range = np.arange(0, 0.26, 0.01)

# Let's set a dark background
plt.style.use('dark_background')
# Set a default font size for better readability
plt.rcParams.update({'font.size': 14})
# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
# Plot a histogram of the absolute magnitude distribution
ax.hist(neo_diam_array, bins=bins_range, color='tab:orange', alpha=0.7)
# Set labels for the x and y axes
ax.set_xlabel('NEO Diameter in km')
ax.set_ylabel('Number of NEOs')
# Limit the xlim
ax.set_xlim(0, 0.25)
# Set a grid
ax.grid(axis='both', linestyle='dashed', alpha=0.2)
plt.show()

# Compute a cumulative distribution of the absolute magnitude
neo_absmag_hist, bins_edge = np.histogram(neo_df["AbsMag_"], \
                                       bins=np.arange(10.0, 31.0, 1.0))
cumul_neo_absmag_hist = np.cumsum(neo_absmag_hist)
# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
# Create a scatter plot of the cumulative distribution.
ax.scatter(bins_edge[:-1]+1, cumul_neo_absmag_hist, color='tab:orange', alpha=0.7, \
           marker='o')
# Set labels for the x and y axes
ax.set_xlabel('NEO Absolute Magnitude')
ax.set_ylabel('Cumulative number of NEOs')
# Set a grid
ax.grid(axis='both', linestyle='dashed', alpha=0.2)
# Set a logarithmic y axis
ax.set_yscale('log')

plt.show()
