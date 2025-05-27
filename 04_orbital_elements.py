# Import modules
import pathlib
import sqlite3
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


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

# print(neo_df.head(5))

sns.set_theme()
# Creating quickly an overview plot with seaborn
# Use bins=... to set the number of bins or
# bin_width=... to set the width of each bins. Quite easy!
sns_plt = sns.displot(neo_df, x="AbsMag_")
plt.xlim(0, 35)
plt.savefig(f'./images/04/absmag_histogram.png', dpi=300)

sns_plt = sns.displot(neo_df.loc[neo_df["NEOClass"] != "Other"],
                      x="AbsMag_",
                      hue="NEOClass",
                      element="step",
                      stat="probability",
                      common_norm=False,
                      binwidth=1)
plt.xlim(15, 30)
plt.savefig(f'./images/04/absmag_neoclass_histograms.png', dpi=300)

sns.displot(neo_df.loc[(neo_df["NEOClass"] != "Other") & (neo_df["SemMajAxis_AU"] < 3.0)],
            x="SemMajAxis_AU",
            hue="NEOClass",
            kind="kde",
            multiple="stack")
plt.xlim(0.5, 3)
plt.savefig(f'./images/04/semimajor_neoclass_kde.png', dpi=300)

neo_df_filtered = neo_df.loc[
    (neo_df["NEOClass"] != "Other")
    & (neo_df["SemMajAxis_AU"] <= 5)
    & (neo_df["AbsMag_"] <= 30)
    & (neo_df["AbsMag_"] >= 15)
    & (neo_df["Ecc_"] <= 2.5)
    & (neo_df["Incl_deg"] <= 150)
    & (neo_df["Perihel_AU"] >= 0.0)
].copy()
neo_df_filtered = neo_df_filtered[["SemMajAxis_AU",
                                   "Ecc_",
                                   "Incl_deg",
                                   "AbsMag_",
                                   "Perihel_AU",]].copy()

print(neo_df_filtered.describe())

g = sns.PairGrid(neo_df_filtered)
g.map_upper(sns.histplot, bins=100)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True, bins=100)
plt.savefig(f'./images/04/histogram_pairplots.png', dpi=300)
