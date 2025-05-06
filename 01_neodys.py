import re
import requests
import pathlib
import sqlite3

# Define the directory and name of the NEODyS data
raw_data_dir = pathlib.Path("raw_data/")
raw_data_file = pathlib.Path("neodys.cat")

raw_data_filepath = raw_data_dir / raw_data_file
print(f"Our NEODyS file path: {raw_data_filepath}")

pathlib.Path.mkdir(raw_data_dir, exist_ok=True)

# Get the current number of known NEOs
http_response = requests.get("https://newton.spacedys.com/neodys/index.php?pc=1.0")
html_content = http_response.content

# Extract the number of NEOs from a specific HTML position
neodys_nr_neos = int(re.findall(r"<b>(.*?) objects</b> in the NEODyS", str(html_content))[0])
print(f"Number of currently known NEOs: {neodys_nr_neos}")

# Download the NEODyS file and store it
response = requests.get("https://newton.spacedys.com/~neodys2/neodys.cat")
download_file_path = pathlib.Path(raw_data_filepath)
with download_file_path.open(mode="wb+") as file_obj:
    file_obj.write(response.content)

# Set a placeholder dictionary where the data will be stored
neo_dict = []

# Open the NEODyS file. Ignore the header (first 6 rows) and iterate through the file row-wise.
# Read the content and save it in the dictionary

with open(raw_data_filepath) as f_temp:
    neo_data = f_temp.readlines()[6:]
    for neo_data_line_f in neo_data:
        neo_data_line = neo_data_line_f.split()
        neo_dict.append(
            {
                "Name": neo_data_line[0].replace("'", ""),
                "Epoch_MJD": float(neo_data_line[1]),
                "SemMajAxis_AU": float(neo_data_line[2]),
                "Ecc_": float(neo_data_line[3]),
                "Incl_deg": float(neo_data_line[4]),
                "LongAscNode_deg": float(neo_data_line[5]),
                "ArgP_deg": float(neo_data_line[6]),
                "MeanAnom_deg": float(neo_data_line[7]),
                "AbsMag_": float(neo_data_line[8]),
                "SlopeParamG_": float(neo_data_line[9]),
            }
        )
print(f"Does the file contain the same number of NEOs as the NEODyS website? \n"
      f"{">Yes" if len(neo_dict) == neodys_nr_neos else ">No"}")

# Create the NEODyS SQLite database for our future project work
database_dir = pathlib.Path("databases/neos/")
database_file = pathlib.Path("neodys.db")
database_filepath = database_dir / database_file

# Create the directory
pathlib.Path.mkdir(database_dir, parents=True, exist_ok=True)

# Establish a connection to the database and set a cursor
neodys_db_con = sqlite3.connect(database_filepath)
neodys_db_cur = neodys_db_con.cursor()

# Create the main table
neodys_db_cur.execute(
    "CREATE TABLE IF NOT EXISTS main(Name TEXT PRIMARY KEY, "
    "Epoch_MJD FLOAT, "
    "SemMajAxis_AU FLOAT, "
    "Ecc_ FLOAT, "
    "Incl_deg FLOAT, "
    "LongAscNode_deg FLOAT, "
    "ArgP_deg FLOAT, "
    "MeanAnom_deg FLOAT, "
    "AbsMag_ FLOAT, "
    "SlopeParamG_ FLOAT)"
)
neodys_db_con.commit()

# Insert the raw data into the database
neodys_db_cur.executemany(
    "INSERT OR IGNORE INTO main(Name, "
    "Epoch_MJD, "
    "SemMajAxis_AU, "
    "Ecc_, "
    "Incl_deg, "
    "LongAscNode_deg, "
    "ArgP_deg, "
    "MeanAnom_deg, "
    "AbsMag_, "
    "SlopeParamG_) "
    "VALUES (:Name, "
    ":Epoch_MJD, "
    ":SemMajAxis_AU, "
    ":Ecc_, "
    ":Incl_deg, "
    ":LongAscNode_deg, "
    ":ArgP_deg, "
    ":MeanAnom_deg, "
    ":AbsMag_, "
    ":SlopeParamG_)",
    neo_dict,
)
neodys_db_con.commit()

# Add new columns in the main table
for col_name in ["Aphel_AU", "Perihel_AU"]:

    # SQL query for creating new columns
    sql_col_create = f"ALTER TABLE main ADD COLUMN {col_name} FLOAT"

    # Try to create a new column. If it exists, an error will raise. Pass this error.
    try:
        neodys_db_cur.execute(sql_col_create)
        neodys_db_con.commit()
    except sqlite3.OperationalError:
        pass

# Get orbital elements to compute the derived parameters
neodys_db_cur.execute("SELECT Name, SemMajAxis_AU, Ecc_ FROM main")

# Fetch the data
_neo_data = neodys_db_cur.fetchall()

# Iterate through the results, compute the derived elements and put them in a list of dictionaries
_neo_deriv_param_dict = []
for _neo_data_line_f in _neo_data:
    _neo_deriv_param_dict.append(
        {
            "Name": _neo_data_line_f[0],
            "Aphel_AU": (1.0 + _neo_data_line_f[2]) * _neo_data_line_f[1],
            "Perihel_AU": (1.0 - _neo_data_line_f[2]) * _neo_data_line_f[1],
        }
    )

# Insert the data into the main table
neodys_db_cur.executemany(
    "UPDATE main SET Aphel_AU = :Aphel_AU, Perihel_AU = :Perihel_AU "
    "WHERE Name = :Name",
    _neo_deriv_param_dict,
)
neodys_db_con.commit()

neodys_db_con.close()
