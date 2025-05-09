# Import modules
import pathlib
import sqlite3

# Now we create the NEODyS SQLite database for our future project work
database_dir = pathlib.Path("./databases/neos/")
database_file = pathlib.Path("neodys.db")
database_filepath = database_dir / database_file

# Create the directory
pathlib.Path.mkdir(database_dir, parents=True, exist_ok=True)

# Establish a connection to the database and set a cursor
neodys_db_con = sqlite3.connect(database_filepath)
neodys_db_cur = neodys_db_con.cursor()


def neo_class(sem_maj_axis_au: float,
              peri_helio_au: float,
              ap_helio_au: float) -> str:
    """Classify the NEO based on the orbital parameters.
    Depending on the semi-major axis, perihelion and / or aphelion a NEO can be classified as an
    Amor, Apollo, Aten or Atira.
    Parameters
    ----------
    sem_maj_axis_au : float
        Semi-major axis of the NEO. Given in AU
    peri_helio_au : float
        Perihelion of the NEO. Given in AU
    ap_helio_au : float
        Aphelion of the NEO. Given in AU
    Returns
    -------
    neo_type : str
        NEO class / type.
    References
    ----------
    -1- Link to the NEO classifiction schema: https://cneos.jpl.nasa.gov/about/neo_groups.html
    """
    # Determine the NEO class in an extensive if-else statement
    if (sem_maj_axis_au > 1.0) & (1.017 < peri_helio_au < 1.3):
        neo_type = 'Amor'

    elif (sem_maj_axis_au > 1.0) & (peri_helio_au < 1.017):
        neo_type = 'Apollo'

    elif (sem_maj_axis_au < 1.0) & (ap_helio_au > 0.983):
        neo_type = 'Aten'

    elif (sem_maj_axis_au < 1.0) & (ap_helio_au < 0.983):
        neo_type = 'Atira'

    else:
        neo_type = 'Other'

    return neo_type


# Testing the function
amor1221 = [1.9191, 1.0832, 2.7550]
apollo1862 = [1.4702, 0.64699, 	2.2935]
aten2062 = [0.9668, 0.7901, 1.1434]
atira163693 = [0.7411, 0.5024, 0.9798]

assert "Amor" == neo_class(*amor1221)
assert "Apollo" == neo_class(*apollo1862)
assert "Aten" == neo_class(*aten2062)
assert "Atira" == neo_class(*atira163693)


def create_col(sqlitecon : sqlite3.Connection,
               sqlitecur: sqlite3.Cursor,
               table: str,
               col_name: str,
               col_type: str) -> None:
    """
    Function to create new columns in tables.
    Parameters
    ----------
    sqlitecon : sqlite3.Connection
        Connection to the SQLite database.
    sqlitecur : sqlite3.Cursor
        Cursor that points to the SQLite database.
    sqlitetable : str
        Table name, where a new column shall be added.
    col_name : str
        Column name.
    col_type : str
        SQLite column type (FLOAT, INT, TEXT, etc.).
    """
    # Generic f-string that represents an SQLite command to alter a table (adding a new column
    # with its dtype).
    sql_col_create = f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"

    # Try to create a new column. If is exists an sqlite3.OperationalError will raise. Pass
    # this error.
    try:
        sqlitecur.execute(sql_col_create)
        sqlitecon.commit()
    except sqlite3.OperationalError:
        pass


# Add a new column in the main table
create_col(sqlitecon=neodys_db_con,
           sqlitecur=neodys_db_cur,
           table="main",
           col_name="NEOClass",
           col_type="Text")


# Get the orbital elements to compute the NEO class
neodys_db_cur.execute("SELECT Name, SemMajAxis_AU, Perihel_AU, Aphel_AU FROM main")

# Fetch the data
neo_data = neodys_db_cur.fetchall()

# Iterate through the results, compute the NEO class and put in into the results
neo_class_param_dict = []
for neo_data_row in neo_data:
    neo_class_param_dict.append(
        {
            "Name": neo_data_row[0],
            "NEOClass": neo_class(
                sem_maj_axis_au=neo_data_row[1],
                peri_helio_au=neo_data_row[2],
                ap_helio_au=neo_data_row[3]
            )
        }
    )

# print(f"NEO class dictionary: {neo_class_param_dict}")

# Insert the data into the main table
neodys_db_cur.executemany(
    "UPDATE main SET NEOClass = :NEOClass "
    "WHERE Name = :Name",
    neo_class_param_dict,
)
neodys_db_con.commit()
