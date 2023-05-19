# sql_lib.py
#
# Bunch of functions for creating new PostGRESQL database using psycopg2
# Also some functions for querying that table and returning
# a pandas dataframe
#
# Functions for making new table:
# - create_new_database
# - create_fresh_table
# - check_and_create_columns
#
# Functions for calling the table:
# - safe_sql_query
# - get_column_query
# - get_dataframe

import psycopg2
import psycopg2.sql as sql
import pandas as pd


# Make new database by logging into
# OLD - not great.
def create_new_database(database_name):
    conn = psycopg2.connect(dbname="jonathan", host="localhost")
    # need elevated permissions in order to create a new database from within python, to automatically
    # commit changes.
    conn.set_isolation_level(
        psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT
    )  # automatically commit changes
    cur = conn.cursor()
    # create table
    t1 = sql.Identifier(database_name)
    q0 = sql.SQL("CREATE DATABASE {0}").format(t1)
    try:
        cur.execute(q0)
    except Exception as e:
        print(repr(e))
        print("Creating database: " + database_name + " failed.")
    cur.close()
    conn.close()
    return None


# make new connection and cursor for direct connection to SQL database via psycopg
def create_conn_and_cur(database_name, table_name):
    conn = psycopg2.connect(dbname=database_name, host="localhost")
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    return conn, cur


# Create fresh table in a database by dropping the old one, and putting a new blank one in.
def create_fresh_table(database_name, table_name, init_column="init"):
    conn = psycopg2.connect(dbname=database_name, host="localhost")
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    t1 = sql.Identifier(table_name)

    # Drop Table to start from scratch.
    try:
        q_drop = sql.SQL("DROP TABLE {0}").format(t1)
        cur.execute(q_drop)
        conn.commit()
        print("DROPPED " + table_name)
    except Exception as e:
        print(repr(e))
        print("Could not DROP " + table_name)

    # Create Blank table with first column given by first column of data.
    try:
        c1 = sql.Identifier(init_column)
        q_create = sql.SQL("CREATE TABLE {0} ({1} TEXT)").format(t1, c1)
        cur.execute(q_create)
        conn.commit()
        print("CREATED " + table_name)
    except Exception as er:
        print(repr(er))
        print("Could not CREATE TABLE " + table_name)

    return conn, cur


# Make sure required columns are present.
def check_and_create_columns(table_name, cur, df):
    for column_name in df.columns:
        t1 = sql.Identifier(table_name)
        c1 = sql.Identifier(column_name)
        try:
            # just check if there is a column with the right name.
            # allow no records since might have to start with empty table.
            q2 = sql.SQL("SELECT {1} FROM {0} LIMIT 0").format(t1, c1)
            cur.execute(q2)
            print("Success in retrieving column: " + column_name)
        except Exception as e:
            print(repr(e))
            print("Trying to read from column: " + column_name + " failed.")
            print("Trying to Add column: " + column_name)
            # if not then create that column.
            q3 = sql.SQL("ALTER TABLE {0} ADD COLUMN {1} TEXT").format(t1, c1)
            cur.execute(q3)
    return None


# make SQL queries, with desired list of columns in "out_columns".
# Assume we are searching through name for entries with desired type of series, for particular states,
# as well as generation type.
def safe_sql_query(table_name, out_columns, match_names, freq):
    """safe_sql_query(table_name, out_column, match_names, freq)
    Extract a set of columns where the name matches certain critera.

    Input:
    table_name - name for table
    out_columns - list of desired columns
    match_names - desired patterns that the name must match.  (All joined via AND)
    freq   - desired frequency

    Return:
    sql query to carry out desired command.
    """

    col_query = sql.SQL(" ,").join(map(sql.Identifier, out_columns))
    # make up categories to match the name by.
    namelist = []
    for namevar in match_names:
        namelist.append(sql.Literal("%" + namevar + "%"))
        # join together these matches with ANDs to match them all
        name_query = sql.SQL(" AND name LIKE ").join(namelist)
    # Total SQL query to select desired columns with features
    q1 = sql.SQL("SELECT {0} FROM {1} WHERE (name LIKE {2} AND f LIKE {3}) ").format(
        col_query, sql.Identifier(table_name), name_query, sql.Literal(freq)
    )
    return q1


def get_column_query(table_name, out_column):
    """get_column_query(table_name, out_column)
    Return SQL query to extract 'out_column' from 'table_name'
    """
    # make up categories to match the name by.
    # Total SQL query to select desired columns with features
    q1 = sql.SQL("SELECT {0} FROM {1}").format(
        sql.Identifier(out_column), sql.Identifier(table_name)
    )
    return q1


# Get a dataframe from SQL database for given psycopg2 cursor,
# with desired output columns.
# Must select data based on series type, state, and type of generation.
def get_dataframe(cur, table_name, out_columns, match_names, freq):
    """get_dataframe(cur, table_name, out_columns, match_names, freq)
    Generate pandas dataframe from calling SQL database.
    Dataframe will contain 'out_columns', in cases where the names
    contain all of the entries in 'match_names'

    Input: cur - psycopg2 cursor connected to database
    table_name -SQL table name
    out_columns - columns to extract from SQL
    match_names - list of strings that the 'name' must match
    freq      - desired frequency

    Output:
    df  - pandas Dataframe
    """

    q = safe_sql_query(table_name, out_columns, match_names, freq)
    cur.execute(q)
    df0 = cur.fetchall()
    df = pd.DataFrame(df0, columns=out_columns)
    return df
