# Goal is to convert ELEC.txt into a SQL database for easier querying.
# (Whole JSON file does not fit into memory).
# Used "split" to slit ELEC.txt into chunks with 10000 lines each.
# Each of those will be read in as a pandas dataframe.
# Will then check if each column in the dataframe exists in the SQL table.
# If not, use psycopg to create that column.

#Read in JSON Data.
#Read in Electric System Bulk Operating Data
#Export whole data frame to SQL database.
import pandas as pd
import numpy as np
import sqlalchemy
import psycopg2
import os
from psycopg2 import sql

# #make a new table.

#Make new database by logging into
def create_new_database(database_name):
    """create_new_database(database_name)

    Make a new SQL database with a given name.
    Uses the "master" database to start working from.

    database_name: name of the new database.

    """

    conn = psycopg2.connect(dbname='jonathan', host='localhost')
    #need elevated permissions in order to create a new database from within python, to automatically
    #commit changes.
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)  #automatically commit changes
    cur = conn.cursor()
    #create table
    t1 = sql.Identifier(database_name)
    q0 = sql.SQL("CREATE DATABASE {0}").format(t1)
    try:
        cur.execute(q0)
    except:
        print('Creating database: '+database_name+' failed.')
    cur.close()
    conn.close()
    return None

#Create fresh table in a database by dropping the old one, and putting a new blank one in.
def create_fresh_table(database_name, table_name, init_column='name'):
    """create_fresh_table(database_name, table_name,init_column='name')
    Makes a new SQL table (and overwrites any existing table).

    database_name:name of the SQL database to connect to
    table_name:name of SQL table to create
    init_column:initial name of the column to insert
    """
    conn = psycopg2.connect(dbname=database_name, host='localhost')
    conn.set_session(autocommit=True)
    cur = conn.cursor()
    t1 = sql.Identifier(table_name)

    #Drop Table to start from scratch.
    try:
        q_drop = sql.SQL("DROP TABLE {0}").format(t1)
        cur.execute(q_drop)
        conn.commit()
        print('DROPPED '+table_name)
    except:
        print('Could not DROP '+table_name)

    #Create Blank table with first column given by first column of data.
    try:
        c1 = sql.Identifier(init_column)
        q_create = sql.SQL("CREATE TABLE {0} ({1} TEXT)").format(t1, c1)
        cur.execute(q_create)
        conn.commit()
        print('CREATED '+table_name)
    except:
        print('Could not CREATE TABLE '+table_name)

    return conn, cur

#Make sure required columns are present.
def check_and_create_columns(table_name, cur, df):
    """check_and_create_columns(table_name, cur, df)
    Checks SQL table has columns for all of the columns
    in a given dataframe.

    table_name: SQL table name
    cur: SQLalchemy cursor to the connected SQL database
    df: pandas dataframe we want to insert into the SQL table.
    """
    for column_name in df.columns:
        t1 = sql.Identifier(table_name)
        c1 = sql.Identifier(column_name)
        try:
            #just check if there is a column with the right name.
            #allow no records since might have to start with empty table.
            q2 = sql.SQL("SELECT {1} FROM {0} LIMIT 0").format(t1, c1)
            cur.execute(q2)
            print('Success in retrieving column: '+column_name)
        except:
            print('Trying to read from column: '+column_name+' failed.')
            print('Trying to Add column: '+column_name)
            #if not then create that column.
            q3 = sql.SQL("ALTER TABLE {0} ADD COLUMN {1} TEXT").format(t1, c1)
            cur.execute(q3)
    return None


#Loop over splits, and upload them to the SQL database.
#Assumes that large data files have been split using something like
# "split -l 100 -d fname.txt split_data/fname"
# to break initial files into smaller chunks.
def put_data_into_sql(base_file_tag, table_name, cur, engine):
    """put_data_int_sql(base_file_tag,table_name,cur, engine)
    Load data that has been split into small files, first into data frame.
    Convert any series into strings to store the whole series at once.
    Then inject those into a SQL database.

    base_file_tag: tag for the split files
    table_name: SQL table to insert data into
    cur: sqlalchemy cursor to the SQL database
    engine: sqlalchemy engine to SQL database.

    """
    path = 'data/split_data/'
    flist = os.listdir(path)
    flist.sort()
    flist = flist[0:2]
    for fn in flist:
        if fn.find(base_file_tag) >= 0:
            print('Reading in :'+fn)
            #       fname_split=path+fname+str("%02d"%(i));
            #       print(fname_split)
            df = pd.read_json(path+fn, lines=True);
            #use str() to protect with quotes, to just store the whole string in SQL, (which otherwise
            #gets confused by brackets and commas in data and childseries).
        if 'data' in df.columns:
            df['data'] = df['data'].astype('str')
        ##Initially tried to protect childseries, and vertex which give connections between.
        if 'childseries' in df.columns:
            print('childseries in columns')
            df['childseries'] = df['childseries'].astype('str')
        if 'vertex' in df.columns:
            print('vertex in columns')
            df['vertex'] = df['vertex'].astype('str')

        check_and_create_columns(table_name, cur, df)
        df.to_sql(table_name, engine, index=False, if_exists='append');
    return None

def convert_data(df):
    """convert_data(df)
    Function to convert a dataframe with rows for each series, and data as a list of lists
    into a dataframe with a common time index, and one column for each series.
    (Other versions try

    Input:
    df: input dataframe

    Output
    data_array: output list of pandas Series

    """
    Nrows = len(df)
    print(Nrows)
    data_array = [];
    for i in range(0, Nrows):
        #check there's actually data there.
        #use next line since the read in dataframe has returned a string.
        print('Trying to Convert Data Series#', i)
        try:
            init_series = np.asarray(df.iloc[i]['data'])
            dat2 = init_series[:, 1].astype(float);
            f  =  df.iloc[i]['f']
            time_index = pd.DatetimeIndex(init_series[:, 0])
            s = pd.Series(dat2, index = time_index)
            data_array.append(s)
        except:
            data_array.append(np.nan)
            print('Skipping Series#', i)
    return data_array

def transpose_df(df, data_series):
    """transpose_df(df, dataseries)
    Transposes the dataframe use a time index, with name columns.

    df - initial dataframe with names as rows, data as long string
    data_series - list of timeseries with

    """
    names = df['name'].values
    #find union of DatetimeIndex's
    Tindex = data_series[0].index
    for i in range(len(data_series)):
        try:
            Tindex = Tindex.union(data_series[i].index)
        except:
            print('skipping index union on #', i)

    #join those together.
    df_new = pd.DataFrame(columns=names, index=Tindex)
    for i in range(len(df)):
        name = names[i]
        df_new[name] = data_series[i]
    return df_new


#Loop over splits, and upload them to the SQL database.
#Assumes that large data files have been split using something like
# "split -l 100 -d fname.txt split_data/fname"
# to break initial files into smaller chunks.
def transpose_all_data(base_file_tag):
    """transpose_all_data(base_file_tag)
    Load data that has been split into small files, first into data frame.
    Convert strings into series, and then combine into one big data frame.

    base_file_tag: tag for the split files

    """
    path = 'data/split_data/'
    flist = os.listdir(path)
    flist.sort()
    df_tot = pd.DataFrame()
    for fn in flist:
        if fn.find(base_file_tag) >= 0:
            print('Reading in :'+fn)
            #       fname_split=path+fname+str("%02d"%(i));
            #       print(fname_split)
            df = pd.read_json(path+fn, lines=True);
            #use str() to protect with quotes, to just store the whole string in SQL, (which otherwise
            #gets confused by brackets and commas in data and childseries).
            if ('data' in df.columns):
                series1 = convert_data(df)
                df2 = transpose_df(df, series1)
                df2 = df2.dropna(axis=1, how='all')
                if (len(df2)>0):
                    df_tot = df_tot.join(df2, how='outer')
    return df_tot

def transpose_df(df, data_series):
    """transpose_df(df, dataseries)
    Transposes a dataframe with many time series to use a common time index, with columns
    given by the names.

    """
    names = df['name'].values
    Tindex = data_series[0].index
    df_new = pd.DataFrame(columns=names, index=Tindex)

    for i in range(len(df)):
        name = names[i]
        df_new[name] = data_series[i]
    return df_new

if __name__ == '__main__':

    #Regular EBA into SQL table.
    # # #Borrowed from Introducing Python pg 180.
    # database_name='US_ELEC'
    # fname='EBA'
    # table_name=fname
    # engine=sqlalchemy.create_engine("postgresql+psycopg2://localhost/"+database_name)

    # #Make brand-new databases.
    # create_new_database(database_name)
    # #Make new table NB: Erases old table!
    # conn, cur=create_fresh_table(database_name, table_name)
    # put_data_into_sql(fname, table_name, cur, engine)

    # #Borrowed from Introducing Python pg 180.
    database_name='US_ELEC'
    fname = 'EBA'
    table_name = 'EBA_time'
    engine = sqlalchemy.create_engine("postgresql+psycopg2://localhost/"+database_name)

    #Make new table NB: Erases old table!
    conn, cur = create_fresh_table(database_name, table_name)
    #df_tot = transpose_all_data('EBA')
    df_small = df_tot.iloc[0:100, 0:5]
    colfixed = collist.str.replace("\(region\)", "- region")
    df_small.columns = colfixed
    check_and_create_columns(table_name, cur, df_small)
    df_small.to_sql(table_name, engine, if_exists='replace');

# def put_time_df_to_sql(df, table_name, engine):
#     df.to_sql(table_name, engine, index=False, if_exists='replace');
