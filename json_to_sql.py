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
    
    conn=psycopg2.connect(dbname='jonathan',host='localhost')
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
def create_fresh_table(database_name, table_name,init_column='name'):
    """create_fresh_table(database_name, table_name,init_column='name')
    Makes a new SQL table (and overwrites any existing table).

    database_name:name of the SQL database to connect to
    table_name:name of SQL table to create
    init_column:initial name of the column to insert
    """
    conn=psycopg2.connect(dbname=database_name,host='localhost')
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
        q_create = sql.SQL("CREATE TABLE {0} ({1} TEXT)").format(t1,c1)
        cur.execute(q_create)
        conn.commit()
        print('CREATED '+table_name)
    except:
        print('Could not CREATE TABLE '+table_name)

    return conn,cur

#Make sure required columns are present.
def check_and_create_columns(table_name,cur,df):
    """check_and_create_columns(table_name,cur,df)
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
            q2 = sql.SQL("SELECT {1} FROM {0} LIMIT 0").format(t1,c1)
            cur.execute(q2)
            print('Success in retrieving column: '+column_name)
        except:
            print('Trying to read from column: '+column_name+' failed.')
            print('Trying to Add column: '+column_name)
            #if not then create that column.  
            q3 = sql.SQL("ALTER TABLE {0} ADD COLUMN {1} TEXT").format(t1,c1)
            cur.execute(q3)
    return None

#Loop over splits, and upload them to the SQL database.
#Assumes that large data files have been split using something like
# "split -l 100 -d fname.txt split_data/fname"
# to break initial files into smaller chunks.
def put_data_into_sql(base_file_tag,table_name,cur,engine):
    """put_data_int_sql(base_file_tag,table_name,cur,engine)
    Load data that has been split into small files, first into data frame.
    Convert any series into strings to store the whole series at once.  
    Then inject those into a SQL database.  

    base_file_tag: tag for the split files
    table_name: SQL table to insert data into
    cur: sqlalchemy cursor to the SQL database
    engine: sqlalchemy engine to SQL database.

    """
    path='data/split_data/'
    flist=os.listdir(path)
    flist.sort()
    #flist=flist[0:2]
    for fn in flist:
        if fn.find(base_file_tag) >= 0:
            print('Reading in :'+fn)
            #       fname_split=path+fname+str("%02d"%(i));
            #       print(fname_split)
            df=pd.read_json(path+fn,lines=True);
            #use str() to protect with quotes, to just store the whole string in SQL, (which otherwise
            #gets confused by brackets and commas in data and childseries).
        if 'data' in df.columns:    
            df['data']=df['data'].astype('str')
        if 'childseries' in df.columns: 
            print('childseries in columns')
            df['childseries']=df['childseries'].astype('str')
        if 'vertex' in df.columns:  
            print('vertex in columns')
            df['vertex']=df['vertex'].astype('str')

        check_and_create_columns(table_name,cur,df)
        df.to_sql(table_name,engine,index=False,if_exists='append');        

# #Borrowed from Introducing Python pg 180.
database_name='US_ELEC'
fname='EBA'
table_name=fname
engine=sqlalchemy.create_engine("postgresql+psycopg2://localhost/"+database_name)

#Make brand-new databases.  
create_new_database(database_name)
##NB: Erases old table!
conn,cur=create_fresh_table(database_name,table_name)
put_data_into_sql(fname,table_name,cur,engine)
