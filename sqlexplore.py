#Read in JSON Data.
#Read in Electric System Bulk Operating Data
#Look at columns.

import pandas as pd
import numpy as np
import sqlalchemy
import psycopg2
from psycopg2 import sql

from sql_lib import create_conn_and_cur

database_name='US_ELEC'
fname='EBA_time'
table_name=fname
engine=sqlalchemy.create_engine("postgresql+psycopg2://localhost/"+database_name)

conn=psycopg2.connect(dbname=database_name,host='localhost')
#conn.set_session(autocommit=True)
cur = conn.cursor()

#Goal: Convert data to TimeSeries, with dates in index.
t1 = sql.Identifier(table_name)
query_text="SELECT name, series_id data FROM {0} WHERE name LIKE"
#List of names to recover
column_names=['name','series_id','data','interval','start','end','units']

#make safe SQL queries, with deired list of columns in "out_columns".
#Assume we are searching through name for entries with desired type of series, for particular states,
#as well as generation type.

def safe_sql_query(table_name, out_columns, match_names, freq):
    col_query=sql.SQL(' ,').join(map(sql.Identifier,out_columns))
    #make up categories to match the name by.
    namelist=[];
    for namevar in match_names:
        namelist.append(sql.Literal('%'+namevar+'%'))
        #join together these matches with ANDs to match them all
        name_query=sql.SQL(' AND name LIKE ').join(namelist)
    #Total SQL query to select desired columns with features 
    q1 = sql.SQL("SELECT {0} FROM {1} WHERE (name LIKE {2} AND f LIKE {3}) ").format(
        col_query,
        sql.Identifier(table_name),
        name_query,
        sql.Literal(freq))
    return(q1)

# def safe_sql_query(table_name,
# 		   out_columns,
# 		   series_type='Net generation', 
# 		   state='Oregon',
# 		   gen_type='all',
# 		   freq='M'):
# 	#make up categories to match the name by. 
# 	query_match_list=list()
# 	l1 = sql.Literal(series_type+' :%')
# 	l2 = sql.Literal('%: '+state+' :%')
# 	l3 = sql.Literal('%'+gen_type+'%')
# 	#join together these matches with ANDs to match them all
# 	like_query=sql.SQL(' AND name LIKE ').join([l1,l2,l3])

# 	#Total SQL query to select desired columns with features 
# 	q1 = sql.SQL("SELECT {} FROM {} WHERE (name LIKE {} AND f LIKE {}) ").format(
#         sql.SQL(' ,').join(map(sql.Identifier,out_columns)),
#         sql.Identifier(table_name),
#         like_query,
#         sql.Literal(freq))

# 	return(q1)


#Get a dataframe from SQL database for given psycopg2 cursor,
#with desired output columns.  
#Must select data based on series type, state, and type of generation.

def get_dataframe(cur, table_name, out_columns, match_names, freq):
    q = safe_sql_query(table_name,out_columns,match_names,freq)
    cur.execute(q);
    df0=cur.fetchall();
    df = pd.DataFrame(df0,columns=out_columns);
    return df

# def get_dataframe(cur,
# 		  out_columns,
# 		  table="ELEC",
# 		  series_type='Net Generation',
# 		  state='Oregon',
# 		  gen_type='solar',
# 		  freq='M'):
# 	q = safe_sql_query(table,out_columns,series_type,state,gen_type,freq)
# 	cur.execute(q);
# 	df0=cur.fetchall();
# 	df = pd.DataFrame(df0,columns=out_columns);
# 	return df

#Make a list of lists, with first sublist entry as time, second sublist entry is data
#into a pandas timeseries.  Extract the interval from the geoset ID, and use to construct 
#the Period Index.
def make_df_periodindex(series,interval):
	#make empty series
	series2=np.asarray(series);
	indx=series2[:,0];
	dat2=series2[:,1];
	# for item in series:
	# 	print(item)
	# 	indx.append(item[0])
	# 	dat2.append(item[1])
	return pd.Series(dat2,index=pd.PeriodIndex(indx,freq=interval))

#test_df['data2']=make_df_periodindex(test_df,'data')

#Initial readin of SQL dataframes returns 'data' as a string of 
# a list of lists.  
#This function goes row by row, converting that 'data' column
#into a new series, with datetimeindex in 'data2'
def convert_df(df):
	Nrows=len(df)
#	df['data2']=pd.Series()
	data_array=[];
	for i in range(0,Nrows):
		#check there's actually data there.
#		try:
		print('Making',i,'dataset')
		interval=df.loc[i,'f']
		#use next line since the read in dataframe has returned a string.
		init_series=eval(df.loc[i,'data'])
		data_series=make_df_periodindex(init_series,interval)
		data_array.append(data_series)
		#this next line does not work.  Might need to setup a multi-index?
		#Have hierarchy: row number, time-index?
		#df.loc[i,'data2']=data_series
		# except:
		# 	print('Failed to make dataset for '+df.loc[i,'name'])
		# 	print('Continuing to next')
	return data_array

# q=safe_sql_query(table_name="ELEC",out_columns=('name','series_id'),gen_type='solar')
# print(q.as_string(conn))
# cur.execute(q)

#Can Identify useful tags by splitting at colons":"
# out_col=('name','data','f')
# df=get_dataframe(cur,out_col,series_type='Net generation',state='Oregon',gen_type='solar');
