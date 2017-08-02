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
fname='ELEC'
table_name=fname
engine=sqlalchemy.create_engine("postgresql+psycopg2://localhost/"+database_name)

conn=psycopg2.connect(dbname=database_name,host='localhost')
conn.set_session(autocommit=True)
cur = conn.cursor()

#Goal: Convert data to TimeSeries, with dates in index.
t1 = sql.Identifier(table_name)
query_text="SELECT name, series_id data FROM {0} WHERE name LIKE"
#List of names to recover
column_names=['name','series_id','data','interval','start','end','units']

#make safe SQL queries, with deired list of columns in "out_columns".
#Assume we are searching through name for entries with desired type of series, for particular states,
#as well as generation type.
def safe_sql_query(table_name,out_columns, series_type='Net generation', state='Oregon', gen_type='all'):
	#make up categories to match the name by. 
	query_match_list=list()
	l1 = sql.Literal(series_type+' :%')
	l2 = sql.Literal('%: '+state+' :%')
	l3 = sql.Literal('%'+gen_type+'%')
	#join together these matches with ANDs to match them all
	like_query=sql.SQL(' AND name LIKE ').join([l1,l2,l3])

	#Total SQL query to select desired columns with features 
	q1 = sql.SQL("SELECT {} FROM {} WHERE (name LIKE {} ) ").format(
        sql.SQL(' ,').join(map(sql.Identifier,out_columns)),
        sql.Identifier(table_name),
				like_query)

	return(q1)

#Get a dataframe from SQL database with desired name, and columns.
#Can select tpe of series, state, and type.
def get_dataframe(out_columns,cur,series_type='Net Generation', state='Oregon', gen_type='all'):
	q = safe_sql_query("ELEC",out_columns,series_type,state,gen_type)
	cur.execute(q);
	df0=cur.fetchall()
	df = pd.DataFrame(df0,columns=out_columns)
	return df

#Make a list of lists, with first sublist entry as time, second sublist entry is data
#into a pandas timeseries.  Extract the interval from the geoset ID, and use to construct 
#the Period Index.
def make_df_periodindex_np(series,interval):

	#make empty series
	series2=np.asarray(series);
	indx=series[:,0];
	dat2=series[:,1];
	# for item in series:
	# 	print(item)
	# 	indx.append(item[0])
	# 	dat2.append(item[1])
	return pd.Series(dat2,index=pd.PeriodIndex(indx,freq=interval))

#test_df['data2']=make_df_periodindex(test_df,'data')
		
#function to convert whole dataframe's data to 
def convert_df(df):
	Nrows=len(df)
	df['data2']=pd.Series()
	for i in range(0,Nrows):
		if (len(df['data']) > 1):
			print('Making',i,'dataset')
			interval=df.loc[i,'f']
			data_series=make_df_periodindex(df.loc[i,'data'],interval)
			df.loc[i,'data2']=data_series


# q=safe_sql_query(table_name="ELEC",out_columns=('name','series_id'),gen_type='solar')
# print(q.as_string(conn))
# cur.execute(q)



#identify useful fields. name, series_id.

#Try to select data from fields with names such as net generation and desired state.
#Possibly also by electricity source.  Nuclear/Solar/Wind/Gas/Coal etc.  

#Useful fields: Net Generation : state : type
#Also interesting: Average retail price of electricity.  
#Retail sales of electricity

#Can Identify useful tags by splitting at colons":"


