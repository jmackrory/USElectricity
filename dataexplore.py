#Read in JSON Data.
#Read in Electric System Bulk Operating Data
#Look at columns.

import pandas as pd
import numpy as np
import json

#endif

#need to convert date string to DateTime

# #Need to convert dates to date/time
# def make_df_timeindex(df):
# 	#get date_time_index based on period, start and end

# 	#Make a Pandas Series with DateTimeIndex.
# 	#Use Start, End, with final label from geoset_id.
# 	#Use Final label from Geoset_id which should have values in M,Q,A
# 	interval=df['f']

# 	if (interval=='M'):
# 		start_str=df['start']
# 		end_str=df['end']
# 		date_format='%Y%m'
# 	elif (interval=='Q'):
# 		start_str=quarter_to_month(df['start'])
# 		end_str=quarter_to_month(df['end'])
# 		date_format='%Y%m'
# 	elif (interval=='A'):
# 		start_str=df['start']
# 		end_str=df['end']
# 		date_format='%Y'

# 	print([start_str,end_str,date_format])
# 	start_date=pd.to_datetime(start_str,format=date_format)
# 	end_date=pd.to_datetime(end_str,format=date_format)
# 	date_indx=pd.date_range(start_date,end_date,freq=interval,closed='left')

# 	return date_indx

#Make a list of lists, with first sublist entry as time, second sublist entry is data
#into a pandas timeseries.  Extract the interval from the geoset ID, and use to construct 
#the Period Index.
def make_df_periodindex(df,interval):

	#make empty series
	indx=list()
	dat2=list()
	for item in df:
		print(item)
		indx.append(item[0])
		dat2.append(item[1])
	return pd.Series(dat2,index=pd.PeriodIndex(indx,freq=interval))


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

#Just use name for column labels, with common period index.  




#Write function to pick out states, and see mix of energy.
#Write function to pull out 

#Goal: Convert data to TimeSeries, with dates in index.
