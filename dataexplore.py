#Read in JSON Data.
#Read in Electric System Bulk Operating Data
#Look at columns.

import pandas as pd
import numpy as np
import json

#Read in ALL State-Level Bulk Electricity Data
#elec_dat = pd.read_json('data/ELEC.txt',lines=True)
#manifest_df = pd.read_json('data/manifest.txt')

#Used grep to split huge ELEC data-set into:
#coal, coke, gas, nuclear, petroleum,wind,solar,hydro,oil and state.

# # #Can further select out trim down.
if 'test_df' not in locals():
 state_elec_df = pd.read_json('data/ELEC_split/ELEC_state.txt',lines=True)
# # #keep what seem to be important non-zero columns.
 state_elec_trim=state_elec_df.loc[:,['name','geoset_id','geography','data','start','end']]

 # # #try to subset based on solar, wind (at this level)
 # # #Use data_frame.str.contains('solar').sum()  to find matches
 # solar_msk=state_elec_trim.name.str.contains('solar')
 # #Use Series.str.get_dummies(sep=':') to split name based on levels.
 # state_elec_names_split=state_elec_trim.name.str.get_dummies(sep=':')
 # #The above gives me a list of the available fields: much easier to understand 
 # #(even if I won't use it for this splitting)
 # #Need to search for solar, wind, oil, nuclear.
 #grab some test monthly data to play with and plot.  
 test_df = state_elec_trim.loc[1878]  #data on East North Central Monthly Solar
 
 #access data via this call
 test_dat = test_df['data']

#endif

#need to convert date string to DateTime

#Need to convert dates to date/time
def make_df_timeindex(df):
	#get date_time_index based on period, start and end

	#Make a Pandas Series with DateTimeIndex.
	#Use Start, End, with final label from geoset_id.
	#Use Final label from Geoset_id which should have values in M,Q,A
	interval=df['geoset_id'][-1]

	if (interval=='M'):
		start_str=df['start']
		end_str=df['end']
		date_format='%Y%m'
	elif (interval=='Q'):
		start_str=quarter_to_month(df['start'])
		end_str=quarter_to_month(df['end'])
		date_format='%Y%m'
	elif (interval=='A'):
		start_str=df['start']
		end_str=df['end']
		date_format='%Y'

	print([start_str,end_str,date_format])
	start_date=pd.to_datetime(start_str,format=date_format)
	end_date=pd.to_datetime(end_str,format=date_format)
	date_indx=pd.date_range(start_date,end_date,freq=interval,closed='left')

	return date_indx

#Make a list of lists, with first sublist entry as time, second sublist entry is data
#into a pandas timeseries.  Extract the interval from the geoset ID, and use to construct 
#the Period Index.
def make_df_periodindex(df,data_num):
	interval=df['geoset_id'][-1]
	#make empty series
	indx=list()
	dat2=list()
	for item in test_df[data_num]:
		print(item)
		indx.append(item[0])
		dat2.append(item[1])

	return pd.Series(dat2,index=pd.PeriodIndex(indx,freq=interval))

test_df['data2']=make_df_periodindex(test_df,'data')
		




#Write function to pick out states, and see mix of energy.
#Write function to pull out 

#Goal: Convert data to TimeSeries, with dates in index.
