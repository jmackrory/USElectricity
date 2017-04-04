#Read in JSON Data.
#Read in Electric System Bulk Operating Data
#Look at columns.

import pandas as pd
import numpy as np
import json

#Read in State-Level Bulk Electricity Data
#elec_dat = pd.read_json('data/ELEC.txt',lines=True)
#manifest_df = pd.read_json('data/manifest.txt')

#Used grep to split huge ELEC data-set into:
#coal, coke, gas, nuclear, petroleum,wind,solar,hydro,oil and state.

#Can further select out trim down.
state_elec_df = pd.read_json('data/ELEC_split/ELEC_state.txt',lines=True)

#keep what seem to be important non-zero columns.
state_elec_trim=state_elec_df.loc[:,['name','geoset_id','geography','data','start','end']]

#try to subset based on solar, wind (at this level)

#Use data_frame.str.contains('solar').sum()  to find matches

#Use Series.str.get_dummies(sep=':') to split name based on levels.
state_elec_names_split=state_elec_trim.name.str.get_dummies(sep=':')

#The above gives me a list of the available fields: much easier to understand 
#(even if I won't use it for this splitting)
#Need to search for solar, wind, oil, nuclear.

#grab some test quarterly data to play with and plot.  
test_df = state_elec_trim[14043]  #data on New England nuclear.

#access data via this call
test_dat = test_df['data']

#need to convert date string to DateTime



#Need to convert dates to date/time
def change_date_format():
	#check length of string.
	#possibilities: yyyymm  or yyyyQi
	#can tell this from geoset_id tag.
	#  if last_char==A:
	#no change
	#  elif last_char==M:
	#DateTime with monthly
	# elif last_char==Q:
	#DateTime with 3-monthly
	


#Write function to pick out states, and see mix of energy.

#Write function to pull out 
