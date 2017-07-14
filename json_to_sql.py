#Read in JSON Data.
#Read in Electric System Bulk Operating Data
#Export whole data frame to SQL database.   
import pandas as pd
import numpy as np
import sqlalchemy as sa
import cProfile, io, pstats, json,os

pr=cProfile.Profile()
pr.enable()

# #Borrowed from Introducing Python pg 180.
fname='ELEC'
db_name=fname+'.db'
try:
	os.remove('data/'+db_name)
	print('File: data/'+db_name+' was deleted.')
except:
	print('File: data/'+db_name+' does not exist')
engine=sa.create_engine('sqlite:///data/'+db_name)
# #
# #make a new table.  
# #Did this since SQL threw a fit when a new column showed up.
metadata=sa.MetaData()
ELEC_short= sa.Table(fname,metadata,
										 sa.Column('copyright',sa.String),
										 sa.Column('data',sa.String),
										 sa.Column('description',sa.String),
										 sa.Column('end',sa.String), 
										 sa.Column('f',sa.String),
										 sa.Column('geography',sa.String),
										 sa.Column('iso3166',sa.String),
										 sa.Column( 'last_updated',sa.String),
										 sa.Column('lat',sa.String),
										 sa.Column( 'latlon',sa.String),
										 sa.Column( 'lon',sa.String),
										 sa.Column( 'name',sa.String),
										 sa.Column('series_id',sa.String),
										 sa.Column( 'source',sa.String),
										 sa.Column('start',sa.String),
										 sa.Column('units',sa.String)
)

metadata.create_all(engine)

nfile=57;
for i in range(0,nfile):
	fname_split='data/split_dat/'+fname+str("%02d"%(i));
	#stop reading file.
	print(fname_split)
	df=pd.read_json(fname_split,lines=True);
	# 	#values ensure no tailing name/dtype.
	# 	#use str() to protect with quotes, to just store the whole string in SQL, (which otherwise
	# 	#gets confused by brackets and commas in data.
	df.loc[:,'data']=str(df.loc[:,'data'].values);
	df.loc[:,'childseries']=str(df.loc[:,'data'].values);
		#Use SQLite since only single user, read operations for this exploratory phase. 
	df.to_sql(fname_split,engine,index=False,if_exists='append');

# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
