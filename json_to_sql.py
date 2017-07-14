#Read in JSON Data.
#Read in Electric System Bulk Operating Data
#Export whole data frame to SQL database.   
import pandas as pd
import numpy as np
import sqlalchemy as sa
import cProfile, io, pstats, json,os

pr=cProfile.Profile()
pr.enable()

#Convert ELEC data to SQL
# elec_dat = pd.read_json('data/ELEC_short.txt',lines=True)
# elec_json = json.loads('data/ELEC_short.txt')

#Borrowed from Introducing Python pg 180.
fname='ELEC'
elec_file=open('data/'+fname+'.txt','r')
db_name=fname+'.db'
os.remove('data/'+db_name)
engine=sa.create_engine('sqlite:///data/'+db_name)
#
#make a new table.  
#Did this since SQL threw a fit when a new column showed up.
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

#while True:
d0=pd.DataFrame()
df_tot=pd.DataFrame()
i=0
for i in range(0,1000):
	line=elec_file.readline();
	#stop reading file.
	if not line:
		break
	i+=1
	#every thousand lines output that to a data frame.
	df=pd.read_json(line,lines=True);
	df_tot=pd.concat([df_tot,df]);
	#values ensure no tailing name/dtype.
	#use str() to protect with quotes, to just store the whole string in SQL, (which otherwise
	#gets confused by brackets and commas in data.
	if (i%100):
		df_tot.loc[:,'data']=str(df_tot.loc[:,'data'].values);
		#Use SQLite since only single user, read operations for this exploratory phase. 
		df_tot.to_sql(fname,engine,index=False,if_exists='append');
		df_tot=d0;

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
