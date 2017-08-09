#Test methods of converting "list of lists" to string,
#for storage in SQL database.  Intend to use on ELEC data to speed up slicing,
#and avoid having to store all of the data in RAM.

import pandas as pd
import numpy as np
import sqlalchemy as sa
import json

d={'A':['a','b','c','None'], 
	 'B':[1,4,9,16],
	 'C':[ [[0,1],[2,3]],[[4,5],[6,7]],[[8,9],[10,11]],[[12,13],[14,15]]]
};

d1={'C':[ [[0,1],[2,3]],[[4,5],[6,7]],[[8,9],[10,11]],[[12,13],[14,15]]]
};

df=pd.DataFrame(d);

length=df.shape[0];
for i in range(0,length):
	df.loc[i,'C']=str(df.loc[i,'C'])

	#Use SQLite since only single user, read operations for this exploratory phase. 

engine=sa.create_engine('sqlite:///test.db')

metadata=sa.MetaData()
table_name='data'
data_db= sa.Table(table_name,metadata)
metadata.create_all(engine)

# data_types={'A':sa.types.TEXT,'B':sa.types.INTEGER,'C':sa.types.TEXT}
# data_types={'C':sa.types.TEXT}
df.to_sql('table_name',engine,index=False,if_exists='replace')#,dtype=data_types)

l1=('A','B','C','D')
for column_name in l1:
	table_name='data'
	column_type='TEXT'
	# com_str=str('CASEIF COL_LENGTH('+table_name+','+column_name+') IS NULL'
	# 						+' ALTER TABLE '+table_name+' ADD COLUMN '+column_name+' '+column_type )
	engine.execute(com_str)

#read out selected columns from the data under certain conditions  Huzzah!
df2 = pd.read_sql('SELECT A,C FROM data WHERE B==4',engine)

