#Test methods of converting "list of lists" to string,
#for storage in SQL database.  Intend to use on ELEC data to speed up slicing,
#and avoid having to store all of the data in RAM.


import pandas as pd
import numpy as np
import sqlalchemy
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

# # #Use SQLite since only single user, read operations for this exploratory phase. 
engine=sqlalchemy.create_engine('sqlite:///test.db')
# data_types={'A':sqlalchemy.types.TEXT,'B':sqlalchemy.types.INTEGER,'C':sqlalchemy.types.TEXT}
# data_types={'C':sqlalchemy.types.TEXT}
df.to_sql('data',engine,chunksize=2,index=False,if_exists='replace')#,dtype=data_types)

#js_txt=json.loads('data/ELEC_short.txt')

df2 = pd.read_sql('data',engine)

