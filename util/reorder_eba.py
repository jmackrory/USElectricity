


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#load in initial dataframe
try :
    len(df)>0
except:
    df=pd.read_json('data/split_data/EBA00',lines=True);

#load in mixed dataframe with childseries to test that.
try :
    len(df26)>0
except:
    df26=pd.read_json('data/split_data/EBA27',lines=True);

#Initial readin of SQL dataframes returns 'data' as a string of a list of lists.  
#This function goes row by row, converting that 'data' column
#into a new series, with datetimeindex in 'data2'

def convert_data(df):
    """convert_data(df)
    Function to convert a dataframe with rows for each series, and data in a string or list of lists
    into a dataframe with a common time index, and one column for each series.

    Input: 
    df: input dataframe

    Output
    data_array: output list of pandas Series

    """
    Nrows=len(df)
    print(Nrows)
    data_array=[];
    for i in range(0,Nrows):
        #check there's actually data there.
        #use next line since the read in dataframe has returned a string.
        print('Trying to Convert Data Series#',i)
        try:
            init_series=np.asarray(df.iloc[i]['data'])
            dat2=init_series[:,1].astype(float);
            f = df.iloc[i]['f']        
            time_index=pd.DatetimeIndex(init_series[:,0])
            s=pd.Series(dat2,index=time_index)
            data_array.append(s)
        except:
            data_array.append(np.nan)
            print('Skipping Series#',i)
    return data_array
   
def transpose_df(df,data_series):
    """transpose_df(df,dataseries)
    Transposes the dataframe use a time index, with name columns.

    df - initial dataframe with names as rows, data as long string
    data_series - list of timeseries with 

    """
    names=df['name'].values
    #find union of DatetimeIndex's
    Tindex=data_series[0].index
    for i in range(len(data_series)):
        try:
            Tindex=Tindex.union(data_series[i].index)
        except:
            print('skipping index union on #',i)

    #join those together.
    df_new=pd.DataFrame(columns=names,index=Tindex)
    for i in range(len(df)):
        name=names[i]
        df_new[name]=data_series[i]
    return df_new

#Loop over splits, and upload them to the SQL database.
#Assumes that large data files have been split using something like
# "split -l 100 -d fname.txt split_data/fname"
# to break initial files into smaller chunks.
def transpose_all_data(base_file_tag):
    """transpose_all_data(base_file_tag)
    Load data that has been split into small files, first into data frame.
    Convert strings into series, and then combine into one big data frame.

    base_file_tag: tag for the split files

    """
    path='data/split_data/'
    flist=os.listdir(path)
    flist.sort()
    #flist=flist[0:2]
    df_tot=pd.DataFrame()
    for fn in flist:
        if fn.find(base_file_tag) >= 0:
            print('Reading in :'+fn)
            #       fname_split=path+fname+str("%02d"%(i));
            #       print(fname_split)
            df=pd.read_json(path+fn,lines=True);
            #use str() to protect with quotes, to just store the whole string in SQL, (which otherwise
            #gets confused by brackets and commas in data and childseries).
            if ('data' in df.columns):
                series1 = convert_data(df)
                df2=transpose_df(df,series1)
                df2=df2.dropna(axis=1,how='all')
                if (len(df2)>0):
                    df_tot=df_tot.join(df2,how='outer')
    return df_tot

df_tot=transpose_all_data('EBA')
#make new table.
#check if data exists. 
