import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def convert_isd_to_df(filename,city,state):
    """
    convert_to_df(filename)
    
    Read in a automated weather stations data from file.
    Data is space separated columns, with format given in
    "isd-lite-format.txt".
    Converts to pandas dataframe using date/time columns as DateTimeIndex.
    with other variables 
    """

    """Format info:
    1: Year
    2: Month
    3: Day
    4: Hour 
    5: Temperature (x10) in celcius
    6: Dew point temperature (x10) in celcius
    7: Sea level pressure (x10 in hectopascals)
    8: Wind direction (degrees from north)
    9: Wind speed 
    10: Cloud Coverage (categorical)
    11: Precipitation for One Hour (x10, in mm)
    12: Precipitation total for Six hours (x10 in mm)

    All missing values are -9999.
    """
    #use fixed width format to read in (isd-lite-format has data format)
    col_names=['year','month','day','hour',
               'Temp','DewTemp','Pressure',
               'WindDir','WindSpeed','CloudCover',
               'Precip-1hr','Precip-6hr']
    df=pd.read_fwf(filename,compression='gzip',
                   na_values='-9999',names=col_names)
                   #parse_dates=[0,1,2,3],infer_datetime_format=True,
    times=pd.to_datetime({'year':df['year'],
                          'month':df['month'],
                          'day':df['day'],
                          'hour':df['hour']})
    df.index=pd.DatetimeIndex(times)

    #df=pd.read_fwf(filename,compression='gzip',na_values='-9999')
    
    return df

fn = 'data/ISD/702650-26407-2015.gz'
city='Blah'
state='MEH'

df=convert_isd_to_df(fn,city,state)
