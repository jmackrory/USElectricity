# Download data sets from NOAA.
#See weather_dataframe.py for converting this data into a combined dataframe.
#Same as get_weather_data.py - just has basemap stuff commented out.
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import wget
#from mpl_toolkits.basemap import Basemap
import pickle

# #Get API keys from JSON file
# with open('keys.json') as key_file:
#     keys=json.load(key_file)
#NCDC's call to website.

biggest_cities=[
    ['AL','Birmingham','Mobile','Huntsville'],
    ['AK','Anchorage','Fairbanks','Juneau'],
    ['AZ','Phoenix','Tucson','Mesa'],
    ['AR','Little Rock','Fort Smith','Fayetteville'],
    ['CA','Los Angeles','San Diego','San Jose'],
    ['CO','Denver','Colorado Springs','Aurora'],
    ['CT','Bridgeport','New Haven','Hartford'],
    ['DE','Wilmington','Dover','Newark'],
    ['FL','Jacksonville','Miami','Tampa'],
    ['GA','Atlanta','Augusta','Columbus'],
    ['HI','Honolulu','Hilo','Kailua'],
    ['ID','Boise','Nampa','Idaho Falls'],
    ['IL','Chicago','Aurora','Rockford'],
    ['IN','Indianapolis','Fort Wayne','Evansville'],
    ['IA','Des Moines','Cedar Rapids','Davenport'],
    ['KS','Wichita','Overland Park','Kansas City'],
    ['KY','Louisville','Lexington','Owensboro'],
    ['LA','New Orleans','Shreveport','Baton Rouge'],
    ['ME','Portland','Lewiston','Bangor'],
    ['MD','Baltimore','Frederick','Gaithersburg'],
    ['MA','Boston','Worcester','Springfield'],
    ['MI','Detroit','Grand Rapids','Warren'],
    ['MN','Minneapolis','St. Paul','Rochester'],
    ['MS','Jackson','Gulfport','Biloxi'],
    ['MO','Kansas City','St. Louis','Springfield'],
    ['MT','Billings','Missoula','Great Falls'],
    ['NE','Omaha','Lincoln','Bellevue'],
    ['NV','Las Vegas','Reno','Henderson'],
    ['NH','Manchester','Nashua','Concord'],
    ['NJ','Newark','Jersey City','Paterson'],
    ['NM','Albuquerque','Las Cruces','Rio Rancho'],
    ['NY','New York','Buffalo','Rochester'],
    ['NC','Charlotte','Raleigh','Greensboro'],
    ['ND','Fargo','Bismarck','Grand Forks'],
    ['OH','Columbus','Cleveland','Cincinnati'],
    ['OK','Oklahoma City','Tulsa','Norman'],
    ['OR','Portland','Salem','Eugene'],
    ['PA','Philadelphia','Pittsburgh','Allentown'],
    ['RI','Providence','Warwick','Cranston'],
    ['SC','Charleston','Columbia','North Charleston'],
    ['SD','Sioux Falls','Rapid City','Aberdeen'],
    ['TN','Memphis','Nashville','Knoxville'],
    ['TX','Houston','San Antonio','Dallas'],
    ['UT','Salt Lake City','West Valley City','Provo'],
    ['VT','Burlington','South Burlington','Rutland'],
    ['VA','Virginia Beach','Norfolk','Chesapeake'],
    ['WA','Seattle','Spokane','Tacoma'],
    ['WV','Charleston','Huntington','Parkersburg'],
    ['WI','Milwaukee','Madison','Green Bay'],
    ['WY','Cheyenne','Casper','Laramie']];


#try to map states to power producing regions.
#This is not quite correct, since some states are split between
#multiple regions (TN, MS, ND).  But will try as first attempt.
region_dict={
    'AL':'Southeast',
    'AK':'AK',
    'AZ':'Southwest',
    'AR':'Midwest',
    'CA':'California',
    'CO':'Northwest',
    'CT':'Northeast',
    'DE':'Northeast',
    'FL':'Florida',
    'GA':'Southeast',
    'HI':'HI',
    'ID':'Northwest',
    'IL':'Midwest',
    'IN':'Midwest',
    'IA':'Midwest',
    'KS':'Central',
    'KY':'Mid-Atlantic',
    'LA':'Midwest',
    'ME':'Northeast',
    'MD':'Mid-Atlantic',
    'MA':'Northeast',
    'MI':'Midwest',
    'MN':'Midwest',
    'MS':'Midwest',
    'MO':'Midwest',
    'MT':'Northwest',
    'NE':'Central',
    'NV':'Southwest',
    'NH':'Northeast',
    'NJ':'Mid-Atlantic',
    'NM':'Southwest',
    'NY':'New York',
    'NC':'Carolinas',
    'ND':'Central',
    'OH':'Mid-Atlantic',
    'OK':'Central',
    'OR':'Northwest',
    'PA':'Mid-Atlantic',
    'RI':'Northeast',
    'SC':'Carolinas',
    'SD':'Central',
    'TN':'Tennessee',
    'TX':'Texas',
    'UT':'Northwest',
    'VT':'New England',
    'VA':'Mid-Atlantic',
    'WA':'Northwest',
    'WV':'Mid-Atlantic',
    'WI':'Midwest',
    'WY':'Northwest'}


# #now make a dict of city names, and station locations.
# #Find allowed ID number corresponding to largest cities.
# #Note not all of these have entries.  
def get_airport_code(dataframe,city_list,depth=3):
    """get_airport_code(dataframe,city_list,depth=3)
    Extract the ICAO code/callsign for one airport in each city.
    Return a dataframe with the city,state and callsign.
    
    dataframe: initial dataframe with list of global airports, locations, ICAO callsigns.
    city_list: list of lists cities to find the callsigns for. Containts cities in states.
    depth: how many cities in each state to look for.  

    """
    aircode_df=pd.DataFrame()
    nrows=len(city_list)
    for i in range(0,nrows):
        for j in range(1,depth+1):
            city=city_list[i][j]
            state=city_list[i][0]
            msk=dataframe['City'].str.contains(city)
            #Need separate handling for Portland and Charleston.
            #(Ugly as sin, but should not break.)
            if (city =='Portland'):
                if (state=='OR'):
                    call='KPDX'
                elif (state=='ME'):
                    call='KPWM'
                msk=dataframe['ICAO'].str.contains(call)
            elif (city =='Charleston'):
                if (state=='SC'):
                    call='KCHS'
                elif (state=='WV'):                
                    call='KCRW'
                msk=dataframe['ICAO'].str.contains(call)
            #If more than one entry, just pick the first one.
            #check there is an entry
            if (sum(msk)==0):
                print('could not find city:'+city)
                df_small=pd.DataFrame()
            #otherwise pick the first.
            else:
                df_small=dataframe[msk].head(n=1)
                df_small['State']=state
            aircode_df=aircode_df.append(df_small)

    aircode_df=aircode_df.rename(columns={'ICAO':'CALL'})
    return aircode_df

def make_airport_df():
     """make_airport_df
     Read in a list of global airports, and extract their ICAO codes.
     Restrict then to US cities.
     """
     #read in list of airports
     airport_df = pd.read_csv('data/airports.dat',skiprows=1,na_values='\\N')
     #only keep US airports, and name,city, and ICAO codes
     msk2=airport_df['Country']=='United States'
     airport_df=airport_df[msk2][['Name','City','ICAO']]
     
     airport_codes=get_airport_code(airport_df,biggest_cities)

     return airport_codes

def read_isd_df():
    """make_airport_df
    Read in list of weather stations and USAF-WBAN codes for the weather stations.  
    Trim to only stations that have operated since 2015.
    """

    #now compare with stations from ISD database.
    isd_name_df=pd.read_fwf('data/ISD/isd-history.txt',skiprows=20)
    #also only keep airports still operational in time period.
    msk = isd_name_df['END']>20150000
    isd_name_df=isd_name_df[msk]
    isd_name_df=isd_name_df[['USAF','WBAN','CALL','LAT',"LON"]]
    return isd_name_df

def merge_air_isd(airport_codes,isd_name_df):
    """merge_air_isd_df
    
    Merge the airport and weather data frames on name.
    Trim out duplicates.

    """
    airport_total = pd.merge(airport_codes,isd_name_df,on='CALL')
    
    #drop any duplicated entries. (i.e. multiple at same airport)
    msk3 = airport_total['CALL'].duplicated().values
    print('Duplicated values for:')
    print(airport_total[msk3][['CALL','City']])
    airport_total=airport_total[~msk3]
    # msk1 = airport_codes2['USAF']!=999999
    # msk2 = airport_codes2['WBAN']!=99999
    # airport_codes2=airport_codes2[msk1&msk2]
    
    #make these codes integers.
    airport_total['USAF']=airport_total['USAF'].astype(int)
    airport_total['WBAN']=airport_total['WBAN'].astype(int)
    
    return airport_total

# def plot_airports(air_df):
#     """plot_airports(air_df)
#     Plot the locations of the airports contained within air_df.
#     Useful for eyeballing if there are systematic flaws in the locations 
#     that made the cut.
#     """
#     # try:
#     #     m=pickle.load(open('usstates.pickle','rb'))
#     #     print('Loading Map from pickle')
#     # except:
#     #if not, remake the Basemap (costs lots of time)
#     try:
#         plt.figure()  
#         print('Creating Fine BaseMap and storing with pickle')
#         m=Basemap(projection='merc',llcrnrlon=-130,llcrnrlat=25,
#                   urcrnrlon=-65,urcrnrlat=50,resolution='l', 
#                   lon_0=-115, lat_0=35)
#         m.drawstates()
#         m.drawcountries()
#         m.drawcoastlines()
#         pickle.dump(m,open('usstates.pickle','wb'),-1)
#     except:
#         print(meh)
#         #actually draw the map
#     lons = air_df['LON'].values
#     lats = air_df['LAT'].values
#     m.scatter(lons,lats,latlon=True)
#     plt.show()
#     return None

#now download the data from NOAA:
def wget_data(USAF,WBAN,yearstr,city,airport):
    """wget_data(USAF,WBAN,yearstr,city,airport)
    Download automated weather station data from NOAA for a given year at a given airport.
    
    USAF: USAF 6 digit code for airport.
    WBAN: NOAA code for weather station at airport
    yearstr: a string containing the 4 digit year.
    city: city the airport is located in
    airport: Name of the airport.
    """
    base_url='ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/'
    file_name=isd_filename(yearstr,USAF,WBAN)
    url=base_url+file_name
    try:
        print('\n trying: {}'.format(url))
        wget.download(url,out='data/ISD')
    except:
        print('\n could not download data from city:',city,airport)
    return None

def isd_filename(yearstr,USAF,WBAN):
    """ isd_filename(yearstr,USAF,WBAN)
    Make filename corresponding to zipped file names used in ISD database.
    """
    #put in some padding {:0>5} for shorter codes.
    fn="{0}/{1}-{2:0>5}-{0}.gz".format(yearstr,str(USAF),str(WBAN))
    return fn
        
#download weather data for all of the airports specified in aircode
def get_all_data(aircode,years=['2015','2016','2017']):
    """get_all_data(aircode,years=['2015','2016','2017'])
    Download the data for all airports we could find weather stations for in desired cities.

    aircode: datafram containing airport codes, NOAA station numbers, airports
    years: array of strings for the years to seek data.
    """
    for yearstr in years:
        for i in range(len(aircode)):
            ap = aircode.iloc[i]
            usaf=ap['USAF']
            wban=ap['WBAN']
            city=ap['City']
            airport=ap['Name']
            wget_data(usaf,wban,yearstr,city,airport)
    return None

#now read it in, convert to time-series.

def convert_isd_to_df(filename,city,state):
    """
    convert_to_df(filename)
    
    Read in a automated weather stations data from file.
    Data is space separated columns, with format given in
    "isd-lite-format.txt".
    Converts to pandas dataframe using date/time columns as DateTimeIndex.
    Format info:
    1: Year
    2: Month
    3: Day
    4: Hour 
    5: Temperature (x10) in celcius
    6: Dew point temperature (x10) in celcius
    7: Sea level pressure (x10 in hectopascals)
    8: Wind direction (degrees from north)
    9: Wind speed (x10 in meters per second)
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
                   na_values=['-9999','999'],names=col_names)
    city_st=city+', '+state
    df['city']=city
    df['state']=state
    df['city, state']=city_st
    df['region']=region_dict[state]        
    #make a time index.
    times=pd.to_datetime({'year':df['year'],
                          'month':df['month'],
                          'day':df['day'],
                          'hour':df['hour']})
    Tindex=pd.DatetimeIndex(times)
    df.index=Tindex
    #df.index=pd.MultiIndex.from_product([Tindex,[city_st]])
    #delete those columns
    df=df.drop(labels=['year','month','day','hour'],axis=1)
    return df

# fn = 'data/ISD/702650-26407-2015.gz'
# city='Blah'
# state='OR'
            
# df=convert_isd_to_df(fn,city,state)

def convert_state_isd(air_df,ST):
    """convert_all_isd(air_df)
    convert the weather files for a particular state into 
    one big data frame.
    """
    data_dir='data/ISD/'
    Tindex=pd.DatetimeIndex(start='2015-07',end='2017-11',freq='h')
    df_tot=pd.DataFrame(index=Tindex)
    #select out only the entries for the desired state.
    msk = air_df['State']==ST
    air_msk=air_df[msk]
    for i in range(len(air_msk)):
        for yearstr in ['2015','2016','2017']:
            ap = air_msk.iloc[i]
            usaf=ap['USAF']
            wban=ap['WBAN']
            city=ap['City']
            state=ap['State']
            file_name="data/ISD/{1}-{2:0>5}-{0}.gz".format(yearstr,str(usaf),str(wban))
            df=convert_isd_to_df(file_name,city,state)
            df_tot=df_tot.append(df)
        print('done with {}'.format(ap['Name']))
    return df_tot

def convert_all_isd(air_df):
    """convert_all_isd(air_df)
    convert all the weather files for all stations and all years into 
    one big data frame.
    """
    data_dir='data/ISD/'
    Tindex=pd.DatetimeIndex(start='2015-07',end='2017-11',freq='h')
    df_tot=pd.DataFrame(index=Tindex)
    nmax=len(air_df)
    for i in range(nmax):
        for yearstr in ['2015','2016','2017']:
            ap = air_df.iloc[i]
            usaf=ap['USAF']
            wban=ap['WBAN']
            city=ap['City']
            state=ap['State']
            file_name="data/ISD/{1}-{2:0>5}-{0}.gz".format(yearstr,str(usaf),str(wban))
            df=convert_isd_to_df(file_name,city,state)
            df_tot=df_tot.append(df)
        print('done with {}'.format(ap['Name']))
    return df_tot

# def make_weather_multiindex(air_df):
#     #Tindex=pd.DatetimeIndex(start='2015-07',end='2017-11',freq='h')
#     Tindex=pd.DatetimeIndex(start='2015-07',end='2016-03',freq='m')    
#     city_list=list()
#     nmax=4;#len(air_df)
#     for i in range(nmax):
#         ap = air_df.iloc[i]
#         city=ap['City']
#         state=ap['State']
#         city_ST=city+', '+state
#         city_list.append(city_ST)
#     joint_index=pd.MultiIndex.from_product([Tindex,city_list])
#     return joint_index

#make dataframes with codes.
try:
    air_df=pd.read_csv('data/air_code_df.gz')
except:
    airport_codes=make_airport_df()
    isd_names=read_isd_df()
    air_df=merge_air_isd(airport_codes,isd_names)
    #write output to csv
    air_df.to_csv('data/air_code_df.gz',compression='gzip',header=True)

#ind=make_weather_multiindex(air_df)
    
##actually download the data from 2015-2017 from the stations listed in air_code_df.  (takes a few minutes)
#get_all_data(air_df)

# d0=convert_all_isd(air_df)
# # #converted file to csv to save time.
# d0.to_csv('data/airport_weather.gz',compression='gzip',header=True)
