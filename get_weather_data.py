# Download data sets from NOAA.
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
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
            #If more than one entry, just pick the first one.
            if (sum(msk)>=1):
                df_small=dataframe[msk].head(n=1)
                df_small['State']=state
                aircode_df=aircode_df.append(df_small)
            else:
                print('could not find city:'+city)
    aircode_df=aircode_df.rename(columns={'ICAO':'CALL'})
    return aircode_df

def make_airport_df():
    """make_airport_df
    Read in a list of global airports, and extract their ICAO codes.
    Then cross-reference with USAF-WBAN codes for the weather stations at these
    airports.  Return a dataframe of those codes/numbers for just a few of the biggest cities in 
    each state in the US.
    """
    #read in list of airports
    airport_df = pd.read_csv('data/airports.dat',skiprows=1,na_values='\\N')
    print(len(airport_df))
    #only keep US airports, and name,city, and ICAO codes
    msk2=airport_df['Country']=='United States'
    airport_df=airport_df[msk2][['Name','City','ICAO']]
    print(len(airport_df))

    airport_codes=get_airport_code(airport_df,biggest_cities)
    
    #now compare with stations from ISD database.
    isd_name_df=pd.read_fwf('data/ISD/isd-history.txt',skiprows=20)
    #also only keep airports still operational in time period.
    msk = isd_name_df['END']>20150000
    isd_name_df=isd_name_df[msk]
    isd_name_df=isd_name_df[['USAF','WBAN','CALL','LAT',"LON"]]
    
    #merge together.
    airport_codes2 = pd.merge(airport_codes,isd_name_df,on='CALL')
    
    #drop the entries with 999999. Presumed to be duplicates.
    msk1 = airport_codes2['USAF']!=999999
    msk2 = airport_codes2['WBAN']!=99999
    airport_codes2=airport_codes2[msk1&msk2]
    #make these codes integers.
    airport_codes2['USAF']=airport_codes2['USAF'].astype(int)
    airport_codes2['WBAN']=airport_codes2['WBAN'].astype(int)
    
    return airport_codes2

def plot_airports(air_df):
    """plot_airports(air_df)
    Plot the locations of the airports contained within air_df.
    Useful for eyeballing if there are systematic flaws in the locations 
    that made the cut.
    """
    # try:
    #     m=pickle.load(open('usstates.pickle','rb'))
    #     print('Loading Map from pickle')
    # except:
    #if not, remake the Basemap (costs lots of time)
    try:
        plt.figure()  
        print('Creating Fine BaseMap and storing with pickle')
        m=Basemap(projection='merc',llcrnrlon=-130,llcrnrlat=25,
                  urcrnrlon=-65,urcrnrlat=50,resolution='l', 
                  lon_0=-115, lat_0=35)
        m.drawstates()
        m.drawcountries()
        m.drawcoastlines()
        pickle.dump(m,open('usstates.pickle','wb'),-1)
    except:
        print(meh)
        #actually draw the map
    lons = air_df['LON'].values
    lats = air_df['LAT'].values
    m.scatter(lons,lats,latlon=True)
    plt.show()
    return

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
    url=base_url+isd_filename(yearstr,USAF,WBAN)
    try:
        wget.download(url,out='data/ISD')
    except:
        print('could not download data from city:',city,airport)

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
##actually download that data
#get_all_data(airport_codes2)

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
    df['state']=state
    df['city']=city
    df['region']=region_dict[state]
    #df=pd.read_fwf(filename,compression='gzip',na_values='-9999')
    
    return df

fn = 'data/ISD/702650-26407-2015.gz'
city='Blah'
state='MEH'

df=convert_isd_to_df(fn,city,state)

